"""Explainability + uncertainty for VTMT-Net.

Combined explanation = Grad-CAM (3D, bottleneck) x vessel prior  +  vessel-graph
attention  +  vessel-gated clot map  ->  one "final explainable PE map".
"""
import numpy as np
import torch
import torch.nn.functional as F

from model import burden_index


class GradCAM3D:
    def __init__(self, model, layer=None):
        self.model = model
        layer = layer or model.cam_layer
        self.acts = self.grads = None
        layer.register_forward_hook(self._fwd)

    def _fwd(self, mod, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, image, nodes, adj, node_mask, target="det", plus=False):
        """target: 'det' | 'rvlv' | 'loc:k' | 'oi'. Returns CAM (B,1,D,H,W) in [0,1]."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            image = image.clone().requires_grad_(False)
            out = self.model(image, nodes, adj, node_mask)
            if target.startswith("loc:"):
                score = out["loc"][:, int(target.split(":")[1])]
            else:
                score = out[target]
            score.sum().backward()
        A, G = self.acts.detach(), self.grads.detach()
        if plus:  # Grad-CAM++ weights
            g2, g3 = G ** 2, G ** 3
            alpha = g2 / (2 * g2 + A.sum((2, 3, 4), keepdim=True) * g3 + 1e-8)
            wts = (alpha * F.relu(G)).sum((2, 3, 4), keepdim=True)
        else:
            wts = G.mean((2, 3, 4), keepdim=True)
        cam = F.relu((wts * A).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[2:], mode="trilinear", align_corners=False)
        mn = cam.flatten(1).min(1).values.view(-1, 1, 1, 1, 1)
        mx = cam.flatten(1).max(1).values.view(-1, 1, 1, 1, 1)
        return (cam - mn) / (mx - mn + 1e-8)


def node_attention_importance(node_attn, node_mask):
    """How much attention each vessel node receives (column sum), normalised to [0,1]."""
    if node_attn is None:
        return None
    s = (node_attn * node_mask[:, :, None].float()).sum(1) * node_mask.float()
    return s / s.max(1, keepdim=True).values.clamp(min=1e-8)


@torch.no_grad()
def mc_dropout_predict(model, image, nodes, adj, node_mask, n=10):
    """Predictive mean / std of PE probability, location probs and burden index (MC-dropout)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout,)):
            m.train()
    det, loc, bi = [], [], []
    for _ in range(n):
        o = model(image, nodes, adj, node_mask)
        det.append(torch.sigmoid(o["det"]))
        loc.append(torch.sigmoid(o["loc"]))
        bi.append(burden_index(torch.sigmoid(o["node_logit"]), nodes, node_mask))
    model.eval()
    st = lambda xs: (torch.stack(xs).mean(0), torch.stack(xs).std(0))
    return {"det": st(det), "loc": st(loc), "burden_index": st(bi)}


@torch.no_grad()
def explain_study(model, cam_engine, image, nodes, adj, node_mask, loc_names, bi_cuts=(0.15, 0.40)):
    """Single-study explainable report (batch size 1). Returns dict of arrays + summary."""
    out = model(image, nodes, adj, node_mask)
    cam = cam_engine(image, nodes, adj, node_mask, "det")       # re-enables grad internally
    vessel = image[:, 1:2]
    gated = out["p"]
    final = cam * vessel * (0.5 + 0.5 * gated / gated.max().clamp(min=1e-6))
    imp = node_attention_importance(out["node_attn"], node_mask)
    nprob = torch.sigmoid(out["node_logit"])
    bi = float(burden_index(nprob, nodes, node_mask)[0])
    cat = "low" if bi < bi_cuts[0] else ("moderate" if bi < bi_cuts[1] else "high")
    top = torch.argsort(nprob[0] * node_mask[0], descending=True)[:5].tolist()
    return {
        "pe_probability": float(torch.sigmoid(out["det"])[0]),
        "location_probability": {n: float(p) for n, p in zip(loc_names, torch.sigmoid(out["loc"])[0])},
        "burden_index": bi, "burden_category": cat,   # cut-offs are placeholders: calibrate on validation data
        "rvlv_ge_1_probability": float(torch.sigmoid(out["rvlv"])[0]),
        "top_vessel_nodes": [{"node": j, "xyz_norm": nodes[0, j, 3:6].tolist(), "radius_norm": float(nodes[0, j, 0]),
                              "clot_prob": float(nprob[0, j]),
                              "attention": float(imp[0, j]) if imp is not None else None} for j in top],
        "maps": {"gradcam": cam[0, 0].cpu().numpy(), "clot_prob_gated": gated[0, 0].cpu().numpy(),
                 "final_explainable": final[0, 0].cpu().numpy()},
    }
