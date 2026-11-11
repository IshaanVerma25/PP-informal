"""Multi-task loss with weak-supervision support.

L = w_seg*L_seg + w_cl*L_clDice + w_ves*L_vessel + w_mil*L_mil + w_slice*L_slice
    + w_node*L_node + w_det*L_det + w_loc*L_loc + w_q*L_quant + w_cons*L_cons + w_sev*L_sev

Every term returns None when its labels are absent from the batch, so the same
loss works for (a) fully annotated data (masks), (b) RSNA-STR-style data with only
study-level + slice-level labels, and (c) mixtures.

Label conventions in `batch` (unknown = -1, or has_mask=False):
  seg (B,1,D,H,W) + has_mask (B,) | det (B,) | loc (B,n_loc) | slice_label (B,D)
  node_label (B,N) | oi (B,) | burden (B,) | rvlv_hi (B,)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_erode(x):
    p1 = -F.max_pool3d(-x, (3, 1, 1), 1, (1, 0, 0))
    p2 = -F.max_pool3d(-x, (1, 3, 1), 1, (0, 1, 0))
    p3 = -F.max_pool3d(-x, (1, 1, 3), 1, (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_open(x):
    return F.max_pool3d(_soft_erode(x), 3, 1, 1)


def soft_skeleton(x, iters=3):
    skel = F.relu(x - _soft_open(x))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cl_dice(prob, target, iters=3, eps=1e-6):
    """Soft centerline-Dice (topology/connectivity-preserving, good for thin clots)."""
    sp, st = soft_skeleton(prob, iters), soft_skeleton(target, iters)
    tprec = ((sp * target).flatten(1).sum(1) + eps) / (sp.flatten(1).sum(1) + eps)
    tsens = ((st * prob).flatten(1).sum(1) + eps) / (st.flatten(1).sum(1) + eps)
    return 1 - 2 * tprec * tsens / (tprec + tsens)


def _m(x, valid):
    valid = valid.float()
    return (x * valid).sum() / valid.sum().clamp(min=1)


class MultiTaskLoss(nn.Module):
    DEFAULT_W = dict(seg=1.0, cl=0.3, ves=0.5, mil=0.5, slice=1.0, node=0.5, det=1.0,
                     loc=0.5, quant=0.5, cons=0.2, sev=0.5)

    def __init__(self, weights=None, det_pos_weight=1.0, seg_pos_weight=10.0, cl_iters=3):
        super().__init__()
        self.w = dict(self.DEFAULT_W, **(weights or {}))
        self.det_pw, self.seg_pw, self.cl_iters = det_pos_weight, seg_pos_weight, cl_iters

    def forward(self, out, batch):
        L = {}
        dev = out["det"].device
        has = batch.get("has_mask")
        has = has.bool() if has is not None else torch.zeros_like(out["det"], dtype=torch.bool)

        # ---- supervised segmentation (only where masks exist) ------------------
        if has.any():
            lg, t = out["seg"][has], batch["seg"][has]
            p = torch.sigmoid(lg) * F.max_pool3d(batch["image"][has][:, 1:2], 3, 1, 1)
            inter = (p * t).flatten(1).sum(1)
            den = p.flatten(1).sum(1) + t.flatten(1).sum(1)
            dice = 1 - (2 * inter + 1) / (den + 1)
            bce = F.binary_cross_entropy_with_logits(
                lg, t, pos_weight=torch.tensor(self.seg_pw, device=dev), reduction="none").flatten(1).mean(1)
            L["seg"] = (dice + bce).mean()
            if out.get("seg_aux") is not None:
                ta = F.adaptive_max_pool3d(t, out["seg_aux"].shape[2:])
                L["seg"] = L["seg"] + 0.5 * F.binary_cross_entropy_with_logits(
                    out["seg_aux"][has], ta, pos_weight=torch.tensor(self.seg_pw, device=dev))
            L["cl"] = cl_dice(p, t, self.cl_iters).mean()

        # ---- topology prior: clot probability must stay inside the vessel tree --
        ves = batch["image"][:, 1:2]
        p_all = torch.sigmoid(out["seg"])
        L["ves"] = ((p_all * (1 - F.max_pool3d(ves, 3, 1, 1))).flatten(1).sum(1) /
                    p_all.flatten(1).sum(1).clamp(min=1.0)).mean()

        # ---- study-level detection ----------------------------------------------
        det = batch.get("det")
        if det is not None:
            ok = det >= 0
            if ok.any():
                bce = F.binary_cross_entropy_with_logits(
                    out["det"], det.clamp(min=0), pos_weight=torch.tensor(self.det_pw, device=dev), reduction="none")
                L["det"] = _m(bce, ok)
                # weak supervision (no mask): max clot evidence must agree with the study label
                weak = ok & ~has
                if weak.any():
                    pm = out["p_max"].clamp(1e-4, 1 - 1e-4)
                    L["mil"] = _m(F.binary_cross_entropy(pm, det.clamp(min=0), reduction="none"), weak)
                    if out["node_logit"].abs().sum() > 0:
                        nl = out["node_logit"].masked_fill(~batch["node_mask"], -1e4)
                        pn = torch.sigmoid(nl.max(1).values).clamp(1e-4, 1 - 1e-4)
                        L["mil"] = L["mil"] + _m(F.binary_cross_entropy(pn, det.clamp(min=0), reduction="none"), weak)

        # ---- slice-level weak localisation (RSNA-style image labels) ---------------
        sl = batch.get("slice_label")
        if sl is not None and (sl >= 0).any():
            ok = sl >= 0
            s = out["slice_score"].clamp(1e-4, 1 - 1e-4)
            L["slice"] = _m(F.binary_cross_entropy(s, sl.clamp(min=0), reduction="none"), ok)

        # ---- node-level (vessel-tree) localisation where node labels exist --------
        nl_lab = batch.get("node_label")
        if nl_lab is not None and (nl_lab >= 0).any():
            ok = (nl_lab >= 0) & batch["node_mask"]
            bce = F.binary_cross_entropy_with_logits(out["node_logit"], nl_lab.clamp(min=0), reduction="none")
            L["node"] = _m(bce, ok)

        # ---- location (multi-label; unknown = -1) -------------------------------
        loc = batch.get("loc")
        if loc is not None and (loc >= 0).any():
            ok = loc >= 0
            L["loc"] = _m(F.binary_cross_entropy_with_logits(out["loc"], loc.clamp(min=0), reduction="none"), ok)

        # ---- quantification ---------------------------------------------------------
        lab = batch.get("oi")
        if lab is not None and (lab >= 0).any():
            L["quant"] = _m(F.smooth_l1_loss(out["oi"], lab, reduction="none", beta=0.1), lab >= 0)
        # consistency: learned OI should track the voxel-level clot/vessel fraction
        L["cons"] = F.l1_loss(out["oi"], (out["ratio"].detach() * 6).clamp(max=1))

        # ---- severity surrogate (RV/LV >= 1) ----------------------------------------
        rv = batch.get("rvlv_hi")
        if rv is not None and (rv >= 0).any():
            ok = rv >= 0
            L["sev"] = _m(F.binary_cross_entropy_with_logits(out["rvlv"], rv.clamp(min=0), reduction="none"), ok)

        total = sum(self.w[k] * v for k, v in L.items())
        return total, {k: float(v.detach()) for k, v in L.items()}
