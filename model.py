"""Vessel-Topology-Aware 3D Multi-Task Network (VTMT-Net).  PyTorch.

Input  image : (B, in_ch, D, H, W)   channels = [CT, vessel prior, vessel distance]
       nodes : (B, N, 6)             vessel-graph node features  (vessel_graph.py)
       adj   : (B, N, N) bool        vessel-graph adjacency (self loops included)
       node_mask : (B, N) bool       valid nodes

Output dict:
  seg        (B,1,D,H,W) clot logits (vessel-gated prob = sigmoid(seg)*vessel_dilated)
  seg_aux    (B,1,D/2,H/2,W/2) deep-supervision logits
  node_logit (B,N)  per-vessel-node clot logit        -> localisation on the tree
  node_attn  (B,N,N) mean graph-attention weights     -> explanation
  det        (B,)   study-level PE logit
  loc        (B,n_loc) location logits (multi-label)
  oi         (B,)   learned obstruction index in [0,1] (where reference exists)
  rvlv       (B,)   logit for RV/LV >= 1  (severity surrogate)
  p_max,p_top,slice_score  weak-supervision statistics of the gated clot map

Ablation switches: in_ch (vessel representation), use_multiscale, use_topology.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c):
    return nn.GroupNorm(min(8, c), c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), _gn(cout), nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), _gn(cout))
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.red = nn.Conv3d(cin, cout, 1)
        self.blk = ResBlock(cout + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(self.red(x), size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.blk(torch.cat([x, skip], 1))


class GraphAttentionLayer(nn.Module):
    """Multi-head self-attention restricted to vessel-tree edges, with a learned
    additive bias from pairwise geometry (d radius, d depth, euclidean distance)."""

    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dk = heads, dim // heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj, edge_bias):
        B, N, D = x.shape
        sp = lambda t: t.view(B, N, self.h, self.dk).transpose(1, 2)
        q, k, v = sp(self.q(x)), sp(self.k(x)), sp(self.v(x))
        att = q @ k.transpose(-1, -2) / math.sqrt(self.dk) + edge_bias          # B,h,N,N
        att = att.masked_fill(~adj[:, None], torch.finfo(att.dtype).min)
        w = self.drop(att.softmax(-1))
        out = (w @ v).transpose(1, 2).reshape(B, N, D)
        x = self.n1(x + self.drop(self.o(out)))
        x = self.n2(x + self.drop(self.ffn(x)))
        return x, w.mean(1)


class VesselTopologyModule(nn.Module):
    """Samples encoder features on the vessel centerline nodes, runs graph attention
    over the vessel tree, and projects the result back onto the feature volume."""

    def __init__(self, feat_ch, dim=128, layers=3, heads=4, n_loc=3, geo_dim=6):
        super().__init__()
        self.heads = heads
        self.inp = nn.Linear(feat_ch + geo_dim, dim)
        self.edge_mlp = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, heads))
        self.layers = nn.ModuleList([GraphAttentionLayer(dim, heads) for _ in range(layers)])
        self.node_head = nn.Linear(dim, 1)
        self.back = nn.Linear(dim, feat_ch)
        self.diffuse = nn.Sequential(nn.Conv3d(feat_ch, feat_ch, 3, padding=1), _gn(feat_ch), nn.GELU())
        self.gate = nn.Parameter(torch.zeros(1))
        # readout queries: 1 global + n_loc location queries
        self.queries = nn.Parameter(torch.randn(1 + n_loc, dim) * 0.02)
        self.readout = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n_loc = n_loc

    def forward(self, feat, nodes, adj, node_mask):
        B, C, d, h, w = feat.shape
        N = nodes.shape[1]
        xyz = nodes[..., 3:6]                                               # (z,y,x) in [0,1]
        grid = (xyz[..., [2, 1, 0]] * 2 - 1).view(B, N, 1, 1, 3).to(feat.dtype)
        samp = F.grid_sample(feat, grid, mode="bilinear", align_corners=True)
        samp = samp.view(B, C, N).transpose(1, 2)                           # B,N,C
        x = self.inp(torch.cat([samp, nodes.to(samp.dtype)], -1))

        r, dep = nodes[..., 0], nodes[..., 1]
        dist = torch.cdist(xyz.float(), xyz.float())
        rel = torch.stack([r[:, :, None] - r[:, None, :], dep[:, :, None] - dep[:, None, :], dist], -1)
        bias = self.edge_mlp(rel.to(x.dtype)).permute(0, 3, 1, 2)           # B,h,N,N

        attn = None
        for layer in self.layers:
            x, attn = layer(x, adj, bias)
        node_logit = self.node_head(x).squeeze(-1)

        # scatter node embeddings back to the bottleneck grid
        size = torch.tensor([d, h, w], device=feat.device)
        idx = (xyz * (size - 1)).round().long()
        idx = torch.minimum(idx.clamp(min=0), size - 1)
        flat = (idx[..., 0] * h + idx[..., 1]) * w + idx[..., 2]            # B,N
        m = node_mask.to(x.dtype)
        back = (self.back(x) * m[..., None]).transpose(1, 2)                # B,C,N
        vol = torch.zeros(B, C, d * h * w, device=feat.device, dtype=back.dtype)
        vol.scatter_add_(2, flat[:, None, :].expand(-1, C, -1), back)
        cnt = torch.zeros(B, 1, d * h * w, device=feat.device, dtype=back.dtype)
        cnt.scatter_add_(2, flat[:, None, :], m[:, None, :])
        vol = (vol / cnt.clamp(min=1)).view(B, C, d, h, w)
        fused = feat + torch.tanh(self.gate) * self.diffuse(vol)

        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        ro, ro_w = self.readout(q, x, x, key_padding_mask=~node_mask, need_weights=True)
        return {"feat": fused, "node_logit": node_logit, "node_attn": attn,
                "g": ro[:, 0], "loc_emb": ro[:, 1:], "readout_attn": ro_w}


class VTMTNet(nn.Module):
    def __init__(self, in_ch=3, channels=(16, 32, 64, 128, 256), graph_dim=128, graph_layers=3,
                 heads=4, n_loc=3, use_multiscale=True, use_topology=True, dropout=0.2):
        super().__init__()
        self.use_multiscale, self.use_topology, self.n_loc = use_multiscale, use_topology, n_loc
        self.enc0 = ResBlock(in_ch, channels[0])
        self.downs = nn.ModuleList([
            nn.Sequential(nn.Conv3d(channels[i], channels[i + 1], 2, stride=2), ResBlock(channels[i + 1], channels[i + 1]))
            for i in range(len(channels) - 1)])
        if use_topology:
            self.vtm = VesselTopologyModule(channels[-1], graph_dim, graph_layers, heads, n_loc)
        else:  # ablation: plain pooled readout, no graph
            self.plain_g = nn.Linear(channels[-1], graph_dim)
            self.plain_loc = nn.ModuleList([nn.Linear(channels[-1], graph_dim) for _ in range(n_loc)])
        self.cam_layer = nn.Identity()  # Grad-CAM hook point (fused bottleneck)

        self.ups = nn.ModuleList([UpBlock(channels[j + 1], channels[j], channels[j])
                                  for j in reversed(range(len(channels) - 1))])
        self.seg_out = nn.Conv3d(channels[0], 1, 1)
        self.aux_out = nn.Conv3d(channels[1], 1, 1)

        # cross-scale fusion: global descriptor of every encoder scale
        fuse_in = sum(channels) if use_multiscale else channels[-1]
        self.scale_fuse = nn.Sequential(nn.Linear(fuse_in, 128), nn.GELU())
        head_in = graph_dim + 128 + 4
        mk = lambda o: nn.Sequential(nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, o))
        self.det_head, self.quant_head, self.sev_head = mk(1), mk(1), mk(1)
        self.loc_head = nn.Sequential(nn.Linear(graph_dim, 64), nn.GELU(), nn.Linear(64, 1))

    @staticmethod
    def seg_stats(seg_logit, vessel):
        """Vessel-gated clot probability and weak-supervision statistics."""
        gate = F.max_pool3d(vessel, 3, 1, 1)                       # dilated vessel prior
        p = torch.sigmoid(seg_logit) * gate
        pd = F.max_pool3d(p, 2)
        flat = pd.flatten(1)
        top = flat.topk(min(256, flat.shape[1]), dim=1).values
        p_max, p_top = top[:, 0], top.mean(1)
        ratio = (p * vessel).flatten(1).sum(1) / vessel.flatten(1).sum(1).clamp(min=1.0)
        # slice score along depth axis: mean of top-16 in-plane probabilities
        sl = p.flatten(3).squeeze(1)                                # B,D,HW
        slice_score = sl.topk(min(16, sl.shape[-1]), dim=-1).values.mean(-1)
        feats = torch.stack([p_max, p_top, ratio.clamp(max=1) * 5, p.flatten(1).mean(1) * 10], 1)
        return p, p_max, p_top, ratio, slice_score, feats

    def forward(self, image, nodes, adj, node_mask):
        node_mask = node_mask.clone()
        node_mask[:, 0] = True                                     # never an empty graph
        # in_ch == 1 is the "no vessel representation" ablation: gate becomes a no-op
        vessel = image[:, 1:2] if image.shape[1] > 1 else torch.ones_like(image[:, :1])

        feats = [self.enc0(image)]
        for dn in self.downs:
            feats.append(dn(feats[-1]))
        skips, bott = feats[:-1], feats[-1]

        if self.use_topology:
            t = self.vtm(bott, nodes, adj, node_mask)
            bott = t["feat"]
            g, loc_emb = t["g"], t["loc_emb"]
            node_logit, node_attn = t["node_logit"], t["node_attn"]
        else:
            gp = bott.mean((2, 3, 4))
            g = self.plain_g(gp)
            loc_emb = torch.stack([m(gp) for m in self.plain_loc], 1)
            node_logit = torch.zeros(nodes.shape[:2], device=image.device)
            node_attn = None
        bott = self.cam_layer(bott)

        x, aux = bott, None
        for i, up in enumerate(self.ups):
            x = up(x, skips[-(i + 1)])
            if i == len(self.ups) - 2:
                aux = self.aux_out(x)
        seg = self.seg_out(x)

        p, p_max, p_top, ratio, slice_score, sfeat = self.seg_stats(seg, vessel)

        scales = [f.mean((2, 3, 4)) for f in skips] + [bott.mean((2, 3, 4))]
        ms = self.scale_fuse(torch.cat(scales, 1) if self.use_multiscale else scales[-1])
        h = torch.cat([g, ms, sfeat], 1)
        return {
            "seg": seg, "seg_aux": aux, "p": p, "p_max": p_max, "p_top": p_top, "ratio": ratio,
            "slice_score": slice_score, "node_logit": node_logit, "node_attn": node_attn,
            "det": self.det_head(h).squeeze(-1),
            "loc": self.loc_head(loc_emb).squeeze(-1),
            "oi": torch.sigmoid(self.quant_head(h)).squeeze(-1),
            "rvlv": self.sev_head(h).squeeze(-1),
        }


def burden_index(node_prob, nodes, node_mask):
    """Research-derived, topology-weighted PE burden index in [0, 1].

        BI = sum_i w_i * c_i / sum_i w_i ,  w_i = radius_i^2 (cross-section proxy)

    c_i = clot probability at vessel node i. Proximal (large-calibre) occlusion
    therefore counts more than distal. NOT clinically validated.
    """
    w = (nodes[..., 0].clamp(min=1e-3) ** 2) * node_mask.float()
    return (w * node_prob).sum(1) / w.sum(1).clamp(min=1e-6)


def volumetric_burden(p, vessel, voxel_ml=None):
    """Soft clot volume (voxels, or mL if voxel_ml) and clot/vessel volume fraction."""
    vol = p.flatten(1).sum(1)
    frac = (p * vessel).flatten(1).sum(1) / vessel.flatten(1).sum(1).clamp(min=1.0)
    return (vol * voxel_ml if voxel_ml else vol), frac
