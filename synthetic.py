"""Synthetic CTPA-like volumes for smoke tests and pipeline debugging (numpy only).

A random branching "pulmonary artery tree" is grown, filling defects (clots) are
placed inside branches, and all labels used by the network are derived.

NOT a substitute for real data: severity labels here are placeholder rules.

Location classes (index): 0 main/central, 1 right PA, 2 left PA, 3 lobar,
4 segmental, 5 subsegmental.
"""
import numpy as np
from scipy import ndimage as ndi

from vessel_graph import build_vessel_graph

LOC_NAMES = ["central", "right_pa", "left_pa", "lobar", "segmental", "subsegmental"]
N_LOC = len(LOC_NAMES)


def _rotate(v, angle, axis):
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def _ball_mask(shape, center, r):
    lo = np.maximum(np.floor(center - r - 1).astype(int), 0)
    hi = np.minimum(np.ceil(center + r + 2).astype(int), shape)
    if np.any(hi <= lo):
        return None, None
    zz, yy, xx = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    m = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2 <= r * r
    return tuple(slice(l, h) for l, h in zip(lo, hi)), m


def _grow(rng, shape, pos, direction, radius, depth, region, branches, max_depth):
    length = 15.0 * (0.82 ** depth) * rng.uniform(0.8, 1.2)
    n = max(int(length), 3)
    path = []
    p = pos.astype(float).copy()
    for _ in range(n):
        p = p + direction
        if np.any(p < 2) or np.any(p > np.array(shape) - 3):
            break
        path.append(p.copy())
    if len(path) < 2:
        return
    path = np.array(path)
    branches.append({"path": path, "radius": radius, "region": region, "depth": depth})
    if depth >= max_depth:
        return
    u = _unit(rng.normal(size=3))
    axis = _unit(np.cross(direction, u))
    ang = np.deg2rad(rng.uniform(25, 45))
    for k, sign in enumerate((+1, -1)):
        child_dir = _unit(_rotate(direction, sign * ang, axis))
        if depth == 0:
            child_region = 1 if k == 0 else 2
        else:
            child_region = min(depth + 2, 5)
        _grow(rng, shape, path[-1], child_dir, radius * 0.72, depth + 1,
              child_region, branches, max_depth)


def make_study(seed, size=64, max_nodes=128, pos_rate=0.6):
    rng = np.random.default_rng(seed)
    shape = (size, size, size)

    # ---- vessel tree -------------------------------------------------------
    branches = []
    start = np.array([size / 2, size / 2, 3.0])
    _grow(rng, shape, start, np.array([0.0, 0.0, 1.0]),
          radius=0.07 * size, depth=0, region=0, branches=branches, max_depth=4)

    vessel = np.zeros(shape, bool)
    region = np.zeros(shape, np.int8)  # region id + 1 inside vessel, 0 outside
    for b in branches:
        for p in b["path"]:
            sl, m = _ball_mask(shape, p, b["radius"])
            if sl is None:
                continue
            vessel[sl] |= m
            reg = region[sl]
            reg[m & (reg == 0)] = b["region"] + 1

    # ---- clots -------------------------------------------------------------
    clot = np.zeros(shape, bool)
    is_pe = rng.random() < pos_rate
    if is_pe:
        n_clots = int(rng.integers(1, 4))
        cand = [b for b in branches if b["depth"] >= 1] or branches
        for _ in range(n_clots):
            b = cand[int(rng.integers(len(cand)))]
            L = len(b["path"])
            s = int(rng.integers(0, max(L - 1, 1)))
            e = int(min(L, s + max(2, int(rng.uniform(0.3, 1.0) * L))))
            offset_dir = _unit(rng.normal(size=3)) * b["radius"] * rng.uniform(0.0, 0.5)
            for p in b["path"][s:e]:
                sl, m = _ball_mask(shape, p + offset_dir, b["radius"] * rng.uniform(0.7, 1.0))
                if sl is not None:
                    clot[sl] |= m
        clot &= vessel
    if clot.sum() < 3:
        clot[:] = False
        is_pe = False

    # ---- CT-like image -----------------------------------------------------
    ct = np.full(shape, -0.6, np.float32)
    ct[vessel] = 0.8
    ct[clot] = 0.1
    ct += rng.normal(0, 0.12, shape).astype(np.float32)
    ct = ndi.gaussian_filter(ct, 0.7)

    # vessel "prior" as if produced by an upstream vessel segmenter (soft, imperfect)
    vessel_prior = ndi.gaussian_filter(vessel.astype(np.float32), 1.0)
    dist = ndi.distance_transform_edt(vessel)
    dist = (dist / max(float(dist.max()), 1.0)).astype(np.float32)
    image = np.stack([ct, vessel_prior, dist], 0).astype(np.float32)

    # ---- graph and node labels --------------------------------------------
    g = build_vessel_graph(vessel, max_nodes=max_nodes)
    vox = g["node_vox"]
    if clot.any():
        dclot = ndi.distance_transform_edt(~clot)
        node_label = (dclot[tuple(vox.T)] <= 2.0).astype(np.float32)
    else:
        node_label = np.zeros(max_nodes, np.float32)
    node_label[~g["node_mask"]] = -1.0
    node_region = (region[tuple(vox.T)].astype(np.int64) - 1).clip(0)

    # ---- study-level labels -----------------------------------------------
    loc = np.zeros(N_LOC, np.float32)
    for k in range(N_LOC):
        loc[k] = float((clot & (region == k + 1)).any())
    ratio = float(clot.sum() / max(vessel.sum(), 1))
    oi = float(np.clip(ratio * 6.0, 0.0, 1.0))  # placeholder obstruction index
    rvlv = float(0.6 + 1.2 * oi + 0.15 * loc[0] + rng.normal(0, 0.05))
    rvlv_hi = float(is_pe and rvlv >= 1.0)  # RSNA-style binary RV/LV >= 1 label
    slice_label = clot.any(axis=(1, 2)).astype(np.float32)  # per-slice PE presence (axis 0)

    return {
        "image": image,
        "seg": clot[None].astype(np.float32),
        "has_mask": np.array(True),
        "nodes": g["nodes"], "adj": g["adj"], "node_mask": g["node_mask"],
        "node_label": node_label, "node_region": node_region,
        "det": np.float32(is_pe), "loc": loc,
        "oi": np.float32(oi), "burden": np.float32(ratio),
        "rvlv_hi": np.float32(rvlv_hi), "slice_label": slice_label,
    }
