"""Vessel-tree graph extraction (numpy / scipy / scikit-image only, no torch).

Given a binary pulmonary-artery mask, build a fixed-size graph:

* nodes  = farthest-point-sampled voxels of the 3D centerline (skeleton)
* edges  = two nodes are adjacent if their skeleton "territories" touch
           (multi-source geodesic Voronoi on the skeleton graph), so edges follow
           real vessel connectivity, not Euclidean proximity
* node features (6): [radius, geodesic depth from root, skeleton degree, z, y, x]
           radius and depth are normalised to [0, 1]; z, y, x are normalised
           voxel coordinates in [0, 1].

The root is the skeleton voxel with the largest inscribed radius, which in a
CTPA vessel mask is the main pulmonary artery trunk.
"""
import itertools

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

# 13 of the 26 neighbours (the other half is obtained by symmetry)
_OFFSETS = [o for o in itertools.product((-1, 0, 1), repeat=3) if o > (0, 0, 0)]
NODE_DIM = 6


def _empty_graph(shape, max_nodes):
    nodes = np.zeros((max_nodes, NODE_DIM), np.float32)
    nodes[0, 3:6] = 0.5  # dummy centre node so every sample has >= 1 valid node
    node_mask = np.zeros(max_nodes, bool)
    node_mask[0] = True
    return {
        "nodes": nodes,
        "adj": np.eye(max_nodes, dtype=bool),
        "node_mask": node_mask,
        "node_vox": np.tile(np.array(shape) // 2, (max_nodes, 1)).astype(np.int64),
    }


def build_vessel_graph(vessel, max_nodes=256):
    """vessel: (D, H, W) bool/0-1 array. Returns dict of fixed-size arrays."""
    vessel = np.asarray(vessel) > 0.5
    shape = vessel.shape
    if vessel.sum() == 0:
        return _empty_graph(shape, max_nodes)

    skel = skeletonize(vessel)
    coords = np.argwhere(skel)
    if len(coords) == 0:  # very thin / tiny mask
        coords = np.argwhere(vessel)[:1]
    M = len(coords)

    edt = ndi.distance_transform_edt(vessel)
    radius = edt[tuple(coords.T)]

    # --- skeleton connectivity graph (26-neighbourhood) -------------------
    idx_vol = -np.ones(shape, np.int64)
    idx_vol[tuple(coords.T)] = np.arange(M)
    rows, cols, wts = [], [], []
    for off in _OFFSETS:
        nb = coords + np.array(off)
        ok = np.all((nb >= 0) & (nb < np.array(shape)), axis=1)
        j = np.full(M, -1, np.int64)
        j[ok] = idx_vol[tuple(nb[ok].T)]
        hit = j >= 0
        rows.append(np.nonzero(hit)[0])
        cols.append(j[hit])
        wts.append(np.full(int(hit.sum()), np.linalg.norm(off)))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    w = np.concatenate(wts)
    G = csr_matrix(
        (np.concatenate([w, w]), (np.concatenate([r, c]), np.concatenate([c, r]))),
        shape=(M, M),
    )

    # --- root, geodesic depth, degree ------------------------------------
    root = int(np.argmax(radius))
    depth = dijkstra(G, directed=False, indices=root)
    finite = np.isfinite(depth)
    depth[~finite] = depth[finite].max() if finite.any() else 0.0
    degree = np.asarray((G > 0).sum(axis=1)).ravel()

    # --- farthest point sampling (root first) -----------------------------
    K = min(max_nodes, M)
    sel = np.empty(K, np.int64)
    sel[0] = root
    d = np.linalg.norm(coords - coords[root], axis=1)
    for i in range(1, K):
        sel[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(coords - coords[sel[i]], axis=1))

    # --- edges between sampled nodes via geodesic Voronoi -----------------
    adj = np.eye(max_nodes, dtype=bool)
    if K > 1:
        _, _, src = dijkstra(G, directed=False, indices=sel, min_only=True,
                             return_predecessors=True)
        inv = -np.ones(M, np.int64)
        inv[sel] = np.arange(K)
        owner = np.where(src >= 0, inv[np.clip(src, 0, M - 1)], -1)
        a, b = owner[r], owner[c]
        ok = (a >= 0) & (b >= 0) & (a != b)
        adj[a[ok], b[ok]] = True
        adj[b[ok], a[ok]] = True

    nodes = np.zeros((max_nodes, NODE_DIM), np.float32)
    nodes[:K, 0] = radius[sel] / max(float(radius.max()), 1e-6)
    nodes[:K, 1] = depth[sel] / max(float(depth.max()), 1e-6)
    nodes[:K, 2] = np.clip(degree[sel], 0, 6) / 6.0
    nodes[:K, 3:6] = coords[sel] / np.maximum(np.array(shape) - 1, 1)
    node_mask = np.zeros(max_nodes, bool)
    node_mask[:K] = True
    node_vox = np.tile(np.array(shape) // 2, (max_nodes, 1)).astype(np.int64)
    node_vox[:K] = coords[sel]
    return {"nodes": nodes, "adj": adj, "node_mask": node_mask, "node_vox": node_vox}
