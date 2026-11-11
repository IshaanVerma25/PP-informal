"""Runs without torch:  python -I tests/test_numpy_parts.py   (from the project root)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from synthetic import make_study
from postprocess import extract_candidates, suppress_false_positives, mark_emboli, lesion_metrics
from moo import SimpleES, RandomSearch, optimize

# 1. graph invariants
d = make_study(1)
nm, adj, nodes = d["node_mask"], d["adj"], d["nodes"]
assert nm.sum() > 10 and adj.shape == (128, 128) and (adj == adj.T).all() and adj.diagonal().all()
assert nodes[nm].min() >= 0 and nodes[nm].max() <= 1.0 + 1e-6
assert not adj[nm][:, ~nm].any(), "valid nodes must not connect to padding"

# 2. candidates + FP suppression on a noisy 'prediction' of the true clot
gt, ves = d["seg"][0], d["image"][1]
rng = np.random.default_rng(0)
pred = np.clip(gt * 0.9 + rng.random(gt.shape) * 0.1, 0, 1)
fake = np.zeros_like(gt); fake[2:5, 2:5, 2:5] = 0.9          # off-vessel false positive
cands, lab = extract_candidates(pred + fake, ves, d["nodes"], d["nodes"][:, 3:6] * 63, thr=0.5)
kept = suppress_false_positives(cands)
assert all(c["vessel_overlap"] >= 0.7 for c in kept) and len(kept) <= len(cands)
m = lesion_metrics(kept, gt)
print("candidates", len(cands), "kept", len(kept), m)
assert m["fp"] == 0 or d["det"] == 0
mark_emboli(pred, ves, lab, kept)

# 3. optimiser scaffold on a toy objective
def toy(p):
    return dict(auc_det=1 - abs(p["lr"] - 3e-4) * 100, lesion_sens=0.8, fp_per_scan=abs(p["det_thr"] - 0.5), dice=0.6)
for opt in (RandomSearch(seed=0), SimpleES(seed=0)):
    best = optimize(toy, opt, budget=30)
    print(type(opt).__name__, "best score %.3f" % best[1])
print("OK")
