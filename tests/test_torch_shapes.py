"""Needs torch (NOT run in the authoring sandbox):  python tests/test_torch_shapes.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch.utils.data import DataLoader

from dataset import SyntheticPEDataset
from losses import MultiTaskLoss
from model import VTMTNet, burden_index
from explain import GradCAM3D, explain_study, mc_dropout_predict
from synthetic import N_LOC, LOC_NAMES

b = next(iter(DataLoader(SyntheticPEDataset(4, 64, 128, weak_fraction=0.5), batch_size=2)))
for kw in [dict(), dict(use_topology=False), dict(use_multiscale=False)]:
    m = VTMTNet(n_loc=N_LOC, **kw)
    o = m(b["image"], b["nodes"], b["adj"], b["node_mask"])
    assert o["seg"].shape == (2, 1, 64, 64, 64) and o["det"].shape == (2,) and o["loc"].shape == (2, N_LOC)
    assert o["slice_score"].shape == (2, 64)
    loss, parts = MultiTaskLoss()(o, b)
    loss.backward()
    assert torch.isfinite(loss), parts
    print(kw, "loss %.3f" % float(loss), sorted(parts))

m = VTMTNet(n_loc=N_LOC)
cam = GradCAM3D(m)
x = {k: v[:1] for k, v in b.items()}
rep = explain_study(m, cam, x["image"], x["nodes"], x["adj"], x["node_mask"], LOC_NAMES)
print({k: v for k, v in rep.items() if k != "maps"})
print(mc_dropout_predict(m, x["image"], x["nodes"], x["adj"], x["node_mask"], n=3)["det"])
print("OK")
