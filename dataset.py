"""Torch datasets: synthetic (runs now) and an RSNA-STR adapter (needs your data).

RSNA-STR PE challenge labels (public challenge description; verify against your copy of
the TFRecords, whose feature names I could NOT inspect):
  study  : negative_exam_for_pe, rv_lv_ratio_gte_1, rv_lv_ratio_lt_1, leftsided_pe,
           rightsided_pe, central_pe, chronic_pe, acute_and_chronic_pe, indeterminate
  slice  : pe_present_on_image          (no segmentation masks!)
=> localisation = weak (slice labels + MIL), quantification has NO reference standard,
   severity = RV/LV >= 1.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from synthetic import make_study, N_LOC

BATCH_KEYS_FLOAT = ["image", "seg", "nodes", "node_label", "det", "loc", "oi", "burden",
                    "rvlv_hi", "slice_label"]


class SyntheticPEDataset(Dataset):
    def __init__(self, n=64, size=64, max_nodes=128, seed=0, weak_fraction=0.0):
        """weak_fraction: share of studies whose masks / node labels are hidden, to mimic
        RSNA-style (study + slice labels only) supervision."""
        self.n, self.size, self.max_nodes, self.seed, self.weak = n, size, max_nodes, seed, weak_fraction

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        d = make_study(self.seed * 100003 + i, self.size, self.max_nodes)
        weak = (np.random.default_rng(self.seed * 7919 + i).random() < self.weak)
        if weak:
            d["has_mask"] = np.array(False)
            d["node_label"] = np.full_like(d["node_label"], -1.0)
            d["oi"] = np.float32(-1.0)
            d["burden"] = np.float32(-1.0)
            d["loc"] = np.full_like(d["loc"], -1.0)
        out = {k: torch.from_numpy(np.asarray(v)) for k, v in d.items()}
        out["has_mask"] = out["has_mask"].bool()
        out["adj"] = out["adj"].bool()
        out["node_mask"] = out["node_mask"].bool()
        out["node_region"] = out["node_region"].long()
        return out


class RSNAStudyDataset(Dataset):
    """Adapter skeleton for RSNA-STR studies already converted to .npz per study.

    Expected per-study npz (create with a one-off conversion script from the TFRecords,
    e.g. using tf.data once, then train without TensorFlow):
        ct          (D,H,W) float32, HU windowed to [-100, 700] -> scaled to [0, 1] ... or raw HU
        vessel      (D,H,W) uint8   pulmonary-artery mask from an UPSTREAM vessel segmenter
        slice_label (D,)    float32 pe_present_on_image (aligned to the resampled z axis)
        study       dict-like npz scalars: negative_exam_for_pe, leftsided_pe, rightsided_pe,
                    central_pe, rv_lv_ratio_gte_1, chronic_pe, indeterminate
    Studies flagged indeterminate or chronic-only should be excluded/handled explicitly.
    """

    def __init__(self, paths, size=(96, 128, 128), max_nodes=256):
        from vessel_graph import build_vessel_graph  # lazy
        self.paths, self.size, self.max_nodes = list(paths), size, max_nodes
        self._graph = build_vessel_graph

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from scipy import ndimage as ndi
        z = np.load(self.paths[i])
        ct, ves, sl = z["ct"].astype(np.float32), z["vessel"].astype(np.float32), z["slice_label"].astype(np.float32)
        f = [t / s for t, s in zip(self.size, ct.shape)]
        ct = ndi.zoom(ct, f, order=1)
        ves = (ndi.zoom(ves, f, order=0) > 0.5).astype(np.float32)
        sl = (ndi.zoom(sl, f[0], order=0) > 0.5).astype(np.float32)
        dist = ndi.distance_transform_edt(ves)
        dist = (dist / max(float(dist.max()), 1.0)).astype(np.float32)
        g = self._graph(ves, self.max_nodes)
        neg = float(z["negative_exam_for_pe"])
        loc = np.array([float(z["leftsided_pe"]), float(z["rightsided_pe"]), float(z["central_pe"])], np.float32)
        if neg > 0.5:
            loc[:] = 0.0
        out = {
            "image": torch.from_numpy(np.stack([ct, ves, dist]).astype(np.float32)),
            "seg": torch.zeros(1, *self.size), "has_mask": torch.tensor(False),
            "nodes": torch.from_numpy(g["nodes"]), "adj": torch.from_numpy(g["adj"]),
            "node_mask": torch.from_numpy(g["node_mask"]),
            "node_label": torch.full((self.max_nodes,), -1.0),
            "det": torch.tensor(1.0 - neg), "loc": torch.from_numpy(loc),
            "oi": torch.tensor(-1.0), "burden": torch.tensor(-1.0),       # no reference
            "rvlv_hi": torch.tensor(float(z["rv_lv_ratio_gte_1"])),
            "slice_label": torch.from_numpy(sl),
        }
        return out
