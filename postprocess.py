"""Candidate extraction, false-positive suppression, embolus marking (numpy/scipy only).

pipeline: gated clot probability -> threshold -> 3D connected components
          -> per-candidate features -> rule/logistic FP suppression -> marked emboli
          (3D mask, 3D bbox, nearest vessel node + anatomical label, confidence).
"""
import numpy as np
from scipy import ndimage as ndi

LOC_NAMES_6 = ["central", "right_pa", "left_pa", "lobar", "segmental", "subsegmental"]


def extract_candidates(prob, vessel, nodes=None, node_vox=None, node_attn_score=None,
                       thr=0.5, min_vox=8, vessel_overlap=0.7):
    """prob, vessel: (D,H,W) numpy. Returns (candidates list, labelled volume)."""
    lab, n = ndi.label(prob >= thr)
    vdil = ndi.binary_dilation(vessel > 0.5, iterations=1)
    cands = []
    for i in range(1, n + 1):
        m = lab == i
        vol = int(m.sum())
        zz, yy, xx = np.nonzero(m)
        c = dict(
            id=i, volume_vox=vol,
            mean_prob=float(prob[m].mean()), max_prob=float(prob[m].max()),
            vessel_overlap=float(vdil[m].mean()),
            bbox=(int(zz.min()), int(yy.min()), int(xx.min()), int(zz.max()) + 1, int(yy.max()) + 1, int(xx.max()) + 1),
            centroid=(float(zz.mean()), float(yy.mean()), float(xx.mean())),
        )
        if node_vox is not None:
            d = np.linalg.norm(node_vox - np.array(c["centroid"]), axis=1)
            j = int(np.argmin(d))
            c["nearest_node"] = j
            c["node_radius"] = float(nodes[j, 0]) if nodes is not None else None
            c["node_depth"] = float(nodes[j, 1]) if nodes is not None else None
            if node_attn_score is not None:
                c["node_score"] = float(node_attn_score[j])
        cands.append(c)
    return cands, lab


def suppress_false_positives(cands, min_vox=8, min_vessel_overlap=0.7, min_mean_prob=0.5,
                             w=(0.5, 0.3, 0.2)):
    """Rule + weighted-evidence rescoring. Returns kept candidates with `score`.

    A learned classifier (e.g. logistic regression / GBM on these features) can replace
    the weighted sum once candidate-level labels exist.
    """
    kept = []
    for c in cands:
        if c["volume_vox"] < min_vox or c["vessel_overlap"] < min_vessel_overlap or c["mean_prob"] < min_mean_prob:
            continue
        s = w[0] * c["mean_prob"] + w[1] * c["max_prob"] + w[2] * c["vessel_overlap"]
        c = dict(c, score=float(s))
        kept.append(c)
    return sorted(kept, key=lambda c: -c["score"])


def mark_emboli(prob, vessel, lab, kept):
    """3D marked-embolus mask (labelled by candidate rank) for the kept candidates."""
    marked = np.zeros(prob.shape, np.int16)
    for rank, c in enumerate(kept, 1):
        marked[lab == c["id"]] = rank
    return marked


def lesion_metrics(kept, gt_mask, iou_thr=0.1):
    """Lesion-level sensitivity / false positives for a scan with a reference mask."""
    lab_gt, n_gt = ndi.label(gt_mask > 0.5)
    hit_gt, fp = set(), 0
    for c in kept:
        z0, y0, x0, z1, y1, x1 = c["bbox"]
        sub = lab_gt[z0:z1, y0:y1, x0:x1]
        ids = [i for i in np.unique(sub) if i > 0]
        if ids:
            hit_gt.update(ids)
        else:
            fp += 1
    return {"n_gt": n_gt, "tp_gt": len(hit_gt), "fp": fp,
            "sensitivity": len(hit_gt) / n_gt if n_gt else float("nan")}
