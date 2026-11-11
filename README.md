# VTMT-Net: Vessel-Topology-Aware 3D Multi-Task Network for Explainable PE on CTPA

Detect → localise/mark → quantify → assess severity, with explanations.

## Status (read this first)
| Part | State |
|---|---|
| `vessel_graph.py`, `synthetic.py`, `postprocess.py`, `moo.py` | **Run and tested** (`python -I tests/test_numpy_parts.py`) |
| `model.py`, `losses.py`, `explain.py`, `dataset.py`, `train.py` | Written and byte-compiled, **never executed** (no PyTorch in the authoring sandbox). Run `tests/test_torch_shapes.py` first and expect to fix small shape/dtype bugs. |
| RSNA-STR TFRecord reading | **Not implemented.** `RSNAStudyDataset` expects per-study `.npz` you create once from the TFRecords. I could not inspect the Kaggle schema. |
| Vessel segmentation | **Not included.** The RSNA data has no vessel masks; you need an upstream pulmonary-artery segmenter (any pretrained one, or a classical contrast-threshold + vesselness fallback). |

## Data assumption (RSNA-STR PE, per the public challenge description; verify on your copy)
Study-level: `negative_exam_for_pe, rv_lv_ratio_gte_1, rv_lv_ratio_lt_1, leftsided_pe, rightsided_pe, central_pe, chronic_pe, acute_and_chronic_pe, indeterminate`. Slice-level: `pe_present_on_image`. **No segmentation masks, no burden reference.** Consequences built into the design:
- **Localisation** is weakly supervised: slice labels + multiple-instance losses on the vessel-gated clot map and on vessel-node scores. Dice/IoU need a separately annotated subset (or another dataset with reference markings).
- **Location** head is 3-class (left/right/central) on RSNA; the 6-class anatomical version (central, R/L PA, lobar, segmental, subsegmental) is used on synthetic data and any dataset with such labels.
- **Quantification** has no reference standard on RSNA. The burden index is a research-derived proxy.
- **Severity** = the RV/LV ≥ 1 label (a CT surrogate, not clinical risk stratification, which also needs troponin/haemodynamics).

## Architecture
```
CT + vessel prior + vessel distance  (3 ch)
  → multi-scale 3D ResUNet encoder
  → Vessel-Topology Module (bottleneck):
        centerline nodes (radius, geodesic depth, degree, xyz)  ← vessel_graph.py
        sample features at nodes → graph attention restricted to vessel-tree edges
        (+ learned geometric edge bias) → node clot logits, query readouts
        → scatter back to feature volume (gated residual)
  → decoder → clot logits, gated by dilated vessel prior (clot must lie in a vessel)
  → cross-scale fusion (global descriptor of every encoder scale)
  → heads: detection | location | obstruction index | RV/LV≥1
  → post-processing: candidates → false-positive suppression → marked emboli (3D mask, bbox, nearest vessel node)
  → uncertainty (MC-dropout) → explanation (Grad-CAM × vessel × clot map, node attention)
```
Vessel tree G = (V, E): V = sampled centreline nodes with features x_i = [r_i, depth_i, degree_i, z_i, y_i, x_i]; E = pairs whose skeleton territories touch (geodesic Voronoi), so edges follow real connectivity.

## Quantities (my definitions — the formulas in your notes did not come through)
- Topology-weighted burden index: `BI = Σ_i w_i c_i / Σ_i w_i`, `w_i = r_i²` (cross-section proxy), `c_i` = node clot probability. Proximal occlusion weighs more.
- Volumetric burden: `V_clot = Σ_v p(v)·vox_vol`; fraction `Σ p·vessel / Σ vessel`.
- Categories low / moderate / high use placeholder cut-offs (0.15, 0.40); calibrate on validation data. Call it a *research-derived quantitative burden index* unless clinically validated.
- Total loss: `L = Σ_k w_k L_k` over seg, clDice (vessel-structure), vessel-containment, MIL, slice, node, detection, location, quantification, consistency, severity. Terms with missing labels are skipped automatically.

## Run
```bash
pip install -r requirements.txt
python -I tests/test_numpy_parts.py            # no torch needed
python tests/test_torch_shapes.py              # first thing to run once torch is installed
python train.py --epochs 3                     # synthetic smoke test (50% of studies mask-free)
python train.py --no-topology                  # ablations: --no-topology --no-multiscale --in-ch 1
```
Ablation flags map to Experiment B in `EXPERIMENTS.md`.

## Optimisation
`moo.py` provides a plug-in interface (`ask()/tell()`) plus random search and a small evolution strategy, and a composite objective that penalises classification error, localisation error, false positives per scan, burden error and calibration error. GDOA can be dropped in behind `ask()/tell()` — I did not implement it because I am not sure which algorithm you mean.

## Not a medical device. Research code; no clinical validation.
