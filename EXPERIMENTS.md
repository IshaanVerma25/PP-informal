# Experiment plan (what RSNA-STR alone can and cannot support)

Split by study/patient, never by slice. Report mean ± 95% CI (bootstrap), ≥3 seeds.

| Exp | Content | Supported by RSNA labels? |
|---|---|---|
| **A Baselines** | 3D ResNet, DenseNet, EfficientNet-3D, 2.5D CNN→RNN, 3D Transformer (e.g. Swin/ViT-3D), VTMT-Net. Same preprocessing, same folds, tuned equally. Metrics: AUROC, AUPRC, sensitivity @ fixed specificity, ECE. | Yes (study-level) |
| **B Ablation** | baseline → `--in-ch 3` (vessel representation) → multi-scale → topology attention → multi-task heads → FP suppression → full. Flags: `--in-ch 1`, `--no-multiscale`, `--no-topology`, loss weights, `postprocess.suppress_false_positives`. | Yes |
| **C Localisation** | Slice-level AUROC/sensitivity (from `pe_present_on_image`), CAM/clot-map peak-on-positive-slice rate. **IoU, Dice, lesion-level sensitivity, FP/scan need reference markings** → radiologist-annotate a subset (suggest ≥100 positive studies) or use a dataset with marked emboli (e.g. FUMPE, CAD-PE — check access/licence). | Slice-level only |
| **D Quantification** | MAE/RMSE/correlation/Bland–Altman need a reference. **None in RSNA.** Options: radiologist Qanadli/Mastora scores on a subset; otherwise report agreement with proxies only (RV/LV ≥ 1 AUROC, fraction of PE-positive slices, left/right/central) and say so. | No reference |
| **E Explainability** | Grad-CAM vs Grad-CAM++ (`plus=True`) vs node attention vs combined. Metrics: pointing-game / peak-on-positive-slice (slice labels), deletion/insertion AUC, stability across seeds, reader study on a subset. | Partly |
| **F Robustness** | Gaussian noise, slice-thickness resampling, HU-window/kernels, motion/contrast-timing difficult cases, class-imbalance (reweighting vs resampling). Report ΔAUROC and ΔFP/scan. Also test exclusion of the vessel prior's errors (perturb vessel mask). | Yes |

Cross-cutting: external validation on another site/dataset; calibration (reliability curves, temperature scaling); MC-dropout uncertainty vs error (AURC); per-subgroup results (chronic, indeterminate, central/segmental/subsegmental).

Optimisation study (secondary): `moo.optimize` with `composite_objective` vs plain learning-rate tuning; report the Pareto trade-off between AUROC, lesion sensitivity and FP/scan.
