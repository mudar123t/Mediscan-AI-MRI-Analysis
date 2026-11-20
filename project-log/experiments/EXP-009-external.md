# EXP-009 — External test on BDNeuro-MRI (M1.4, H-B3)

- **Date:** 2026-11-20.
  - Data audit about 09:05.
  - Preliminary inference (det + tta) 10:35.
  - **Final inference (det + ens)** 16:07 → about 16:16, at about 110 s per fold model set.
- **Branch / code:** `exp/m1-uncertainty`.
  - `scripts/prepare_bdneuro.py`, `configs/data_bdneuro.yaml`, `scripts/run_shift_external.py --mode external`, `scripts/eval_shift_external.py`.
- **Data:** BDNeuro-MRI v7, Hira et al. 2026, CC BY 4.0, DOI [10.17632/zwr4ntf94j.7](https://doi.org/10.17632/zwr4ntf94j.7).
  - Zip SHA-256 `0d8fdb6a…0be0c24`, matching the Mendeley API.
  - Stored locally in `dataset/external/bdneuro_v7/` (git-ignored).
  - Processed to `data/processed/bdneuro_256/` with the same preprocessing as Figshare: 0.5–99.5 percentile scaling, resized to 256², INTER_AREA.
- **Models:** each of the 5 fold model sets (EXP-002c segmenter + seeds 1 and 2, and the EXP-003 classifiers). They are applied unchanged. Policy parameters are fitted on each fold's **clean Figshare val** set, with the score `ens:disagree` frozen in EXP-006.
- **Results:** [EXP-009-external/results.json](EXP-009-external/results.json). Raw: `runs/exp009_external/fold{k}.csv` (git-ignored); preliminary outputs in `runs/exp009_external_prelim_tta/`.
- **Limitation:** BDNeuro has **no masks and no patient IDs**. So there is no segmentation Dice, and the CIs are image-level bootstraps (2,000 resamples), which are likely too narrow if a patient contributes several images [I].

## Duplicate audit (before any evaluation) [V: data/processed/bdneuro_256/report.json]
- **Glioma / meningioma / pituitary images:** 5,154 in total, from all of BDNeuro's own splits. They come in 152 distinct native sizes, mostly 512² (3,293) and 224² (1,675). 3,358 are 3-channel.
- **pHash is too coarse for MRI slices.** 2,300 images are within Hamming distance 8 of some Figshare slice. But different Figshare *patients* are also that close to each other. The decision was therefore made by 64×64 Pearson correlation:
  - Between different Figshare patients the maximum correlation is 0.952 (p99 0.929).
  - **Duplicate** = correlation ≥ 0.96: **1,682** images. They match 215 of the 233 Figshare patients, with 99.5 % label agreement.
  - **Ambiguous** = 0.90–0.96: 442 images, excluded.
  - **Clean** = below 0.90: **3,030** images (glioma 1,380, meningioma 862, pituitary 788). This is the test set.
- **BDNeuro's own test split is contaminated:** 257 of its 771 tumor images (33 %) are duplicates of Figshare slices. Any model trained on Figshare and "externally validated" on that split would be partly scored on its own training data. [I] This is a reportable finding about the dataset.

## Results [V: results.json; "mean" and "range" are over the 5 fold model sets, n = 3,030 each]
| Policy | Bal. acc, mean [range] | AURC, mean [range] | acc @90 % coverage, mean |
|---|---|---|---|
| full + MSP | 0.904 [0.868, 0.917] | 0.0206 [0.0129, 0.0432] | 0.945 |
| **crop32 + MSP** | **0.931 [0.924, 0.934]** | 0.0151 [0.0140, 0.0162] | **0.967** |
| masked + MSP | 0.861 [0.849, 0.867] | 0.0556 | 0.897 |
| router | 0.922 [0.911, 0.933] | 0.0144 [0.0099, 0.0182] | 0.961 |
| router + combined | 0.922 | **0.0141** [0.0113, 0.0172] | 0.960 |
| oracle router (labels) | 0.960 [0.959, 0.962] | 0.0108 | 0.977 |

Per fold model set: the router's crop share, and the router minus each baseline (balanced accuracy, image-level 95 % CI):

| Fold models | Crop share | Router − crop32 | Router − full |
|---|---|---|---|
| 0 | 0.00 (τ1 = −∞) | −0.022 [−0.033, −0.013] | +0.000 |
| 1 | 0.19 | −0.010 [−0.020, −0.001] | +0.004 [−0.001, +0.008] |
| 2 | 0.03 | −0.022 [−0.032, −0.011] | +0.000 [−0.001, +0.002] |
| 3 | 0.89 | +0.008 [+0.003, +0.013] | +0.019 [+0.009, +0.028] |
| 4 | 0.93 | +0.001 [−0.004, +0.006] | +0.066 [+0.055, +0.076] |

Conformal referral (router, α = 0.1):
- Coverage is 0.837–0.885 and referral 0.086–0.133 across the fold models.
- Meningioma coverage is 0.734–0.825.
- At α = 0.05, crop32 coverage is 0.920–0.944.

## Conclusions
- **Performance holds on the external data.** The automatic crop-32 pipeline reaches 0.931 balanced accuracy on 3,030 non-duplicate external images, only −0.010 from in-distribution (0.941). The full-image classifier drops more: 0.928 → 0.904, with a fold spread of 0.868–0.917. So BDNeuro (clean part) is only a **mild shift** for this pipeline.
- **H-B3 is not supported on the external set.**
  - The router is between the crop and the full view on accuracy (−0.009 vs crop, +0.018 vs full). Its AURC equals the crop's (−0.0007).
  - The per-fold table shows why: the outcome is set by each fold's val-fitted τ1 ("almost always full" vs "almost always crop"), not by per-image routing. This is the same τ1 instability seen in EXP-006.
- **Conformal coverage degrades mildly** (0.84–0.89 at α = 0.1), consistent with a mild shift. The meningioma gap from EXP-007 persists.
