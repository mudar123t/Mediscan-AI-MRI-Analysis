# Metrics Reloaded (+ CLAIM 2024, TRIPOD+AI) — evaluation note

- **Citation:** Maier-Hein L, Reinke A, Godau P, Tizabi MD, Buettner F, et al. Metrics Reloaded: recommendations for image analysis validation. *Nature Methods* 21(2):195–212 (2024). doi:10.1038/s41592-023-02151-z. arXiv:2206.01653 (v8, 2024-02-23).
- **Companion:** Reinke A, Tizabi MD, et al. Understanding metric-related pitfalls in image analysis validation. *Nature Methods* 21(2):182–194 (2024). doi:10.1038/s41592-023-02150-0. arXiv:2302.01790.
- **Verified links (2026-11-19):** https://pmc.ncbi.nlm.nih.gov/articles/PMC11182665/ ; https://arxiv.org/abs/2206.01653 ; https://arxiv.org/abs/2302.01790 ; Europe PMC records.

## Framework
Delphi-consensus framework; users build a "problem fingerprint" (domain interest, target structures, dataset, output) and get metric recommendations for image-level classification, object detection, semantic and instance segmentation. Online tool available.

## Recommendations we will apply (verified from the PMC full text)
- **Segmentation (our U-Net):** DSC (or IoU) + a boundary metric; NSD is the default boundary metric when reference annotations are noisy. HD95 is not mentioned in the visible text — report it only as secondary.
- **Aggregation:** compute per image, then aggregate; respect hierarchical structure (images within patients, hospitals) — we should also report per-patient means.
- **Classification (our 3-class CNN):** a multi-class counting metric (Balanced Accuracy or MCC depending on prevalence; Expected Cost if costs differ), AUROC as the multi-threshold default, per-class one-vs-rest metrics, calibration metrics if probabilities are used.
- **Reporting:** confidence intervals for all metrics (small test sets), full confusion matrix, raw-value plots, multiple runs for non-determinism, avoid ranking by closely related metrics.

## CLAIM 2024 (Tejani et al., Radiol Artif Intell 6(4):e240300, doi:10.1148/ryai.240300; PMC11304031)
44 items. Must-haves for us: item 7 (data source incl. public dataset), 19–20 (partitioning, "disjoint at the patient level or higher"), 14/16/17 (reference standard, annotators), 29 (significance tests), 33 (external test data), 37–38 (metrics with 95% CIs), 39 (failure analysis + confusion matrix), 43 (code/model availability).

## TRIPOD+AI (Collins et al., BMJ 385:e078378, 2024, doi:10.1136/bmj-2023-078378; PMC11019967)
27 main items + 13-item abstract checklist. Relevant: 5a (sources for development vs evaluation), 10 (sample size), 12a/12c (partitioning, internal validation), 12d/23b (cluster heterogeneity), 12e (calibration), 14 (fairness), 18e/18f (data/code), 23a (performance with CIs).

## Action items
1. Patient-disjoint 5-fold CV (cvind.mat); 95% CIs by patient-level bootstrap.
2. Segmentation: Dice, IoU, NSD (tolerance stated in mm), HD95; define behaviour for empty predictions.
3. Classification: balanced accuracy, macro-F1, MCC, AUROC, confusion matrix, reliability diagram/ECE.
4. External test (BDNeuro-MRI) reported separately.
5. Fill CLAIM checklist as thesis appendix.
