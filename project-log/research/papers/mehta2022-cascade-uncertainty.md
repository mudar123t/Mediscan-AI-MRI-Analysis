# Mehta et al. 2022 — Propagating Uncertainty Across Cascaded Medical Imaging Tasks

- **Verified URLs:**
  - https://oatml.cs.ox.ac.uk/publications/202109_Gal_Propagating_Uncertainty_Medical_Imaging.html
  - https://api.crossref.org/works/10.1109/TMI.2021.3114097

  Both opened 2026-11-19. The IEEE Xplore page returned empty.
- **Citation:** Mehta R, Christinck T, Nair T, Bussy A, Premasiri S, Costantino M, Chakravarthy MM, Arnold DL, Gal Y, Arbel T. *IEEE Transactions on Medical Imaging* 41(2):360–373, 2022. DOI 10.1109/TMI.2021.3114097.
- **BibTeX key:** `mehta2022propagating`

## Idea
Cascaded pipelines that pass only deterministic outputs compound errors. Passing the upstream **uncertainty map** to the downstream task improves it.

## Experiments (from the OATML abstract)
1. T2 lesion segmentation uncertainty → MS lesion detection (large multi-site clinical-trial data): improved detection.
2. Uncertainty map of a synthesized missing MR volume → brain tumor segmentation (BraTS-2018): improved segmentation.
3. Hippocampus segmentation uncertainty → Alzheimer's clinical-score regression: improved regression.

No numbers were extracted from the page.

## Relevance to Mediscan E13
- **Closest prior art** for "segmentation uncertainty informs a downstream stage." Cite it as the conceptual foundation.
- Difference from E13 (inference):
  - Mehta feeds uncertainty *as an input channel* to a retrained downstream network.
  - E13 uses uncertainty as a *decision variable*: route to ROI vs full image, or refer.
  - Mehta does not address tumor-type classification, abstention, or risk–coverage.
- A cheap ablation inspired by this paper: give the classifier the uncertainty map as an extra channel ("Mehta-style"), and compare it with "E13-style" gating. This is feasible in 2D on 4 GB.
