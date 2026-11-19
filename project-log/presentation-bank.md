# Presentation Bank

The strongest results, figures and "wow" points for the final presentation, collected as we go. Each item has a source.

## Story beats
1. **"We audited ourselves."** The semester-1 test set contained 60 training images (55 %). On those, the full-image classifier is 100 % correct; on the clean 50 it is 80 %. Source: `experiments/EXP-000-semester1-checkpoint-eval.md`. *Honest-evaluation slide; shows scientific maturity.*
2. **"Segmentation-guided classification failed in its naive form."** Predicted-ROI classifier: 0.46 on clean test, close to the majority-class rate (0.44); the full image gets 0.80. Motivates the semester-2 redesign. Source: EXP-000.

3. **Even the "clean" test set was not clean.** 43 of its 50 images came from patients already seen in training (verified with patient IDs). Reason for patient-wise evaluation. Source: `00-project-understanding.md §8.2`.

4. **"We are not alone: the field's benchmarks leak."** Slice-level splits inflate accuracy by 29–55 % (Yagis 2021). Kaggle merges: 28.8 % near-duplicates, 95.5 % patient overlap (Vangala 2026). Our patient-disjoint protocol is a strength. Source: `research/research-summary.md §1A`.
5. **"Even the dataset's authors knew context matters."** Cheng 2015: tight ROI 83.5 % → dilated ROI 88.2 %. This matches our ROI finding. Source: literature-matrix #1.
6. **Positioning slide:** grid of "common" vs "open" angles (`research-summary.md §1B–C`).

## Figures to make
- Bar chart: accuracy on contaminated vs clean test, for each of the 3 inputs (data in EXP-000 `results.json`).
- Diagram: image-level split vs patient-wise split (leakage illustration).
- Before/after repo hygiene: 225 MB → 210 KB (ADR-0001), as an optional engineering slide.
