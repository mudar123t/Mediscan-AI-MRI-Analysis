# Presentation Bank

The strongest results, figures and "wow" points for the final presentation, collected as we go. Each item has a source.

## Story beats
1. **"We audited ourselves."** The semester-1 test set contained 60 training images (55 %). On those, the full-image classifier is 100 % correct; on the clean 50 it is 80 %. Source: `experiments/EXP-000-semester1-checkpoint-eval.md`. *Honest-evaluation slide; shows scientific maturity.*
2. **"Segmentation-guided classification failed in its naive form."** Predicted-ROI classifier: 0.46 on clean test, close to the majority-class rate (0.44); the full image gets 0.80. Motivates the semester-2 redesign. Source: EXP-000.

## Figures to make
- Bar chart: accuracy on contaminated vs clean test, for each of the 3 inputs (data in EXP-000 `results.json`).
- Diagram: image-level split vs patient-wise split (leakage illustration).
- Before/after repo hygiene: 225 MB → 210 KB (ADR-0001), as an optional engineering slide.
