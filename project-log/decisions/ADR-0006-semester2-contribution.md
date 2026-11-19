# ADR-0006 — Semester-2 contribution: segmentation-quality-aware routing (Option C, "B first")

- **Date:** 2026-11-19
- **Status:** Accepted (Kareem: "Option C, built as 'B first, then extras if needed' (we will decide about extras)")

## Context
The university wants a creative addition to a common U-Net→CNN pipeline (R1 in `university-requirements.md`), and the supervisor may co-publish. Phase 4 found that D11 and D12 are common and that classifier-only referral exists (Sharma 2026). We found no verified paper that uses segmentation quality to choose the classification view and to refer (`research/research-summary.md`). Our EXP-000 shows the naive predicted-ROI pipeline collapses (0.46 vs 0.80 full image). Options A–D are in `ideas/phase5-options.md`.

## Options considered
A safe (integrity + "when does seg help") · B core (R1 routing) · C ambitious (B + R2 MedSAM second opinion + R4 XAI diagnostic + R5 grounded report) · D product-centric. Scores are in `ideas/idea-matrix.md § Phase 5`.

## Decision
- **Option C, executed as "B first":**
  1. **Milestone 0:** leak-free strong baseline (P0, A1–A3, B4, D10, E4, E14 + Metrics-Reloaded metrics).
  2. **Option B:** R3 motivation study → R1 segmentation-quality-aware routing (dilated/context ROI view vs full-image view vs refer) + R7 conformal referral, evaluated with the R6 protocol (patient-disjoint folds, pHash audit, external BDNeuro-MRI, synthetic shift).
  3. **Extras R4, R2, R5:** each is **decided separately with Kareem after B is evaluated**. None is committed yet.
- Working title [I]: "Knowing when to trust the mask: segmentation-quality-aware routing for brain-tumor classification".

## Consequences
- The Phase 6 plan covers Milestone 0 and Option B in detail; extras appear only as placeholders.
- Hypotheses H-B1..H-B3 (phase5-options.md) define success. Negative outcomes are reported, not hidden.
- Before any paper submission, a Scopus/Google Scholar novelty re-check is needed (our searches were limited).
