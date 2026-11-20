# ADR-0008 — M1.4 gate: stabilise τ1 and add CIs to the shift summary

- **Date:** 2026-11-20
- **Status:** Accepted (work not started)
- **Context:** M1 is complete (EXP-004 … EXP-009; [HANDOFF](../HANDOFF.md)).
  - H-B1 is not supported.
  - H-B2 is supported against the full-image MSP gate only.
  - H-B3 is partly supported (synthetic noise/blur yes, external no).
  - The weakest point is the router threshold τ1: fitted per fold on one val fold, it swings between "always full" and "almost always crop" (val crop share 0.00–0.94). It also decides the external result (EXP-009 per-fold table).
  - The EXP-008 worst-case/mean-over-conditions summary has no CIs.
- **Options presented at the gate:**
  - (a) accept M1 as is;
  - (b) stabilise τ1;
  - (c) class-conditional (Mondrian) conformal prediction for the meningioma coverage gap;
  - (d) CIs for the EXP-008 summary.

## Decision (Kareem, 2026-11-20)
- **(b) + (d)**, as recommended. Then move to M2.
- (c) is not chosen for now.
- **Nothing is to be run yet.** Kareem will say when.

## Consequences
- **(b) is a protocol change after seeing test results.**
  - It must be reported as a separate, clearly labelled analysis: "EXP-006b / stabilised τ1". The frozen-protocol EXP-006/008/009 results stay the primary results and are not overwritten.
  - The fitting rule must use only val data. Candidate rules, to be fixed in writing before running:
    - fit τ1 on the pooled val frames of all folds except the test fold;
    - or a fixed quantile / Dice-proxy rule.
- **(d)** is the paired patient bootstrap of the min and mean over the 13 EXP-008 conditions (same resamples across conditions). No new inference; CPU only, about 30 min [I].
- Branch `exp/m1-uncertainty` is pushed to `origin` at this gate. `main` was fast-forwarded to the branch at this gate.
