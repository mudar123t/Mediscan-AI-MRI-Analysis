# ADR-0005 — Scope decisions after the future.txt analysis

- **Date:** 2026-11-19
- **Status:** Accepted (Kareem, Phase 3 discussion: "I agree, do what you recommended ... we have plenty of time to be ambitious and creative")

## Context
Phase 3 ([idea-matrix](../ideas/idea-matrix.md)) found that most `future.txt` items improve the baseline but are not novel. B6 (sub-region segmentation) is impossible on the Figshare data, because its masks are binary [V].

## Decision
1. **Drop B6** (multi-class edema/core/enhancing segmentation) for this dataset.
2. Treat P0, A1, A2, A3, B4, D10, E1–E6 and E14 as **mandatory groundwork** (the leak-free, strong semester-2 baseline). They are **not** presented as the novel contribution.
3. The novel contribution is chosen in Phase 5 from E13 / D12 / D11 and the Phase 4 research. **Time is not a binding constraint**: ambitious options are allowed.
4. Phase 4 uses 4 parallel research subagents (one per topic, T1–T4) and starts from the semester-1 literature review (approved by Kareem).

## Consequences
- Phase 6 will start with the groundwork milestone (the new baseline) before any contribution experiments.
- B6 stays in the idea matrix, marked "dropped", for traceability.
