# Phase 5 — Options for the semester-2 contribution

- **Date:** 2026-11-19
- **Status:** proposed; waiting for Kareem's choice.
- **Inputs:** scores in [idea-matrix.md § Phase 5](idea-matrix.md), evidence in [research-summary.md](../research/research-summary.md), baseline in [EXP-000](../experiments/EXP-000-semester1-checkpoint-eval.md).
- **Labels:** [V] verified (source named) · [I] inference · [U] unknown.

**No numeric results are promised.** "Targets" are levels taken from verified literature. "Hypotheses" are what we will test, and any of them can fail. A failed hypothesis is still reported (thesis material).

## Common to every option: Milestone 0 — the leak-free, strong baseline (groundwork)
P0 + A1 + A2 + A3 + B4 + D10 + E4 + E14 + Metrics-Reloaded metrics:
- Full Figshare dataset converted from the raw `.mat` (nearest-neighbour mask resize).
- Patient-disjoint 5-fold CV (`cvind.mat`); loaders read split files.
- Improved U-Net (attention, pretrained encoder, longer training, augmentation).
- Pretrained classifier (ResNet-18 / EfficientNet-B0).
- Seeds, configs and a recorded environment.
- CUDA PyTorch install.
- **Targets [V refs]:** Dice in the region of 0.85–0.93 (Avazov 2024: 0.93 on a patient split). Classification at or above Cheng 2015's 91.28 % patient-wise, with Deepak 2019's 98 % as an upper reference.
- **Effort [I]:** about 8–12 working days plus GPU time. No cloud.
- This milestone alone fixes semester 1's validity problems.

---

## Option A — SAFE: "Honest Mediscan": integrity benchmark + "when does segmentation help?"
**Pitch.** We show, rigorously, *when* segmentation actually helps tumor-type classification. The method:
- Patient-disjoint evaluation of the full seg→cls pipeline.
- A systematic study across mask quality (GT → degraded → predicted) and context margin (tight → dilated → full image) (R3).
- A pHash duplicate audit and an external test on BDNeuro-MRI (R6).
- Calibration and conformal prediction sets (R7).

**Ideas used:** Milestone 0 + R3 + R6 + R7 (+ D11 ablation).

**Expected findings (hypotheses):**
- (H-A1) Accuracy drops as mask quality drops. The ROI-from-predicted-mask gap shrinks once the U-Net is good.
- (H-A2) An optimal context margin exists (Cheng 2015 found about 8 px for hand-crafted features [V]).
- (H-A3) Performance drops on the external set.

**Evaluation:**
- Balanced accuracy, macro-F1 and AUROC per view and per mask-quality level, with patient-bootstrap CIs.
- Internal vs external comparison.
- ECE before and after temperature scaling.
- Conformal coverage and set size.

**Novelty for the committee [I]:** medium-low. Strong rigour, but no new *method*; the committee may still see it as "evaluation only".

**Risks:** low technical risk. The risk is the "creative" perception.

**Effort [I]:** Milestone 0 + about 8–10 days.

---

## Option B — RECOMMENDED: "Trust-aware Mediscan": segmentation-quality-aware routing (R1)
**Pitch.** "The system knows when to trust its own segmentation." For each MRI slice, Mediscan estimates how reliable its tumor mask is, using U-Net uncertainty from MC dropout, test-time augmentation and disagreement across the 5 fold models. It then **routes** the case:
1. **Trusted mask** → classify the tumor with peritumoral context (dilated ROI view).
2. **Untrusted mask** → classify from the full image.
3. **Both uncertain** → **refer to a radiologist**, with a conformal coverage guarantee (R7).

R3 (the "when does segmentation help?" study) is the motivation chapter. R6 (integrity + external test) is the evaluation protocol.

**Why it is creative and new:**
- Prior work always trusts the mask (T1), passes uncertainty only as an *input* (Mehta 2022), predicts segmentation failure without a downstream classifier (DeVries 2018), or refers using *classifier* uncertainty only (Sharma 2026) [V].
- We found no paper that uses segmentation quality to choose the classification **view** and to refer [I; limited search].
- It turns our own semester-1 failure (0.46) into the research question.

**Hypotheses:**
- (H-B1) At 100 % coverage, the router's balanced accuracy is at least as good as the best single view.
- (H-B2) A segmentation-uncertainty gate gives a lower AURC/AUGRC than a classifier-only max-softmax gate, which is the Sharma-style baseline [V that this is the required baseline: Jaeger 2023].
- (H-B3) The benefit is larger under domain shift (BDNeuro-MRI external set; synthetic corruptions such as noise, blur and contrast). There, segmentation fails more and routing matters most [I].

**Evaluation (T2 §4):**
- Baselines: always-full, always-ROI(GT), always-ROI(pred), always-dilated-ROI(pred), the max-softmax gate, and an oracle router (upper bound).
- Gate signals compared by failure-detection AUROC for *classification* error.
- Risk–coverage curves, AURC, AUGRC, accuracy at 95/90/80 % coverage, and referral rate.
- ECE and Brier.
- All on patient-disjoint folds with patient-bootstrap CIs and paired tests, plus the external test.

**Demo:** upload a slice → see the mask overlay, an uncertainty heat-map, the routing decision with its reason, the class with a calibrated confidence, and a "Refer" flag.

**Risks and mitigations:**
- *Once the U-Net is strong (Dice about 0.9), the gate may rarely trigger in-distribution, and the full-image classifier may already be near its ceiling* [I]. → Mitigation: H-B3 (shift and corruption) is where routing should matter. Report honestly if in-distribution gains are small; the referral and selective-prediction gains remain measurable.
- *Novelty could be found later.* → Before writing a paper, Kareem or the supervisor runs a Scopus/Google Scholar check with university access. We frame the work as "segmentation-quality-aware view selection + referral", not as "first uncertainty".
- *Compute:* 5 folds × (U-Net + 2 view classifiers) plus T = 10–20 stochastic passes. Feasible on 4 GB at 256² [I]. Training time to be measured in Milestone 0.

**Effort [I]:** Milestone 0 + about 12–16 days. No cloud.

---

## Option C — AMBITIOUS: Option B + foundation-model second opinion + XAI diagnostic + grounded report
**Pitch.** Everything in B, plus three extras:
- **R2:** a frozen MedSAM/LiteMedSAM, box-prompted by the U-Net, acts as an independent "second reader". U-Net–SAM disagreement and SAM box-jitter spread become extra routing signals.
- **R4:** using the GT masks, we measure *where the classifier looks* (in the tumor / peritumoral ring / elsewhere) and test whether this predicts errors and agrees with the routing.
- **R5:** a grounded 2D report. All numbers are computed deterministically: area, location, class probabilities, uncertainty and routing reason. Wording comes from a template (an optional local LLM may only rephrase), and an automatic number-consistency check guards against hallucination.

**Ideas used:** B + R2 + R4 + R5.

**Extra hypotheses:**
- (H-C1) Cross-model disagreement adds failure-detection power beyond U-Net-only uncertainty.
- (H-C2) Cases where attribution falls outside the tumor and its ring are more often misclassified.

**Novelty [I]:** highest. Several fresh angles are combined, which gives the best publication potential and the most impressive demo.

**Risks:**
- Integration effort.
- SAM is weak with poor boxes (Dice 0.56–0.68 with realistic boxes, Ali 2024 [V]).
- Scope creep.
- R5 is hard to measure quantitatively.

→ Mitigation: build in strict order **B first**, then R4, R2, R5. Each extra is optional and independently reportable.

**Effort [I]:** Milestone 0 + about 22–30 days. No cloud required. MedSAM runs inference-only on 4 GB [I, T3]; an optional local LLM (about 4-bit 4B) is borderline on the GPU and works on CPU.

---

## Option D — PRODUCT-CENTRIC: "Clinician assistant" (demo-first)
**Pitch.** Focus on a usable clinical tool:
- Milestone 0 model.
- Interactive mask correction (human-in-the-loop).
- Grounded report (R5).
- XAI views (R4).
- A clean web UI, with a small usability questionnaire with the supervisor.

**Novelty [I]:** low-medium. HITL and report generation exist (Hart 2026, BTReport 2026 [V]). The 2D grounded report is the main fresh piece.

**Risks:** weaker research contribution and harder to publish. A clinician study is limited without radiologists.

**Effort [I]:** Milestone 0 + about 15–20 days.

---

## Summary
| | A Safe | **B Recommended** | C Ambitious | D Product |
|---|---|---|---|---|
| Novelty for committee [I] | ★★ | ★★★★ | ★★★★★ | ★★ |
| Technical risk [I] | low | medium | medium-high | medium |
| Measurable vs baseline | yes | yes (AURC, selective acc, external) | yes (+ H-C1/H-C2) | weak |
| Demo impact | ★★ | ★★★★ | ★★★★★ | ★★★★★ |
| Publication potential [I] | low-medium | medium-high | high | low |
| Effort beyond Milestone 0 [I] | ~8–10 d | ~12–16 d | ~22–30 d | ~15–20 d |
| Cloud needed | no | no | no | no |

**Recommendation [I]:** **Option C, executed as "B first, then extras"**. Since time is not binding (ADR-0005), C gives the best novelty and demo. Because B is completed and evaluated first, the project is never left half-done: if any extra fails, B still stands on its own. If you prefer lower risk, choose **B**.
