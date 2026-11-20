# ADR-0009 — M2 gate: demo approved; demo weights published as a GitHub Release

- **Date:** 2026-11-20
- **Status:** Accepted
- **Context:**
  - The M2 demo (`app/demo_app.py`) needs the fold-3 weights, which are not in git (ADR-0001).
  - Plan §5 M2 criterion: "demo runs locally from a fresh clone + README instructions".
  - Options presented: (a) GitHub Release asset, (b) cloud-drive link, (c) "train first" in the README.

## Decision (Kareem, 2026-11-20)
- **The demo is approved.** Kareem's feedback: "looks good". He shared two screenshots, now saved as `images/demo_ui_referred.png` and `images/demo_ui_confident.png`.
- **(a) GitHub Release.**
  - Tag `demo-weights-fold3-v1` on `main` `e2e062f`.
  - Asset `mediscan-demo-weights-fold3.zip`: 355.6 MB, 12 files, SHA-256 `da2c6928b294cf7a14e4df5c18b6dca1914684970f3cc9e363c2e248db1cf8eb`.
  - URL: https://github.com/mudar123t/Mediscan-AI-MRI-Analysis/releases/tag/demo-weights-fold3-v1
- **Contents:** 3 U-Net segmenters (seeds 0/1/2), the crop-32 and full-image ResNet-18 classifiers with their `config.yaml`, and 5 example slices from test fold 3 (Figshare, CC BY 4.0, attributed).
  - The unused masked-ROI classifier is left out: `FoldModels(views=...)` now loads only what is needed.

## Consequences
- [V] **Fresh-clone check passed:** clone → `gh release download` → checksum matches → unzip → the demo pipeline from the clone's code reproduces the 5 example decisions.
- The weights are public. The repository has **no LICENSE file** yet. Choosing a licence for the code and weights is an open question for Kareem.
- Other folds' weights stay local.
