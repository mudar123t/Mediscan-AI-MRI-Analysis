"""Mediscan AI — M2 local demo (Gradio). Research prototype, not a medical device.

Upload one brain-MRI slice (T1 contrast-enhanced, like the Figshare set). The app shows the
predicted tumour mask, the segmentation-uncertainty map, the routing decision (32 px context
crop vs full image) with its reason, the calibrated class probabilities and the conformal
referral decision. Models and policy parameters: one cross-validation fold (configs/demo.yaml).

Usage (from the repository root):
    .venv/Scripts/python app/demo_app.py [--config configs/demo.yaml] [--device cpu] [--port 7860]
Needs the trained weights in runs/ (git-ignored) and app/params_<fold>.json
(scripts/export_demo_params.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np

from mediscan.demo import (CLASS_NAMES, analyse, crop_box_view, load_params, overlay_mask, overlay_uncertainty,
                           preprocess_upload)
from mediscan.pipeline import FoldModels
from mediscan.utils.repro import load_config

DISCLAIMER = ("**Research prototype — not a medical device.** Outputs must not be used for diagnosis. "
              "Trained on the public Figshare brain-tumour set (Cheng et al., CC BY 4.0); "
              "expects one T1 contrast-enhanced slice showing a glioma, meningioma or pituitary tumour.")


def build_app(models: FoldModels, params: dict, alpha_default: float, examples: list[Path]) -> gr.Blocks:
    def run(image: np.ndarray | None, normalise: bool, alpha: str):
        if image is None:
            raise gr.Error("Upload a slice first.")
        x = preprocess_upload(image) if normalise else _as_256_gray(image)
        r = analyse(x, models, params, float(alpha))
        view = crop_box_view(x, r.mask) if r.used_crop else x
        label = {CLASS_NAMES[c]: float(r.probs["routed"][c]) for c in range(3)}
        flag = "🔴 **REFER to a radiologist**" if r.refer else "🟢 **No referral**"
        cset = ", ".join(CLASS_NAMES[c] for c in r.conformal_set) or "∅ (empty)"
        md = (f"### Prediction: {CLASS_NAMES[r.pred]} (calibrated confidence {r.confidence:.2f})\n"
              f"{flag}\n\n"
              f"**Routing.** {r.reason_route}\n\n"
              f"**Referral.** {r.reason_refer} Conformal set: {{{cset}}}.\n\n"
              f"<small>Crop-view probabilities: "
              + ", ".join(f"{CLASS_NAMES[c]} {r.probs['crop32'][c]:.2f}" for c in range(3))
              + " · full-image probabilities: "
              + ", ".join(f"{CLASS_NAMES[c]} {r.probs['full'][c]:.2f}" for c in range(3))
              + f" · predicted tumour area {int(r.mask.sum())} px · fold {params['fold']} models.</small>")
        return overlay_mask(x, r.mask), overlay_uncertainty(x, r.uncertainty), view, label, md

    with gr.Blocks(title="Mediscan AI demo") as demo:
        gr.Markdown("# Mediscan AI — segmentation-quality-aware routing\n" + DISCLAIMER)
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="MRI slice", type="numpy", image_mode="L", height=300)
                norm = gr.Checkbox(value=True, label="Apply intensity normalisation (untick for the provided "
                                                     "examples, which are already preprocessed)")
                alpha = gr.Radio(choices=["0.1", "0.05"], value=str(alpha_default),
                                 label="Conformal error level α (lower = refer more)")
                btn = gr.Button("Analyse", variant="primary")
                if examples:
                    gr.Examples(examples=[[str(p), False, str(alpha_default)] for p in examples],
                                inputs=[inp, norm, alpha], label="Examples (test fold, never seen by these models)")
            with gr.Column(scale=2):
                with gr.Row():
                    out_mask = gr.Image(label="Predicted tumour mask", height=256)
                    out_unc = gr.Image(label="Segmentation uncertainty (ensemble entropy)", height=256)
                    out_view = gr.Image(label="What the classifier saw", height=256)
                out_label = gr.Label(label="Class probabilities (calibrated, routed view)", num_top_classes=3)
                out_md = gr.Markdown()
        btn.click(run, inputs=[inp, norm, alpha], outputs=[out_mask, out_unc, out_view, out_label, out_md])
    return demo


def _as_256_gray(image: np.ndarray) -> np.ndarray:
    """Use an already-preprocessed slice as is (only greyscale conversion / resize if needed)."""
    img = np.asarray(image)
    if img.ndim == 3:
        img = img[..., :3].mean(-1)
    if img.shape != (256, 256):
        import cv2

        img = cv2.resize(img.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    cfg = load_config(args.config)
    params = load_params(Path(cfg["params_out"]))
    models = FoldModels(int(cfg["fold"]), Path("runs"), [params["score"].split(":")[0]], params["ensemble_runs"],
                        device=args.device, views=("crop32", "full"))
    examples = sorted(Path(cfg["examples_dir"]).glob("*.png"))
    print(f"models of fold {cfg['fold']} loaded on {models.device}; {len(examples)} examples")
    build_app(models, params, float(cfg["alpha_default"]), examples).launch(server_name="127.0.0.1",
                                                                            server_port=args.port)


if __name__ == "__main__":
    main()
