"""M2 demo logic: one uploaded slice -> mask, uncertainty, routing decision, class, referral.

Pure functions around ``mediscan.pipeline.FoldModels`` so the Gradio app stays thin.
The policy parameters (temperatures, tau1, conformal thresholds) are the EXP-006
frozen-protocol values of one fold, fitted on that fold's validation data and exported by
``scripts/export_demo_params.py``. Research prototype, not a medical device.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mediscan.data.figshare import normalize_intensity, to_uint8
from mediscan.metrics.cls import softmax

CLASS_NAMES = ("meningioma", "glioma", "pituitary")


def preprocess_upload(img: np.ndarray, size: int = 256, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    """Any greyscale/RGB(A) image -> uint8 (size, size), exactly as the data pipeline (EXP-001, BDNeuro)."""
    img = np.asarray(img)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img = img[..., 0]
    h, w = img.shape
    x = normalize_intensity(img, low_pct, high_pct)
    if (h, w) != (size, size):
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA if max(h, w) > size else cv2.INTER_LINEAR)
    return to_uint8(x)


def load_params(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class DemoResult:
    image: np.ndarray                 # uint8 (H, W), preprocessed input
    mask: np.ndarray                  # uint8 (H, W) reference mask (0.5 + largest component)
    uncertainty: np.ndarray           # float (H, W) entropy (bits) of the ensemble-mean probability
    u: float                          # image-level score (frozen EXP-006 score); NaN-filled
    tau1: float
    used_crop: bool
    probs: dict[str, np.ndarray]      # calibrated class probabilities per view, and "routed"
    pred: int
    confidence: float
    conformal_set: list[int]
    alpha: float
    refer: bool
    reason_route: str
    reason_refer: str
    extra: dict[str, Any] = field(default_factory=dict)


def _entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def decide(rec: dict, params: dict[str, Any], alpha: float) -> dict[str, Any]:
    """Routing + calibrated probabilities + conformal referral for one record of ``FoldModels.infer``."""
    score = params["score"]
    u_raw = rec.get(score, float("nan"))
    u = float(params["fill"]) if u_raw is None or not np.isfinite(u_raw) else float(u_raw)
    tau1 = float(params["tau1"])
    probs = {}
    for v in ("crop32", "full"):
        lg = np.array([rec[f"l_{v}_{c}"] for c in range(3)], dtype=float)
        probs[v] = softmax(lg[None] / float(params["temps"][v]))[0]
    used_crop = bool(u <= tau1)
    probs["routed"] = probs["crop32"] if used_crop else probs["full"]
    q = float(params["q"][f"router@{alpha}"])
    cset = [int(c) for c in np.flatnonzero(1.0 - probs["routed"] <= q)]
    pred = int(np.argmax(probs["routed"]))
    if used_crop:
        route = (f"Segmentation uncertainty u = {u:.3f} ≤ τ1 = {tau1:.3f}: the mask is trusted, "
                 f"so the **32 px context crop** around it is classified.")
    else:
        route = (f"Segmentation uncertainty u = {u:.3f} > τ1 = {tau1:.3f}: the mask is not trusted, "
                 f"so the **full image** is classified instead.")
    if rec.get("pred_area", 1) == 0:
        route = "The segmenter found no tumour region (u set to the most suspicious value). " + route
    if len(cset) == 1:
        refer, why = False, f"Conformal set at α = {alpha} is a single class ({CLASS_NAMES[cset[0]]}): no referral."
    elif len(cset) == 0:
        refer, why = True, (f"No class reaches the conformal threshold at α = {alpha} (atypical case): "
                            "**refer to a radiologist**.")
    else:
        names = " / ".join(CLASS_NAMES[c] for c in cset)
        refer, why = True, f"The conformal set at α = {alpha} is ambiguous ({names}): **refer to a radiologist**."
    return {"u": u, "tau1": tau1, "used_crop": used_crop, "probs": probs, "pred": pred,
            "confidence": float(probs["routed"][pred]), "conformal_set": cset, "refer": refer,
            "reason_route": route, "reason_refer": why}


def analyse(image_u8: np.ndarray, models: Any, params: dict[str, Any], alpha: float = 0.1) -> DemoResult:
    """Run the routed pipeline on one preprocessed uint8 (256, 256) slice."""
    recs, arrays = models.infer(image_u8[None])
    d = decide(recs[0], params, alpha)
    method = params["score"].split(":")[0]
    samples = arrays.get(f"samples_{method}", arrays["samples_det"])[:, 0]
    unc = _entropy_bits(samples.mean(0))
    return DemoResult(image=image_u8, mask=arrays["ref"][0].astype(np.uint8), uncertainty=unc, alpha=alpha,
                      extra={"record": recs[0]}, **{k: d[k] for k in ("u", "tau1", "used_crop", "probs", "pred",
                                                                      "confidence", "conformal_set", "refer",
                                                                      "reason_route", "reason_refer")})


def overlay_mask(image_u8: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (42, 120, 214)) -> np.ndarray:
    """RGB image with a 2 px mask contour and a light fill."""
    rgb = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2RGB)
    if mask.any():
        fill = rgb.copy()
        fill[mask > 0] = color
        rgb = cv2.addWeighted(fill, 0.25, rgb, 0.75, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contours, -1, color, 2)
    return rgb


def overlay_uncertainty(image_u8: np.ndarray, unc: np.ndarray, color: tuple[int, int, int] = (235, 104, 52)) -> np.ndarray:
    """RGB image tinted by pixel uncertainty (0 = transparent, 1 bit = full colour); one hue, light to strong."""
    rgb = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2RGB).astype(np.float32)
    a = np.clip(unc, 0, 1)[..., None] * 0.85
    out = rgb * (1 - a) + np.array(color, np.float32) * a
    return out.astype(np.uint8)


def crop_box_view(image_u8: np.ndarray, mask: np.ndarray, margin: int = 32) -> np.ndarray:
    """The crop the classifier saw (for display), or the full image if the mask is empty."""
    from mediscan.views import context_crop

    return context_crop(image_u8, mask, margin)
