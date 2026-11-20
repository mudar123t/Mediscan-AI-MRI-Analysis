"""MedSAM second opinion for EXP-011 (R2): box-prompted MedSAM masks and disagreement signals.

Model: ``wanglab/medsam-vit-base`` (transformers ``SamModel``, Apache-2.0). Preprocessing follows
that port's processor config (ImageNet normalisation after [0, 1] scaling); chosen on val
slices with GT boxes (EXP-011 preprocessing check). Pure helpers are separated from the model
so they can be tested without weights.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from mediscan.metrics.seg import dice

MODEL_ID = "wanglab/medsam-vit-base"
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
SIGNALS = ("sam:disagree", "sam:jitter", "sam:selfvar")


def to_pixel_values(img_u8: np.ndarray) -> np.ndarray:
    """(H, W) uint8 -> (3, 1024, 1024) float32: cubic resize, min-max to [0, 1], ImageNet normalisation."""
    x = cv2.resize(img_u8.astype(np.float32), (1024, 1024), interpolation=cv2.INTER_CUBIC)
    x = (x - x.min()) / max(float(x.max() - x.min()), 1e-8)
    return np.stack([(x - m) / s for m, s in zip(_MEAN, _STD)]).astype(np.float32)


def box_from_mask(mask: np.ndarray, scale: float = 4.0) -> np.ndarray | None:
    """[x0, y0, x1, y1] (end-exclusive) of the mask, scaled to the 1024 grid; None if empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32) * scale


def jitter_boxes(box: np.ndarray, rng: np.random.Generator, k: int = 8, frac: float = 0.1, size: int = 1024) -> np.ndarray:
    """k boxes with each coordinate moved by U(-frac, frac) x box width (x) / height (y), clipped; (k, 4)."""
    w, h = box[2] - box[0], box[3] - box[1]
    d = rng.uniform(-frac, frac, size=(k, 4)) * np.array([w, h, w, h], dtype=np.float32)
    out = np.clip(box[None] + d, 0, size).astype(np.float32)
    x0, x1 = np.minimum(out[:, 0], out[:, 2]), np.maximum(out[:, 0], out[:, 2])
    y0, y1 = np.minimum(out[:, 1], out[:, 3]), np.maximum(out[:, 1], out[:, 3])
    return np.stack([x0, y0, np.maximum(x1, x0 + 1), np.maximum(y1, y0 + 1)], axis=1)


def signals(ref: np.ndarray, masks: np.ndarray) -> dict[str, float]:
    """EXP-011 signals from the U-Net reference mask and MedSAM masks (9, H, W): index 0 = original box."""
    d0 = dice(masks[0], ref)
    dj = [dice(m, ref) for m in masks[1:]]
    pair = [dice(masks[i], masks[j]) for i in range(len(masks)) for j in range(i + 1, len(masks))]
    return {"sam:disagree": 1.0 - d0, "sam:jitter": 1.0 - float(np.mean(dj)), "sam:selfvar": 1.0 - float(np.mean(pair))}


class MedSAM:
    """Thin wrapper: one image embedding, then any number of box prompts."""

    def __init__(self, device: str | None = None) -> None:
        from transformers import SamModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SamModel.from_pretrained(MODEL_ID).to(self.device).eval()

    @torch.no_grad()
    def embed(self, img_u8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(to_pixel_values(img_u8))[None].to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            return self.model.get_image_embeddings(x)

    @torch.no_grad()
    def segment(self, emb: torch.Tensor, boxes: np.ndarray) -> np.ndarray:
        """Binary masks (n, 256, 256) for n boxes (n, 4) in 1024 coordinates."""
        b = torch.from_numpy(boxes.astype(np.float32))[None].to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            out = self.model(image_embeddings=emb, input_boxes=b, multimask_output=False)
        return (torch.sigmoid(out.pred_masks[0, :, 0].float()) > 0.5).cpu().numpy().astype(np.uint8)

    def analyse(self, img_u8: np.ndarray, refs: list[np.ndarray], rngs: list[np.random.Generator],
                gt: np.ndarray | None = None) -> list[dict[str, float]]:
        """For one image and several reference masks (e.g. one per fold): signals (+ Dice vs GT if given)."""
        emb = self.embed(img_u8)
        out = []
        for ref, rng in zip(refs, rngs):
            box = box_from_mask(ref)
            if box is None:
                out.append({s: float("nan") for s in SIGNALS} | ({"medsam_dice_gt": float("nan")} if gt is not None else {}))
                continue
            masks = self.segment(emb, np.concatenate([box[None], jitter_boxes(box, rng)]))
            r = signals(ref, masks)
            if gt is not None:
                r["medsam_dice_gt"] = dice(masks[0], gt)
            out.append(r)
        return out
