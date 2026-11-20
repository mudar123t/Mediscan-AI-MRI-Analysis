"""End-to-end inference of the routed pipeline on arbitrary images (EXP-008 shift, EXP-009 external, demo).

For one fold: EXP-002c segmenter -> reference mask (0.5 + largest component) ->
segmentation-uncertainty scores -> logits of the full, crop-32 and masked-ROI
classifiers (crop/masked views built from the reference mask). The output frame has
the columns expected by ``mediscan.policy_eval`` (``l_<view>_<c>``, ``<method>:<score>``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mediscan.metrics.seg import largest_component
from mediscan.models.cls import build_cls_model
from mediscan.models.seg import build_seg_model, enable_mc_dropout
from mediscan.uncertainty import SCORE_NAMES, image_scores, mc_dropout_samples, tta_samples
from mediscan.utils.repro import load_config
from mediscan.views import build_view

CLS_RUNS = {"full": "exp003_r18_full", "crop32": "exp003_r18_crop32", "masked": "exp003_r18_masked"}


def _seg(run_dir: Path, device: str, dropout: float = 0.2) -> torch.nn.Module:
    m = build_seg_model("unet_r34", pretrained=False, dropout=dropout)
    m.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    return m.to(device).eval()


def _cls(run_dir: Path, device: str) -> tuple[torch.nn.Module, object]:
    cfg = load_config(run_dir / "config.yaml")
    m = build_cls_model(cfg["model"]["name"], **{**cfg["model"].get("kwargs", {}), "pretrained": False})
    m.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    return m.to(device).eval(), build_view(cfg["view"]["name"], cfg["view"].get("margin"))


@torch.no_grad()
def run_fold(
    images: np.ndarray,
    fold: int,
    runs: Path,
    methods: list[str],
    ensemble_runs: list[str],
    mc_samples: int = 20,
    batch: int = 16,
    gt_masks: np.ndarray | None = None,
) -> pd.DataFrame:
    """Frame with logits, uncertainty scores (and Dice vs ``gt_masks`` if given) for ``images`` (N, H, W) uint8."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    seg = _seg(runs / "exp002c_unet_r34" / f"fold{fold}", device)
    mc_model = None
    if "mc" in methods:
        mc_model = _seg(runs / "exp002c_unet_r34" / f"fold{fold}", device)
        enable_mc_dropout(mc_model)
    ens = [_seg(runs / r / f"fold{fold}", device) for r in ensemble_runs] if "ens" in methods else []
    classifiers = {v: _cls(runs / r / f"fold{fold}", device) for v, r in CLS_RUNS.items()}

    recs: list[dict] = []
    for i in range(0, len(images), batch):
        imgs = images[i : i + batch]
        x = torch.from_numpy(imgs.astype(np.float32) / 255.0)[:, None].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            p0 = torch.sigmoid(seg(x).float())[:, 0]
        p0 = p0.cpu().numpy()
        ref = np.stack([largest_component((p > 0.5).astype(np.uint8)) for p in p0])
        samples = {"det": p0[None]}
        if mc_model is not None:
            samples["mc"] = mc_dropout_samples(mc_model, x, mc_samples, amp).cpu().numpy()
        if "tta" in methods:
            samples["tta"] = tta_samples(seg, x, amp).cpu().numpy()
        if ens:
            outs = []
            for m in ens:
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    outs.append(torch.sigmoid(m(x).float())[:, 0])
            samples["ens"] = torch.stack(outs).cpu().numpy()
        logit = {}
        for v, (model, view) in classifiers.items():
            xv = np.stack([view(im, mk) for im, mk in zip(imgs, ref)]).astype(np.float32) / 255.0
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logit[v] = model(torch.from_numpy(xv)[:, None].to(device)).float().cpu().numpy()
        for j in range(len(imgs)):
            r: dict = {"idx": i + j, "pred_area": int(ref[j].sum())}
            for m in methods:
                sc = image_scores(samples[m][:, j], ref[j])
                sc["maxprob_ring"] = -sc["maxprob_ring"]  # higher = less trustworthy, as in policy_eval
                r.update({f"{m}:{k}": sc[k] for k in SCORE_NAMES})
            for v in CLS_RUNS:
                r.update({f"l_{v}_{c}": float(logit[v][j, c]) for c in range(3)})
            if gt_masks is not None:
                from mediscan.metrics.seg import dice

                r["dice_lcc"] = dice(ref[j], gt_masks[i + j])
            recs.append(r)
    return pd.DataFrame(recs)
