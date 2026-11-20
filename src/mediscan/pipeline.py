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


class FoldModels:
    """All models of one fold, loaded once (batch runs and the M2 demo).

    ``views`` limits the classifiers loaded (the demo needs only crop32 and full).
    """

    def __init__(self, fold: int, runs: Path, methods: list[str], ensemble_runs: list[str],
                 device: str | None = None, mc_samples: int = 20, views: tuple[str, ...] = tuple(CLS_RUNS)) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = self.device == "cuda"
        self.methods, self.mc_samples = list(methods), mc_samples
        self.seg = _seg(runs / "exp002c_unet_r34" / f"fold{fold}", self.device)
        self.mc_model = None
        if "mc" in methods:
            self.mc_model = _seg(runs / "exp002c_unet_r34" / f"fold{fold}", self.device)
            enable_mc_dropout(self.mc_model)
        self.ens = [_seg(runs / r / f"fold{fold}", self.device) for r in ensemble_runs] if "ens" in methods else []
        self.classifiers = {v: _cls(runs / CLS_RUNS[v] / f"fold{fold}", self.device) for v in views}

    @torch.no_grad()
    def infer(self, imgs: np.ndarray) -> tuple[list[dict], dict[str, np.ndarray]]:
        """Records (logits, scores, predicted area) and arrays for one batch of uint8 (N, H, W) images.

        Arrays: ``prob`` (N, H, W) deterministic probability, ``ref`` reference mask
        (0.5 + largest component), and ``samples_<method>`` (T, N, H, W).
        """
        dev, amp = self.device, self.amp
        x = torch.from_numpy(imgs.astype(np.float32) / 255.0)[:, None].to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            p0 = torch.sigmoid(self.seg(x).float())[:, 0]
        p0 = p0.cpu().numpy()
        ref = np.stack([largest_component((p > 0.5).astype(np.uint8)) for p in p0])
        samples = {"det": p0[None]}
        if self.mc_model is not None:
            samples["mc"] = mc_dropout_samples(self.mc_model, x, self.mc_samples, amp).cpu().numpy()
        if "tta" in self.methods:
            samples["tta"] = tta_samples(self.seg, x, amp).cpu().numpy()
        if self.ens:
            outs = []
            for m in self.ens:
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    outs.append(torch.sigmoid(m(x).float())[:, 0])
            samples["ens"] = torch.stack(outs).cpu().numpy()
        logit = {}
        for v, (model, view) in self.classifiers.items():
            xv = np.stack([view(im, mk) for im, mk in zip(imgs, ref)]).astype(np.float32) / 255.0
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logit[v] = model(torch.from_numpy(xv)[:, None].to(dev)).float().cpu().numpy()
        recs: list[dict] = []
        for j in range(len(imgs)):
            r: dict = {"pred_area": int(ref[j].sum())}
            for m in self.methods:
                sc = image_scores(samples[m][:, j], ref[j])
                sc["maxprob_ring"] = -sc["maxprob_ring"]  # higher = less trustworthy, as in policy_eval
                r.update({f"{m}:{k}": sc[k] for k in SCORE_NAMES})
            for v in self.classifiers:
                r.update({f"l_{v}_{c}": float(logit[v][j, c]) for c in range(3)})
            recs.append(r)
        arrays = {"prob": p0, "ref": ref, **{f"samples_{m}": s for m, s in samples.items()}}
        return recs, arrays


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
    models = FoldModels(fold, runs, methods, ensemble_runs, mc_samples=mc_samples)
    out: list[dict] = []
    for i in range(0, len(images), batch):
        recs, arrays = models.infer(images[i : i + batch])
        for j, r in enumerate(recs):
            r = {"idx": i + j, **r}
            if gt_masks is not None:
                from mediscan.metrics.seg import dice

                r["dice_lcc"] = dice(arrays["ref"][j], gt_masks[i + j])
            out.append(r)
    return pd.DataFrame(out)
