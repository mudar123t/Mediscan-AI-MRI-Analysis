"""M1.4 / EXP-009 data: convert BDNeuro-MRI (external test, Hira et al. 2026, CC BY 4.0) and audit overlap.

Source: Mendeley Data DOI 10.17632/zwr4ntf94j.7, file ``Brain_Tumor_MRI_Dataset_Final.zip``
(SHA-256 checked here). Only the three classes shared with Figshare are kept (no_tumor is
dropped). All of its train/val/test folders are used: none of our models saw any of them.

Processing is identical to Figshare (EXP-001): greyscale, per-image 0.5-99.5 percentile
clipping to [0, 1], resize to 256² (area), uint8.

Overlap audit: 64-bit perceptual hash (pHash) of every processed image; for each external
image the minimum Hamming distance to all 3,064 Figshare images is reported.

Usage:
    .venv/Scripts/python scripts/prepare_bdneuro.py --config configs/data_bdneuro.yaml
Writes ``<processed>/images.npy``, ``metadata.csv``, ``report.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from mediscan.data.figshare import CLASS_NAMES, normalize_intensity, to_uint8
from mediscan.utils.repro import environment_info, load_config


def phash(img_u8: np.ndarray) -> np.ndarray:
    """64-bit pHash as a boolean vector."""
    return imagehash.phash(Image.fromarray(img_u8)).hash.flatten()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    zpath = Path(cfg["paths"]["zip"])
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    if digest != cfg["sha256"]:
        raise SystemExit(f"SHA-256 mismatch: {digest}")
    out = Path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    size = int(cfg["image_size"])
    lo, hi = cfg["intensity_percentiles"]
    keep = {c: i for i, c in enumerate(CLASS_NAMES)}

    images, recs = [], []
    with zipfile.ZipFile(zpath) as z:
        for name in sorted(z.namelist()):
            parts = name.strip("/").split("/")
            if len(parts) != 3 or parts[1] not in keep or not parts[2].lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            raw = z.read(name)
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise SystemExit(f"cannot decode {name}")
            channels = 1 if img.ndim == 2 else img.shape[2]
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if channels == 4 else cv2.COLOR_BGR2GRAY)
            h, w = img.shape
            x = normalize_intensity(img, lo, hi)
            if (h, w) != (size, size):
                x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA if max(h, w) > size else cv2.INTER_LINEAR)
            images.append(to_uint8(x))
            recs.append({"file": name, "split": parts[0], "class_name": parts[1], "class_index": keep[parts[1]],
                         "native_h": h, "native_w": w, "channels": channels, "sha1": hashlib.sha1(raw).hexdigest()})
    images = np.stack(images)
    meta = pd.DataFrame(recs)

    # ---- pHash overlap audit vs Figshare ----------------------------------------------------
    fig = np.load(Path(cfg["paths"]["figshare_processed"]) / "images.npy")
    hf = np.stack([phash(im) for im in fig]).astype(np.uint8)
    he = np.stack([phash(im) for im in images]).astype(np.uint8)
    dist = (he[:, None, :] != hf[None, :, :]).sum(2)
    meta["phash_min_dist_figshare"] = dist.min(1)
    meta["phash_nearest_figshare_row"] = dist.argmin(1)
    # within-BDNeuro duplicates (exact bytes)
    meta["dup_sha1_within"] = meta.duplicated("sha1", keep=False)
    thr = int(cfg["phash_near_dup_threshold"])

    # ---- pixel-correlation audit (decisive; pHash is too coarse for MRI slices) --------------
    # Pearson correlation of 64x64 area-downsampled images. Reference: the highest correlation
    # between slices of *different* Figshare patients, which no true duplicate should fall below.
    def emb(a: np.ndarray) -> np.ndarray:
        x = np.stack([cv2.resize(i, (64, 64), interpolation=cv2.INTER_AREA) for i in a]).reshape(len(a), -1).astype(np.float32)
        x -= x.mean(1, keepdims=True)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    ef, ee = emb(fig), emb(images)
    corr = ee @ ef.T
    meta["fig_maxcorr"] = corr.max(1)
    meta["fig_nearest_row"] = corr.argmax(1)
    fmeta = pd.read_csv(Path(cfg["paths"]["figshare_processed"]) / "metadata.csv", dtype={"pid": str})
    cf = ef @ ef.T
    pid = fmeta.pid.to_numpy()
    cf[pid[:, None] == pid[None, :]] = -1.0
    ref_between_patients = cf.max(1)
    t_dup, t_clean = float(cfg["corr_duplicate"]), float(cfg["corr_clean_below"])
    meta["audit"] = np.where(meta.fig_maxcorr >= t_dup, "duplicate", np.where(meta.fig_maxcorr >= t_clean, "ambiguous", "clean"))
    dup = meta.audit == "duplicate"
    nearest = meta.loc[dup, "fig_nearest_row"].to_numpy()
    np.save(out / "images.npy", images)
    meta.to_csv(out / "metadata.csv", index=False)

    report = {
        "source": {"doi": "10.17632/zwr4ntf94j.7", "zip_sha256": digest},
        "n_images": int(len(meta)),
        "per_class": meta.class_name.value_counts().to_dict(),
        "per_split_class": meta.groupby(["split", "class_name"]).size().rename("n").reset_index().to_dict("records"),
        "native_sizes_top5": {f"{h}x{w}": int(n) for (h, w), n in
                              meta.groupby(["native_h", "native_w"]).size().sort_values(ascending=False).head(5).items()},
        "n_distinct_native_sizes": int(meta.groupby(["native_h", "native_w"]).ngroups),
        "channels": meta.channels.value_counts().to_dict(),
        "exact_duplicates_within": int(meta.dup_sha1_within.sum()),
        "phash_min_dist_figshare": {"min": int(dist.min()), "p1": float(np.percentile(dist.min(1), 1)),
                                     "median": float(np.median(dist.min(1)))},
        f"n_near_dup_figshare_dist<={thr}": int((meta.phash_min_dist_figshare <= thr).sum()),
        "corr_audit": {
            "figshare_between_patient_maxcorr": {"median": float(np.median(ref_between_patients)),
                                                 "p99": float(np.percentile(ref_between_patients, 99)),
                                                 "max": float(ref_between_patients.max())},
            "thresholds": {"duplicate_ge": t_dup, "clean_below": t_clean},
            "groups": meta.audit.value_counts().to_dict(),
            "groups_by_split": meta.groupby(["audit", "split"]).size().rename("n").reset_index().to_dict("records"),
            "clean_per_class": meta[meta.audit == "clean"].class_name.value_counts().to_dict(),
            "duplicate_label_agreement": float((meta.loc[dup, "class_index"].to_numpy()
                                                == fmeta.class_index.to_numpy()[nearest]).mean()) if dup.any() else None,
            "duplicate_distinct_figshare_patients": int(len(set(pid[nearest]))),
        },
        "environment": environment_info("."),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "environment"}, indent=1, default=str))


if __name__ == "__main__":
    main()
