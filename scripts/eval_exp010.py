"""EXP-010 (R4) analysis: Grad-CAM attribution vs correctness and as a routing signal.

Pre-specified in project-log/experiments/EXP-010-xai-diagnostic.md (analyses A, B, C + 2 figures).

Usage:
    .venv/Scripts/python scripts/eval_exp010.py --out project-log/experiments/EXP-010-xai-diagnostic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from mediscan import policy_eval as pe  # noqa: E402
from mediscan.metrics import cls as clsm  # noqa: E402
from mediscan.metrics import selective as sel  # noqa: E402
from mediscan.stats import paired_stat_bootstrap  # noqa: E402

VIEWS = ("crop32", "full")
CLASS_NAMES = ("meningioma", "glioma", "pituitary")
REGION_COLORS = {"in": "#2a78d6", "ring": "#eda100", "out": "#9a9893"}


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def _fmt(v: dict) -> str:
    return f"{v['estimate']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]"


def load(runs: Path) -> pd.DataFrame:
    """One row per (fold, part, row) with per-view attribution columns ``<view>:<col>``."""
    raw = pd.concat([pd.read_csv(runs / "exp010_xai" / f"fold{k}.csv") for k in range(5)], ignore_index=True)
    raw["susp_gt"] = 1 - (raw.gt_in + raw.gt_ring)
    raw["susp_pred"] = 1 - (raw.pred_in + raw.pred_ring)
    raw["err"] = (raw.pred != raw.y).astype(int)
    raw["msp"] = clsm.softmax(raw[["l_0", "l_1", "l_2"]].to_numpy()).max(1)
    cols = ["pred", "err", "msp", "gt_in", "gt_ring", "gt_out", "pred_in", "pred_ring", "pred_out", "susp_gt", "susp_pred"]
    wide = raw.pivot(index=["fold", "part", "row", "y"], columns="view", values=cols)
    wide.columns = [f"{v}:{c}" for c, v in wide.columns]
    return wide.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    runs, out = Path("runs"), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv("data/processed/figshare_256/metadata.csv", dtype={"pid": str})
    df = load(runs)
    te = df[df.part == "test"].reset_index(drop=True)
    pids = meta.loc[te.row, "pid"].to_numpy()
    res: dict = {"n_test": int(len(te)), "n_patients": int(len(np.unique(pids)))}

    # ---- A: descriptive (GT regions) --------------------------------------------------------
    desc = {}
    for v in VIEWS:
        e = te[f"{v}:err"].to_numpy().astype(bool)
        d = {"n_wrong": int(e.sum())}
        for grp, m in (("correct", ~e), ("wrong", e)):
            d[grp] = {r: float(np.nanmean(te.loc[m, f"{v}:gt_{r}"])) for r in ("in", "ring", "out")}
        d["per_class"] = {CLASS_NAMES[c]: {r: float(np.nanmean(te.loc[te.y == c, f"{v}:gt_{r}"])) for r in ("in", "ring", "out")}
                          for c in range(3)}
        gin = te[f"{v}:gt_in"].to_numpy()

        def stat(idx: np.ndarray, gin=gin, e=e) -> dict[str, float]:
            a, b = gin[idx][~e[idx]], gin[idx][e[idx]]
            return {"in: correct - wrong": float(np.nanmean(a) - np.nanmean(b)) if len(b) else float("nan")}

        d["diff"] = paired_stat_bootstrap(pids, stat, n_boot=args.n_boot)["in: correct - wrong"]
        desc[v] = d
    res["A_descriptive"] = desc

    # ---- B: correctness ----------------------------------------------------------------------
    aurocs = {}
    for v in VIEWS:
        y = te[f"{v}:err"].to_numpy().astype(int)
        S = {f"{v}:susp_gt (oracle)": te[f"{v}:susp_gt"].to_numpy(), f"{v}:susp_pred": te[f"{v}:susp_pred"].to_numpy(),
             f"{v}:MSP": -te[f"{v}:msp"].to_numpy()}
        for k_, s in S.items():
            S[k_] = np.where(np.isnan(s), np.nanmax(s), s)

        def stat(idx: np.ndarray, y=y, S=S) -> dict[str, float]:
            return {k_: _auc(y[idx], s[idx]) for k_, s in S.items()}

        aurocs[f"err_{v}"] = paired_stat_bootstrap(pids, stat, n_boot=args.n_boot)
    res["B_auroc"] = aurocs

    def feats(d: pd.DataFrame, v: str, with_cam: bool) -> np.ndarray:
        x = [-np.log1p(-np.clip(d[f"{v}:msp"].to_numpy(), 0, 1 - 1e-7))]
        if with_cam:
            s = d[f"{v}:susp_pred"].to_numpy()
            x.append(np.where(np.isnan(s), 1.0, s))
        return np.column_stack(x)

    adds = {}
    for v in VIEWS:
        y = te[f"{v}:err"].to_numpy().astype(int)
        preds = {name: np.zeros(len(te)) for name in ("MSP only", "MSP + cam")}
        for k in range(5):
            va = df[(df.part == "val") & (df.fold == k)]
            mk = (te.fold == k).to_numpy()
            for name, wc in (("MSP only", False), ("MSP + cam", True)):
                lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                lr.fit(feats(va, v, wc), va[f"{v}:err"].astype(int).to_numpy())
                preds[name][mk] = lr.predict_proba(feats(te.loc[mk], v, wc))[:, 1]
        fold_of = te.fold.to_numpy()

        def fm(idx: np.ndarray, p: np.ndarray, y=y, fold_of=fold_of) -> float:
            return float(np.nanmean([_auc(y[idx][fold_of[idx] == k], p[idx][fold_of[idx] == k]) for k in range(5)]))

        def stat(idx: np.ndarray, preds=preds, fm=fm) -> dict[str, float]:
            a, b = fm(idx, preds["MSP only"]), fm(idx, preds["MSP + cam"])
            return {"MSP only": a, "MSP + cam": b, "MSP + cam minus MSP": b - a}

        adds[f"err_{v}"] = paired_stat_bootstrap(pids, stat, n_boot=args.n_boot)
    res["B_adds_to_msp"] = adds

    # ---- C: routing ------------------------------------------------------------------------------
    frames = {k: pe.load_fold_frames(k, runs, ["ens"]) for k in range(5)}
    for k in range(5):
        for part in ("val", "test"):
            f = frames[k][part]
            sub = df[(df.fold == k) & (df.part == part)].set_index("row")
            f["cam:full"] = sub.loc[f.row, "full:susp_pred"].to_numpy()
            f["cam:crop"] = sub.loc[f.row, "crop32:susp_pred"].to_numpy()
    val = pd.concat([frames[k]["val"] for k in range(5)], ignore_index=True)
    crop_ok = clsm.softmax(pe.logits(val, "crop32")).argmax(1) == val.y
    full_ok = clsm.softmax(pe.logits(val, "full")).argmax(1) == val.y
    target = (~crop_ok & full_ok).astype(int).to_numpy()
    cands = ["cam:full", "cam:crop", "ens:disagree"]
    val_auc = {c: _auc(target, val[c].fillna(val[c].max()).to_numpy()) for c in cands}
    chosen = max(["cam:full", "cam:crop"], key=val_auc.get)
    res["C_val_auroc_crop_wrong_full_right"] = val_auc
    res["C_chosen_signal"] = chosen
    print(f"C: val AUROC {val_auc}; chosen {chosen}")

    tes = {}
    taus = {}
    for sc in (chosen, "ens:disagree"):
        parts = []
        for k in range(5):
            p = pe.fit_params(frames[k]["val"], sc)
            taus.setdefault(sc, []).append(p["tau1"])
            parts.append(pe.apply_params(p, frames[k]["test"]))
        tes[sc] = pd.concat(parts, ignore_index=True)
    assert (tes[chosen].row.to_numpy() == tes["ens:disagree"].row.to_numpy()).all()
    res["C_tau1"] = taus
    res["C_crop_share_test"] = {sc: float(t.used_crop.mean()) for sc, t in tes.items()}
    pol = pe.policies(tes["ens:disagree"])
    pol = {"full+MSP": pol["full+MSP"], "crop32+MSP": pol["crop32+MSP"], "router(ens)": pol["router"],
           "router(cam)": pe.policies(tes[chosen])["router"], "oracle-router": pol["oracle-router"]}
    yt = tes[chosen].y.to_numpy()
    pids_c = meta.loc[tes[chosen].row, "pid"].to_numpy()

    def stat_c(idx: np.ndarray) -> dict[str, float]:
        o = {}
        for name, (pred, conf) in pol.items():
            o[f"{name}|bal_acc"] = float(clsm.balanced_accuracy_c(clsm.confusion(yt[idx], pred[idx])))
            o[f"{name}|aurc"] = sel.aurc(conf[idx], pred[idx] == yt[idx])
        for b in ("crop32+MSP", "full+MSP", "router(ens)"):
            for m in ("bal_acc", "aurc"):
                o[f"router(cam) - {b}|{m}"] = o[f"router(cam)|{m}"] - o[f"{b}|{m}"]
        return o

    res["C_policies"] = paired_stat_bootstrap(pids_c, stat_c, n_boot=args.n_boot)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    # ---- prints ------------------------------------------------------------------------------
    for v in VIEWS:
        d = desc[v]
        print(f"A {v}: wrong n={d['n_wrong']}; correct in/ring/out "
              + "/".join(f"{d['correct'][r]:.3f}" for r in ("in", "ring", "out")) + "; wrong "
              + "/".join(f"{d['wrong'][r]:.3f}" for r in ("in", "ring", "out")) + f"; in correct-wrong {_fmt(d['diff'])}")
        for c, vv in d["per_class"].items():
            print(f"    {c:10s} in/ring/out " + "/".join(f"{vv[r]:.3f}" for r in ("in", "ring", "out")))
    for t, v in aurocs.items():
        for k_, a in v.items():
            print(f"B AUROC {t:11s} {k_:28s} {a['estimate']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]")
    for t, v in adds.items():
        for k_, a in v.items():
            print(f"B adds {t:11s} {k_:22s} {_fmt(a)}")
    print(f"C tau1 {taus}; crop share {res['C_crop_share_test']}")
    for k_, a in res["C_policies"].items():
        print(f"C {k_:34s} {_fmt(a)}")

    # ---- figure 1: attribution shares --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=200)
    labels, ypos = [], []
    for i, (v, grp) in enumerate([(v, g) for v in VIEWS for g in ("correct", "wrong")]):
        left = 0.0
        for r, col in REGION_COLORS.items():
            w = desc[v][grp][r]
            ax.barh(i, w, left=left, color=col, edgecolor="white", linewidth=1.0, height=0.7,
                    label={"in": "inside tumour", "ring": "16 px border", "out": "elsewhere"}[r] if i == 0 else None)
            if w > 0.06:
                ax.text(left + w / 2, i, f"{w:.2f}", ha="center", va="center", fontsize=7, color="white")
            left += w
        labels.append(f"{'crop 32 px' if v == 'crop32' else 'full image'} · {grp} (n={int((te[f'{v}:err'] == (grp == 'wrong')).sum())})")
        ypos.append(i)
    ax.set_yticks(ypos, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of Grad-CAM mass (ground-truth tumour regions)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(out / "attribution_shares.png", bbox_inches="tight")

    # ---- figure 2: examples (fold 0 test; first 3 correct and first 3 wrong crop-32 predictions) --
    images = np.load("data/processed/figshare_256/images.npy", mmap_mode="r")
    gts = np.load("data/processed/figshare_256/masks.npy", mmap_mode="r")
    cams = np.load(runs / "exp010_xai" / "cams_test_fold0.npz")
    from mediscan.data.datasets import load_predicted_masks

    predm, _ = load_predicted_masks(runs / "exp002c_unet_r34" / "fold0", len(meta))
    t0 = te[(te.fold == 0)].sort_values("row")
    picks = list(t0[t0["crop32:err"] == 0].row[:3]) + list(t0[t0["crop32:err"] == 1].row[:3])
    fig, axes = plt.subplots(len(picks), 3, figsize=(6.3, 2.1 * len(picks)), dpi=160)
    for i, r in enumerate(picks):
        img = cv2.cvtColor(np.asarray(images[r]), cv2.COLOR_GRAY2RGB)
        base = img.copy()
        for m, col in ((np.asarray(gts[r]), (255, 255, 255)), (predm[r], (42, 120, 214))):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(base, cs, -1, col, 2)
        row = te[(te.fold == 0) & (te.row == r)].iloc[0]
        panels = [(base, f"row {r}: {CLASS_NAMES[int(row.y)]}\nGT white, predicted blue")]
        for v in VIEWS:
            c = cv2.resize(cams[f"{v}_{r}"].astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
            c = c / (c.max() + 1e-8)
            a = c[..., None] * 0.75
            ov = (img * (1 - a) + np.array([235, 104, 52]) * a).astype(np.uint8)
            cs, _ = cv2.findContours(np.asarray(gts[r]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cs, -1, (255, 255, 255), 1)
            ok = "correct" if row[f"{v}:err"] == 0 else "WRONG"
            panels.append((ov, f"{'crop 32' if v == 'crop32' else 'full'}: {CLASS_NAMES[int(row[f'{v}:pred'])]} ({ok})\n"
                               f"in {row[f'{v}:gt_in']:.2f} · border {row[f'{v}:gt_ring']:.2f}"))
        for j, (im, title) in enumerate(panels):
            axes[i, j].imshow(im)
            axes[i, j].set_title(title, fontsize=6.5)
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(out / "gradcam_examples.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
