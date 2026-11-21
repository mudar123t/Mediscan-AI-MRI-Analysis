"""EXP-011 (R2) analysis, pre-specified in project-log/experiments/EXP-011-medsam-second-opinion.md.

A descriptive MedSAM Dice; B segmentation-failure AUROC; C signal choice on pooled val + in-distribution
router; D shift (13 conditions, worst/mean with a shared patient resample) and external (per fold model).

Usage:
    .venv/Scripts/python scripts/eval_exp011.py --out project-log/experiments/EXP-011-medsam-second-opinion [--parts ABC D]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mediscan import policy_eval as pe
from mediscan.corrupt import KINDS
from mediscan.medsam import SIGNALS
from mediscan.metrics import cls as clsm
from mediscan.metrics import selective as sel
from mediscan.stats import paired_stat_bootstrap

RUNS = Path("runs")
SAM = RUNS / "exp011_medsam"
ENS = "ens:disagree"
POL = ["full+MSP", "crop32+MSP", "router(ens)", "router(sam)", "oracle-router"]
CLASS_NAMES = ("meningioma", "glioma", "pituitary")


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def _fill(s: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(s), np.nanmax(s), s)


def _fmt(v: dict) -> str:
    return f"{v['estimate']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]"


def id_frames() -> dict[int, dict[str, pd.DataFrame]]:
    frames = {k: pe.load_fold_frames(k, RUNS, ["ens"]) for k in range(5)}
    for k in range(5):
        s = pd.read_csv(SAM / f"clean_fold{k}.csv")
        for part in ("val", "test"):
            sp = s[s.part == part].drop(columns=["fold", "part"])
            frames[k][part] = frames[k][part].merge(sp, on="row", how="left")
    return frames


def policy_arrays(te_ens: pd.DataFrame, te_sam: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    key = "row" if "row" in te_ens else "ext_row"
    assert (te_ens[key].to_numpy() == te_sam[key].to_numpy()).all()
    a, b = pe.policies(te_ens), pe.policies(te_sam)
    return {"full+MSP": a["full+MSP"], "crop32+MSP": a["crop32+MSP"], "router(ens)": a["router"],
            "router(sam)": b["router"], "oracle-router": a["oracle-router"]}


def metrics(pol: dict, y: np.ndarray, idx: np.ndarray, sfx: str = "") -> dict[str, float]:
    o = {}
    for name, (pred, conf) in pol.items():
        o[f"{name}|bal_acc{sfx}"] = float(clsm.balanced_accuracy_c(clsm.confusion(y[idx], pred[idx])))
        o[f"{name}|aurc{sfx}"] = sel.aurc(conf[idx], pred[idx] == y[idx])
    for b in ("crop32+MSP", "full+MSP", "router(ens)"):
        for m in ("bal_acc", "aurc"):
            o[f"router(sam) - {b}|{m}{sfx}"] = o[f"router(sam)|{m}{sfx}"] - o[f"{b}|{m}{sfx}"]
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", nargs="*", default=["ABC", "D"])
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rpath = out / "results.json"
    res: dict = json.loads(rpath.read_text(encoding="utf-8")) if rpath.exists() else {}
    meta = pd.read_csv("data/processed/figshare_256/metadata.csv", dtype={"pid": str})
    frames = id_frames()

    if "ABC" in args.parts:
        te = pd.concat([frames[k]["test"] for k in range(5)], ignore_index=True)
        pids = meta.loc[te.row, "pid"].to_numpy()
        # A: MedSAM (U-Net box) vs U-Net Dice
        md, ud, y = te.medsam_dice_gt.to_numpy(), te.unet_dice_gt.to_numpy(), te.y.to_numpy()

        def stat_a(idx: np.ndarray) -> dict[str, float]:
            o = {"medsam": float(np.nanmean(md[idx])), "unet": float(np.nanmean(ud[idx]))}
            o["medsam - unet"] = o["medsam"] - o["unet"]
            for c, n in enumerate(CLASS_NAMES):
                m = y[idx] == c
                o[f"medsam|{n}"], o[f"unet|{n}"] = float(np.nanmean(md[idx][m])), float(np.nanmean(ud[idx][m]))
            return o

        res["A_dice"] = paired_stat_bootstrap(pids, stat_a, n_boot=args.n_boot)
        res["A_empty_unet_masks_test"] = int(np.isnan(md).sum())
        # B: seg-failure detection
        fail = (ud < 0.7).astype(int)
        S = {s: _fill(te[s].to_numpy()) for s in (*SIGNALS, ENS)}

        def stat_b(idx: np.ndarray) -> dict[str, float]:
            o = {s: _auc(fail[idx], v[idx]) for s, v in S.items()}
            for s in SIGNALS:
                o[f"{s} - {ENS}"] = o[s] - o[ENS]
            return o

        res["B_seg_fail_auroc"] = paired_stat_bootstrap(pids, stat_b, n_boot=args.n_boot)
        res["B_positives"] = int(fail.sum())
        # C: choose on pooled val, router in-distribution
        val = pd.concat([frames[k]["val"] for k in range(5)], ignore_index=True)
        crop_ok = clsm.softmax(pe.logits(val, "crop32")).argmax(1) == val.y
        full_ok = clsm.softmax(pe.logits(val, "full")).argmax(1) == val.y
        tgt = (~crop_ok & full_ok).astype(int).to_numpy()
        val_auc = {s: _auc(tgt, _fill(val[s].to_numpy())) for s in (*SIGNALS, ENS)}
        chosen = max(SIGNALS, key=val_auc.get)
        res["C_val_auroc_crop_wrong_full_right"], res["C_chosen"] = val_auc, chosen
        params = {sc: {k: pe.fit_params(frames[k]["val"], sc) for k in range(5)} for sc in (chosen, ENS)}
        res["C_tau1"] = {sc: [params[sc][k]["tau1"] for k in range(5)] for sc in params}
        t_ens = pd.concat([pe.apply_params(params[ENS][k], frames[k]["test"]) for k in range(5)], ignore_index=True)
        t_sam = pd.concat([pe.apply_params(params[chosen][k], frames[k]["test"]) for k in range(5)], ignore_index=True)
        pol, yt = policy_arrays(t_ens, t_sam), t_ens.y.to_numpy()
        res["C_crop_share_test"] = {"ens": float(t_ens.used_crop.mean()), "sam": float(t_sam.used_crop.mean())}
        res["C_policies"] = paired_stat_bootstrap(meta.loc[t_ens.row, "pid"].to_numpy(),
                                                  lambda idx: metrics(pol, yt, idx), n_boot=args.n_boot)
        rpath.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        print("A", {k: round(v["estimate"], 3) for k, v in res["A_dice"].items()}, "| diff", _fmt(res["A_dice"]["medsam - unet"]))
        print("B positives", res["B_positives"], {k: _fmt(v) for k, v in res["B_seg_fail_auroc"].items()})
        print("C val AUROC", {k: round(v, 3) for k, v in val_auc.items()}, "chosen", chosen, "tau1", res["C_tau1"])
        for k, v in res["C_policies"].items():
            print(f"C {k:34s} {_fmt(v)}")

    if "D" in args.parts:
        chosen = res["C_chosen"]
        params = {sc: {k: pe.fit_params(frames[k]["val"], sc) for k in range(5)} for sc in (chosen, ENS)}
        conds = ["clean_s0"] + [f"{kd}_s{s}" for kd in KINDS for s in (1, 2, 3)]
        per = {}
        for tag in conds:
            parts_e, parts_s = [], []
            for k in range(5):
                f = pd.read_csv(RUNS / "exp008_shift" / f"{tag}_fold{k}.csv")
                if tag == "clean_s0":
                    s = pd.read_csv(SAM / f"clean_fold{k}.csv")
                    s = s[s.part == "test"]
                else:
                    s = pd.read_csv(SAM / f"shift_{tag}_fold{k}.csv")
                f = f.merge(s[["row", *SIGNALS]], on="row", how="left")
                parts_e.append(pe.apply_params(params[ENS][k], f))
                parts_s.append(pe.apply_params(params[chosen][k], f))
            a = pd.concat(parts_e, ignore_index=True).sort_values("row").reset_index(drop=True)
            b = pd.concat(parts_s, ignore_index=True).sort_values("row").reset_index(drop=True)
            per[tag] = {"pol": policy_arrays(a, b), "y": a.y.to_numpy(), "row": a.row.to_numpy(),
                        "crop_share": {"ens": float(a.used_crop.mean()), "sam": float(b.used_crop.mean())}}
        rows0 = per["clean_s0"]["row"]
        assert all((c["row"] == rows0).all() for c in per.values())

        def stat_d(idx: np.ndarray) -> dict[str, float]:
            o: dict[str, float] = {}
            m = {t: metrics(c["pol"], c["y"], idx) for t, c in per.items()}
            for t, mt in m.items():
                o.update({f"{kk}@{t}": vv for kk, vv in mt.items()})
            for p in POL:
                ba = np.array([m[t][f"{p}|bal_acc"] for t in conds])
                au = np.array([m[t][f"{p}|aurc"] for t in conds])
                o[f"{p}|bal_acc@worst"], o[f"{p}|bal_acc@mean"] = float(ba.min()), float(ba.mean())
                o[f"{p}|aurc@worst"], o[f"{p}|aurc@mean"] = float(au.max()), float(au.mean())
            for sfx in ("@worst", "@mean"):
                for b in ("crop32+MSP", "full+MSP", "router(ens)"):
                    for mm in ("bal_acc", "aurc"):
                        o[f"router(sam) - {b}|{mm}{sfx}"] = o[f"router(sam)|{mm}{sfx}"] - o[f"{b}|{mm}{sfx}"]
            return o

        boot = paired_stat_bootstrap(meta.loc[rows0, "pid"].to_numpy(), stat_d, n_boot=args.n_boot)
        res["D_shift_summary"] = {k: v for k, v in boot.items() if k.endswith(("@worst", "@mean"))}
        res["D_shift_per_condition"] = {t: {k.rsplit("@", 1)[0]: v for k, v in boot.items() if k.endswith(f"@{t}")} for t in conds}
        res["D_shift_crop_share"] = {t: c["crop_share"] for t, c in per.items()}
        ext = {}
        for k in range(5):
            f = pd.read_csv(RUNS / "exp009_external" / f"fold{k}.csv").merge(
                pd.read_csv(SAM / f"external_fold{k}.csv")[["ext_row", *SIGNALS]], on="ext_row", how="left")
            a, b = pe.apply_params(params[ENS][k], f), pe.apply_params(params[chosen][k], f)
            pol_e, ye = policy_arrays(a, b), a.y.to_numpy()
            ext[f"fold{k}"] = paired_stat_bootstrap(a.ext_row.to_numpy(), lambda idx, p=pol_e, y=ye: metrics(p, y, idx),
                                                    n_boot=args.n_boot)
            ext[f"fold{k}"]["crop_share"] = {"ens": float(a.used_crop.mean()), "sam": float(b.used_crop.mean())}
        res["D_external"] = ext
        rpath.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        print("D shift summary:")
        for k, v in res["D_shift_summary"].items():
            print(f"  {k:40s} {_fmt(v)}")
        print("D per condition, router(sam) - crop32 / - full / - router(ens) (bal. acc):")
        for t in conds:
            d = res["D_shift_per_condition"][t]
            print(f"  {t:9s} crop {per[t]['crop_share']['sam']:.2f}  " + "  ".join(
                _fmt(d[f"router(sam) - {b}|bal_acc"]) for b in ("crop32+MSP", "full+MSP", "router(ens)")))
        print("D external per fold (bal. acc router(sam) | - crop32 | - router(ens)):")
        for k in range(5):
            e = ext[f"fold{k}"]
            print(f"  fold{k} crop {e['crop_share']['sam']:.2f}  {e['router(sam)|bal_acc']['estimate']:.3f}  "
                  f"{_fmt(e['router(sam) - crop32+MSP|bal_acc'])}  {_fmt(e['router(sam) - router(ens)|bal_acc'])}")


if __name__ == "__main__":
    main()
