"""EXP-006b (ADR-0008): pooled-val tau1 router, plus CIs for the robustness summary over EXP-008.

Specified in ``project-log/experiments/EXP-006b-stable-tau1.md`` before running. Secondary,
post-hoc analysis; the frozen EXP-006/008/009 results stay primary.

(b) router(pooled) = router whose tau1 for test fold k is fitted on the pooled val frames
    of the models j != k-1 (``policy_eval.pooled_val_tau1``). Compared with the frozen
    router and the fixed views in-distribution, under each shift and on the external set.
(d) worst-case and mean (over the 13 EXP-008 conditions) of balanced accuracy and AURC,
    with a paired patient bootstrap that applies the same resample to every condition.

Usage:
    .venv/Scripts/python scripts/eval_exp006b.py --out project-log/experiments/EXP-006b-stable-tau1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mediscan import policy_eval as pe
from mediscan.corrupt import KINDS
from mediscan.metrics import cls as clsm
from mediscan.metrics import selective as sel
from mediscan.stats import paired_stat_bootstrap

POLICIES = ["full+MSP", "crop32+MSP", "masked+MSP", "router", "router(pooled)", "oracle-router"]
ROUTERS = ["router", "router(pooled)"]
BASES = ["crop32+MSP", "full+MSP"]


def policy_arrays(te_frozen: pd.DataFrame, te_pooled: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(prediction, confidence) per policy; both frames hold the same slices in the same order."""
    assert (te_frozen.row.to_numpy() == te_pooled.row.to_numpy()).all() if "row" in te_frozen else True
    pol = pe.policies(te_frozen)
    pol["router(pooled)"] = pe.policies(te_pooled)["router"]
    return {p: pol[p] for p in POLICIES}


def metrics(pol: dict, y: np.ndarray, idx: np.ndarray) -> dict[str, float]:
    out = {}
    for name, (pred, conf) in pol.items():
        out[f"{name}|bal_acc"] = float(clsm.balanced_accuracy_c(clsm.confusion(y[idx], pred[idx])))
        out[f"{name}|aurc"] = sel.aurc(conf[idx], pred[idx] == y[idx])
    return out


def add_diffs(out: dict[str, float], suffix: str = "") -> None:
    for r in ROUTERS:
        for b in BASES + (["router"] if r == "router(pooled)" else []):
            for m in ("bal_acc", "aurc"):
                out[f"{r} - {b}|{m}{suffix}"] = out[f"{r}|{m}{suffix}"] - out[f"{b}|{m}{suffix}"]


def fmt(v: dict) -> str:
    return f"{v['estimate']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp006", default="project-log/experiments/EXP-006-router/results.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    runs = Path("runs")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    score = json.loads(Path(args.exp006).read_text(encoding="utf-8"))["score"]
    meta = pd.read_csv("data/processed/figshare_256/metadata.csv", dtype={"pid": str})
    method = score.split(":")[0]
    frames = {k: pe.load_fold_frames(k, runs, [method]) for k in range(5)}
    vals = {k: frames[k]["val"] for k in range(5)}

    frozen = {k: pe.fit_params(vals[k], score) for k in range(5)}
    pooled = {k: pe.fit_params(vals[k], score, tau1=pe.pooled_val_tau1(vals, k, score)) for k in range(5)}
    taus = {k: {"frozen": frozen[k]["tau1"], "pooled": pooled[k]["tau1"],
                "val_crop_frozen": frozen[k]["frac_crop_val"], "val_crop_pooled": pooled[k]["frac_crop_val"]} for k in range(5)}
    print(f"score {score}")
    for k, v in taus.items():
        print(f"fold {k}: tau1 frozen {v['frozen']:.4f} (val crop {v['val_crop_frozen']:.2f})  "
              f"pooled {v['pooled']:.4f} (val crop {v['val_crop_pooled']:.2f})")

    def both(df_by_fold: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        a = pd.concat([pe.apply_params(frozen[k], d) for k, d in df_by_fold.items()], ignore_index=True)
        b = pd.concat([pe.apply_params(pooled[k], d) for k, d in df_by_fold.items()], ignore_index=True)
        return a, b

    results: dict = {"score": score, "tau1": taus}

    # ---- (b) in-distribution --------------------------------------------------------------
    te_f, te_p = both({k: frames[k]["test"] for k in range(5)})
    pol = policy_arrays(te_f, te_p)
    y = te_f.y.to_numpy()

    def stat_id(idx: np.ndarray) -> dict[str, float]:
        o = metrics(pol, y, idx)
        add_diffs(o)
        return o

    pids = meta.loc[te_f.row, "pid"].to_numpy()
    results["in_distribution"] = paired_stat_bootstrap(pids, stat_id, n_boot=args.n_boot)
    results["in_distribution_crop_share_test"] = {"frozen": float(te_f.used_crop.mean()), "pooled": float(te_p.used_crop.mean())}
    results["in_distribution_conformal_pooled"] = pe.conformal_summary(te_p)
    print("\nIn-distribution (test, 3,064 slices):")
    for k, v in results["in_distribution"].items():
        print(f"  {k:40s} {fmt(v)}")

    # ---- (b)+(d) synthetic shift: per condition and over conditions ------------------------
    conds = ["clean_s0"] + [f"{kd}_s{s}" for kd in KINDS for s in (1, 2, 3)]
    per_cond: dict[str, dict] = {}
    for tag in conds:
        a, b = both({k: pd.read_csv(runs / "exp008_shift" / f"{tag}_fold{k}.csv") for k in range(5)})
        a, b = a.sort_values("row").reset_index(drop=True), b.sort_values("row").reset_index(drop=True)
        per_cond[tag] = {"pol": policy_arrays(a, b), "y": a.y.to_numpy(), "row": a.row.to_numpy(),
                         "crop_share": {"frozen": float(a.used_crop.mean()), "pooled": float(b.used_crop.mean())}}
    rows0 = per_cond[conds[0]]["row"]
    assert all((c["row"] == rows0).all() for c in per_cond.values()), "conditions must cover the same slices"
    pids_s = meta.loc[rows0, "pid"].to_numpy()

    def stat_shift(idx: np.ndarray) -> dict[str, float]:
        o: dict[str, float] = {}
        per = {tag: metrics(c["pol"], c["y"], idx) for tag, c in per_cond.items()}
        for tag, m in per.items():
            for kk, vv in m.items():
                o[f"{kk}@{tag}"] = vv
        for p in POLICIES:
            ba = np.array([per[t][f"{p}|bal_acc"] for t in conds])
            au = np.array([per[t][f"{p}|aurc"] for t in conds])
            o[f"{p}|bal_acc@worst"], o[f"{p}|bal_acc@mean"] = float(ba.min()), float(ba.mean())
            o[f"{p}|aurc@worst"], o[f"{p}|aurc@mean"] = float(au.max()), float(au.mean())
        for sfx in ["@worst", "@mean"] + [f"@{t}" for t in conds]:
            add_diffs(o, sfx)
        return o

    boot = paired_stat_bootstrap(pids_s, stat_shift, n_boot=args.n_boot)
    results["shift_summary"] = {k: v for k, v in boot.items() if k.endswith(("@worst", "@mean"))}
    results["shift_per_condition"] = {t: {k.rsplit("@", 1)[0]: v for k, v in boot.items() if k.endswith(f"@{t}")} for t in conds}
    results["shift_crop_share"] = {t: c["crop_share"] for t, c in per_cond.items()}
    print("\nShift summary over 13 conditions (paired patient bootstrap):")
    for k, v in results["shift_summary"].items():
        print(f"  {k:46s} {fmt(v)}")
    print("\nPer condition, router(pooled) - crop32 / - full / - router (bal. acc):")
    for t in conds:
        d = results["shift_per_condition"][t]
        print(f"  {t:9s} crop {per_cond[t]['crop_share']['pooled']:.2f}  " + "  ".join(
            fmt(d[f"router(pooled) - {b}|bal_acc"]) for b in BASES + ["router"]))

    # ---- (b) external, per fold model set (image-level bootstrap) ---------------------------
    ext = {}
    for k in range(5):
        f = runs / "exp009_external" / f"fold{k}.csv"
        d = pd.read_csv(f)
        a, b = pe.apply_params(frozen[k], d), pe.apply_params(pooled[k], d)
        pol_e, y_e = policy_arrays(a, b), a.y.to_numpy()

        def stat_ext(idx: np.ndarray, pol_e=pol_e, y_e=y_e) -> dict[str, float]:
            o = metrics(pol_e, y_e, idx)
            add_diffs(o)
            return o

        ext[f"fold{k}"] = paired_stat_bootstrap(a.ext_row.to_numpy(), stat_ext, n_boot=args.n_boot)
        ext[f"fold{k}"]["crop_share"] = {"frozen": float(a.used_crop.mean()), "pooled": float(b.used_crop.mean())}
    results["external"] = ext
    print("\nExternal, per fold model set (bal. acc):")
    for k in range(5):
        e = ext[f"fold{k}"]
        print(f"  fold{k} crop share frozen {e['crop_share']['frozen']:.2f} pooled {e['crop_share']['pooled']:.2f}  "
              f"pooled {e['router(pooled)|bal_acc']['estimate']:.3f}  crop32 {e['crop32+MSP|bal_acc']['estimate']:.3f}  "
              f"full {e['full+MSP|bal_acc']['estimate']:.3f}  pooled-crop32 {fmt(e['router(pooled) - crop32+MSP|bal_acc'])}  "
              f"pooled-full {fmt(e['router(pooled) - full+MSP|bal_acc'])}")
    for p in POLICIES:
        v = [ext[f"fold{k}"][f"{p}|bal_acc"]["estimate"] for k in range(5)]
        print(f"  {p:16s} mean {np.mean(v):.4f} range [{np.min(v):.4f}, {np.max(v):.4f}]")

    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
