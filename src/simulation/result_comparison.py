import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from routeA_solver import solve_routeA, solve_inner
from pathlib import Path
import json
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/ -> EU_ETS_project
OUT_BASE_DATA = PROJECT_ROOT / "output" / "data" / "simulation"
OUT_BASE_PLOT = PROJECT_ROOT / "output" / "figures" / "simulation"


def eta_tag(x: float) -> str:
    return f"{x:.6g}".replace(".", "p").replace("-", "m")


def run_full_cred_benchmark(T: int):
    """
    Full credibility benchmark = constant discount factor 1/(1+r).
    We get r from an initial solve_inner call.
    """
    df0, params0 = solve_inner(T=T, delta_path=None, use_hotelling_regime=True)
    r = float(params0["r"])
    delta_full = [None] + [1.0 / (1.0 + r)] * T

    df_bench, params_bench = solve_inner(T=T, delta_path=delta_full, use_hotelling_regime=True)

    # clean tiny negative artefacts if any
    for col in ["Intake", "Reinj", "Cancel"]:
        if col in df_bench.columns:
            df_bench[col] = df_bench[col].clip(lower=0.0)

    df_bench = df_bench.copy()
    df_bench["delta_next"] = np.array(delta_full, dtype=float)
    df_bench["eta"] = np.nan
    df_bench["mode"] = "full_cred"

    return df_bench, params_bench


def summarize_diff(df_a: pd.DataFrame, df_b: pd.DataFrame, cols):
    out = []
    for c in cols:
        if c not in df_a.columns or c not in df_b.columns:
            continue
        d = (df_a[c] - df_b[c]).to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        out.append(
            {"col": c, "max_abs_diff": float(np.max(np.abs(d))), "mean_abs_diff": float(np.mean(np.abs(d)))}
        )
    if not out:
        return pd.DataFrame(columns=["col", "max_abs_diff", "mean_abs_diff"])
    return pd.DataFrame(out).sort_values("max_abs_diff", ascending=False)


def compare_policy_actions(df_a: pd.DataFrame, df_b: pd.DataFrame, eps=1e-9):
    """
    Compare whether (Intake/Reinj/Cancel) are 'on' in each year.
    Returns dict: col -> list of t indices where on/off differs.
    """
    res = {}
    for col in ["Intake", "Reinj", "Cancel"]:
        if col not in df_a.columns or col not in df_b.columns:
            continue
        on_a = df_a[col].to_numpy() > eps
        on_b = df_b[col].to_numpy() > eps
        diff_idx = np.where(on_a != on_b)[0]
        res[col] = diff_idx.tolist()
    return res


def run_and_compare(
    T: int,
    etas=(1e-4, 1e-3, 2e-3),
    Z_modes=("abatement",),
    omega=0.2,
    tol=1e-6,
    max_iter=200,
    verbose_routeA=False,
    out_dir: Path | None = None,
):
    """
    Compare RouteA outcomes vs full-cred benchmark for:
      - multiple etas
      - multiple Z_modes

    Returns:
      df_bench, df_all, summaries
    where summaries[(Z_mode, eta)] contains diff_table, policy_action_diffs, hist_df.
    """
    if out_dir is None:
        raise ValueError("run_and_compare requires out_dir (Path).")

    # 1) benchmark
    df_bench, params_bench = run_full_cred_benchmark(T)
    t0_year = int(params_bench.get("t0_year", 2017))

    # save benchmark
    df_bench.to_csv(out_dir / "benchmark_full_cred.csv", index=False)

    # 2) routeA runs
    results = []
    summaries = {}

    key_cols = [
        "p",
        "e",
        "B",
        "TNAC",
        "S",
        "MSR",
        "Intake",
        "Reinj",
        "Cancel",
        "delta_next",
        "Z",  # may or may not exist
    ]

    for Z_mode in Z_modes:
        for eta in etas:
            tag = eta_tag(float(eta))

            out_csv_eta = out_dir / f"routeA_{Z_mode}_eta_{tag}.csv"
            out_hist_eta = out_dir / f"routeA_{Z_mode}_eta_{tag}_history.csv"

            df_routeA, hist_routeA = solve_routeA(
                T=T,
                eta=eta,
                omega=omega,
                tol=tol,
                max_iter=max_iter,
                Z_mode=Z_mode,
                out_csv=out_csv_eta,
                verbose=verbose_routeA,
            )

            # also save the returned hist (solve_routeA should already save it, but keep this robust)
            if hist_routeA is not None:
                hist_routeA.to_csv(out_hist_eta, index=False)

            if df_routeA is None or len(df_routeA) == 0:
                print(f"\n=== Z_mode={Z_mode} | eta={eta} ===")
                print("WARNING: solve_routeA returned empty df. Skipping.")
                continue

            # align by t
            a = df_routeA.sort_values("t").reset_index(drop=True).copy()
            b = df_bench.sort_values("t").reset_index(drop=True)

            a["eta"] = float(eta)
            a["mode"] = str(Z_mode)

            summ = summarize_diff(a, b, key_cols)
            policy_diff = compare_policy_actions(a, b, eps=1e-9)

            # save per-run summary tables
            summ.to_csv(out_dir / f"diff_vs_fullcred_{Z_mode}_eta_{tag}.csv", index=False)
            with open(out_dir / f"policy_onoff_{Z_mode}_eta_{tag}.json", "w") as f:
                json.dump(policy_diff, f, indent=2)

            summaries[(str(Z_mode), float(eta))] = {
                "diff_table": summ,
                "policy_action_diffs": policy_diff,
                "hist_df": hist_routeA,
            }
            results.append(a)

            # console print
            print(f"\n=== Z_mode={Z_mode} | eta={eta} vs full_cred ===")
            if len(summ) > 0:
                print(summ.head(12).to_string(index=False))
            else:
                print("(No comparable columns found for diff table.)")
            print("Policy on/off differences (t indices):", policy_diff)

            # quick “where is it largest?”
            if ("e" in a.columns) and ("e" in b.columns):
                de = (a["e"] - b["e"]).to_numpy()
                j = int(np.argmax(np.abs(de)))
                print(f"Max |Δe| at t={j} (year={t0_year + j}): Δe={de[j]:.6g}")

            if ("delta_next" in a.columns) and ("delta_next" in b.columns):
                dd = (a["delta_next"] - b["delta_next"]).to_numpy()
                if len(dd) > 1:
                    j = int(np.argmax(np.abs(dd[1:]))) + 1
                else:
                    j = 0
                print(f"Max |Δdelta| at t={j} (year={t0_year + j}): Δdelta={dd[j]:.6g}")

    df_all = pd.concat([df_bench] + results, ignore_index=True) if len(results) else df_bench.copy()
    df_all.to_csv(out_dir / "all_runs.csv", index=False)

    return df_bench, df_all, summaries


# ---- RUN IT ----
T = 22
etas = [1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 3e-1]
Z_modes = ["supply_over_Frem"]

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_slug = f"Z_{'_'.join(Z_modes)}_{run_id}"

out_dir = OUT_BASE_DATA / run_slug
out_dir.mkdir(parents=True, exist_ok=True)

out_dir_plots = OUT_BASE_PLOT / run_slug
out_dir_plots.mkdir(parents=True, exist_ok=True)

df_bench, df_all, summaries = run_and_compare(
    T=T,
    etas=etas,
    Z_modes=Z_modes,
    omega=0.2,
    tol=1e-6,
    max_iter=200,
    verbose_routeA=False,
    out_dir=out_dir,
)

# ---- PLOTS ----
b = df_all[df_all["mode"] == "full_cred"].sort_values("t")

# 1) prices
plt.figure()
for Z_mode in Z_modes:
    for eta in etas:
        d = df_all[(df_all["mode"] == Z_mode) & (df_all["eta"] == eta)].sort_values("t")
        if len(d) == 0:
            continue
        plt.plot(d["t"], d["p"], label=f"{Z_mode} eta={eta}")
plt.plot(b["t"], b["p"], label="full_cred", linewidth=2)
plt.xlabel("t")
plt.ylabel("p")
plt.legend()
plt.title("Price path: RouteA vs full credibility")
plt.savefig(out_dir_plots / "Price_path_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# 2) delta_next
plt.figure()
for Z_mode in Z_modes:
    for eta in etas:
        d = df_all[(df_all["mode"] == Z_mode) & (df_all["eta"] == eta)].sort_values("t")
        if len(d) == 0:
            continue
        plt.plot(d["t"], d["delta_next"], label=f"{Z_mode} eta={eta}")
plt.plot(b["t"], b["delta_next"], label="full_cred", linewidth=2)
plt.xlabel("t")
plt.ylabel("delta_next")
plt.legend()
plt.title("Discount factor: endogenous vs full credibility")
plt.savefig(out_dir_plots / "Discount_factor_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# 3) emissions differences (RouteA - benchmark)
plt.figure()
for Z_mode in Z_modes:
    for eta in etas:
        d = df_all[(df_all["mode"] == Z_mode) & (df_all["eta"] == eta)].sort_values("t")
        if len(d) == 0:
            continue
        # align by t using benchmark index
        de = d["e"].to_numpy() - b["e"].to_numpy()
        plt.plot(d["t"], de, label=f"{Z_mode} eta={eta}")
plt.axhline(0.0, linewidth=1)
plt.xlabel("t")
plt.ylabel("Δe (RouteA - full_cred)")
plt.legend()
plt.title("Emissions difference vs full credibility")
plt.savefig(out_dir_plots / "Emissions_diff_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"\nSaved data to: {out_dir}")
print(f"Saved figures to: {out_dir_plots}")