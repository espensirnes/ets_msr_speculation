import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from routeA_solver import solve_routeA, solve_inner
from pathlib import Path
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # code/ → EU_ETS_project
OUT_BASE = PROJECT_ROOT / "output" / "data" / "discarded_specs"

def run_full_cred_benchmark(T: int):
    """
    Full credibility benchmark = constant discount factor 1/(1+r).
    We get r from an initial solve_inner call.
    """
    df0, params0 = solve_inner(T=T, delta_path=None, use_hotelling_regime=True)
    r = float(params0["r"])
    delta_full = [None] + [1.0/(1.0+r)] * T

    df_bench, params_bench = solve_inner(T=T, delta_path=delta_full, use_hotelling_regime=True)
    # clean tiny negative artefacts if any
    for col in ["Intake", "Reinj", "Cancel"]:
        if col in df_bench.columns:
            df_bench[col] = df_bench[col].clip(lower=0.0)

    # add a benchmark "delta_next" for easy plotting/comparison
    df_bench = df_bench.copy()
    df_bench["delta_next"] = np.array(delta_full, dtype=float)
    df_bench["eta"] = np.nan
    df_bench["mode"] = "full_cred"

    return df_bench, params_bench

def summarize_diff(df_a: pd.DataFrame, df_b: pd.DataFrame, cols):
    """
    df_a, df_b must be aligned on t.
    Returns a summary table of max abs diff, mean abs diff for given columns.
    """
    out = []
    for c in cols:
        if c not in df_a.columns or c not in df_b.columns:
            continue
        d = (df_a[c] - df_b[c]).to_numpy()
        out.append({
            "col": c,
            "max_abs_diff": float(np.max(np.abs(d))),
            "mean_abs_diff": float(np.mean(np.abs(d))),
        })
    return pd.DataFrame(out).sort_values("max_abs_diff", ascending=False)

def compare_policy_actions(df_a: pd.DataFrame, df_b: pd.DataFrame, eps=1e-9):
    """
    Compare whether (Intake/Reinj/Cancel) are 'on' in each year.
    """
    res = {}
    for col in ["Intake", "Reinj", "Cancel"]:
        if col not in df_a.columns or col not in df_b.columns:
            continue
        on_a = (df_a[col].to_numpy() > eps)
        on_b = (df_b[col].to_numpy() > eps)
        diff_idx = np.where(on_a != on_b)[0]
        res[col] = diff_idx.tolist()
    return res
    
def save_bundle(out_dir: Path, eta: float, df_eta: pd.DataFrame, hist_eta: pd.DataFrame,
                diff_table: pd.DataFrame, policy_diff: dict):
    """
    Save all outputs for a single eta run.
    """
    # safe filename token
    eta_tag = f"{eta:.6g}".replace(".", "p").replace("-", "m")

    df_eta.to_csv(out_dir / f"routeA_abatement_eta_{eta_tag}.csv", index=False)
    hist_eta.to_csv(out_dir / f"routeA_abatement_eta_{eta_tag}_history.csv", index=False)
    diff_table.to_csv(out_dir / f"diff_vs_fullcred_eta_{eta_tag}.csv", index=False)

    with open(out_dir / f"policy_onoff_eta_{eta_tag}.json", "w") as f:
        json.dump(policy_diff, f, indent=2)

def run_and_compare(T: int, etas=(1e-4, 1e-3, 2e-3), omega=0.2, tol=1e-6, max_iter=200, out_dir: Path | None = None):
    # 1) benchmark
    df_bench, params_bench = run_full_cred_benchmark(T)
    t0_year = int(params_bench.get("t0_year", 2017))

    # ---- output folder ----
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df_bench.to_csv(out_dir / "full_cred_benchmark.csv", index=False)

        meta = {
            "T": T,
            "etas": list(map(float, etas)),
            "omega": float(omega),
            "tol": float(tol),
            "max_iter": int(max_iter),
            "t0_year": int(t0_year),
        }
        with open(out_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)     
    

    # 2) routeA runs
    results = []
    summaries = {}

    key_cols = ["p", "e", "B", "TNAC", "S", "MSR", "Intake", "Reinj", "Cancel", "delta_next", "Z"]

    for eta in etas:
        df_eta, hist_eta = solve_routeA(
            T=T,
            eta=float(eta),
            omega=omega,
            tol=tol,
            max_iter=max_iter,
            Z_mode="abatement",
            out_csv=None,
            verbose=False,   # you already saw it converge; keep output quiet here
        )

        # align by t (just in case)
        a = df_eta.sort_values("t").reset_index(drop=True)
        b = df_bench.sort_values("t").reset_index(drop=True)

        # add identifiers
        a = a.copy()
        a["eta"] = float(eta)
        a["mode"] = "abatement"

        # diffs summary
        summ = summarize_diff(a, b, key_cols)
        policy_diff = compare_policy_actions(a, b, eps=1e-9)
        if out_dir is not None:
            save_bundle(out_dir, float(eta), a, hist_eta, summ, policy_diff)        
        

        summaries[eta] = {"diff_table": summ, "policy_action_diffs": policy_diff, "hist": hist_eta}
        results.append(a)

        print(f"\n=== eta={eta} vs full_cred ===")
        print(summ.head(12).to_string(index=False))
        print("Policy on/off differences (t indices):", policy_diff)

        # a couple of quick “where is it largest?” diagnostics
        if "e" in a.columns and "e" in b.columns:
            de = (a["e"] - b["e"]).to_numpy()
            j = int(np.argmax(np.abs(de)))
            print(f"Max |Δe| at t={j} (year={t0_year + j}): Δe={de[j]:.6g}")

        if "delta_next" in a.columns and "delta_next" in b.columns:
            dd = (a["delta_next"] - b["delta_next"]).to_numpy()
            j = int(np.argmax(np.abs(dd[1:]))) + 1  # skip t=0 nan
            print(f"Max |Δdelta| at t={j} (year={t0_year + j}): Δdelta={dd[j]:.6g}")

    df_all = pd.concat([df_bench] + results, ignore_index=True)
    if out_dir is not None:
        df_all.to_csv(out_dir / "panel_all_modes.csv", index=False)    
    return df_bench, df_all, summaries


# ---- RUN IT ----
T = 22
etas = [1e-4, 1e-3, 2e-3, 1e-1, 2e-1, 3e-1]

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = OUT_BASE / f"Z_abatement_{run_id}"
out_dir.mkdir(parents=True, exist_ok=True)

df_bench, df_all, summaries = run_and_compare(
    T=T,
    etas=etas,
    omega=0.2,
    tol=1e-6,
    max_iter=200,
    out_dir=out_dir,
)

print(f"\nSaved outputs to: {out_dir.resolve()}")

# ---- OPTIONAL PLOTS ----
# 1) prices over time: benchmark vs each eta
plt.figure()
for eta in etas:
    d = df_all[(df_all["mode"]=="abatement") & (df_all["eta"]==eta)].sort_values("t")
    plt.plot(d["t"], d["p"], label=f"abatement eta={eta}")
b = df_all[df_all["mode"]=="full_cred"].sort_values("t")
plt.plot(b["t"], b["p"], label="full_cred", linewidth=2)
plt.xlabel("t")
plt.ylabel("p")
plt.legend()
plt.title("Price path: abatement RouteA vs full credibility")


plt.savefig(out_dir / "plot_price_path.png", dpi=200, bbox_inches="tight")
#plt.show()   # or comment out if you don’t want interactive windows
plt.close()

# 2) delta_next over time: benchmark vs each eta
plt.figure()
for eta in etas:
    d = df_all[(df_all["mode"]=="abatement") & (df_all["eta"]==eta)].sort_values("t")
    plt.plot(d["t"], d["delta_next"], label=f"abatement eta={eta}")
plt.plot(b["t"], b["delta_next"], label="full_cred", linewidth=2)
plt.xlabel("t")
plt.ylabel("delta_next")
plt.legend()
plt.title("Discount factor: endogenous vs full credibility")

plt.savefig(out_dir / "plot_discount_factor_path.png", dpi=200, bbox_inches="tight")
#plt.show()   # or comment out if you don’t want interactive windows
plt.close()


# 3) emissions differences (abatement - benchmark)
plt.figure()
for eta in etas:
    d = df_all[(df_all["mode"]=="abatement") & (df_all["eta"]==eta)].sort_values("t")
    de = d["e"].to_numpy() - b["e"].to_numpy()
    plt.plot(d["t"], de, label=f"eta={eta}")
plt.xlabel("t")
plt.ylabel("Δe (abatement - full_cred)")
plt.legend()
plt.title("Emissions difference vs full credibility")

plt.savefig(out_dir / "plot_emissions_difference_vs_full_cred.png", dpi=200, bbox_inches="tight")
#plt.show()   # or comment out if you don’t want interactive windows
plt.close()
