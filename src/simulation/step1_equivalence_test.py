import pandas as pd
from pathlib import Path

from src.replication.msr_replication import (
    build_ets_msr_milp,
    solve_and_print,
    dump_solution_to_df,
)


def run_step1_equivalence_test(
    replication_csv: Path,
    T: int = 53,
    tol_p: float = 1e-6,
    tol_e: float = 1e-6,
    tol_B: float = 1e-6,
    tol_MSR: float = 1e-6,
):
    # --- Solve Espen baseline with full credibility (eta=0 => delta=1/(1+r)) ---
    cpx, idx, params = build_ets_msr_milp(
        T=T,
        use_hotelling_regime=True,
        delta_path=None,  # full credibility default
    )
    solve_and_print(cpx)
    df_espen = dump_solution_to_df(cpx, idx, params=params, T=T)

    # --- Load frozen replication output ---
    df_rep = pd.read_csv(replication_csv)

    # Keep only the comparable years/horizon
    df_rep = df_rep[df_rep["t"].between(0, T)].copy()

    # Merge on t (or year if you prefer)
    m = df_espen.merge(
        df_rep[["t", "year", "p", "e", "B", "MSR", "Intake", "Reinj", "Cancel"]],
        on="t",
        suffixes=("_espen", "_rep"),
        how="inner",
    )

    # Compute max abs deviations
    checks = {
        "p": (m["p_espen"] - m["p_rep"]).abs().max(),
        "e": (m["e_espen"] - m["e_rep"]).abs().max(),
        "B": (m["B_espen"] - m["B_rep"]).abs().max(),
        "MSR": (m["MSR_espen"] - m["MSR_rep"]).abs().max(),
        "Intake": (m["Intake_espen"] - m["Intake_rep"]).abs().max(),
        "Reinj": (m["Reinj_espen"] - m["Reinj_rep"]).abs().max(),
        "Cancel": (m["Cancel_espen"] - m["Cancel_rep"]).abs().max(),
    }

    print("\n=== Step 1 equivalence test (full credibility, eta=0) ===")
    for k, v in checks.items():
        print(f"max |Δ{k}| = {v:.3e}")

    # Hard assertions for the core equilibrium objects
    assert checks["p"] < tol_p, f"Price mismatch too large: {checks['p']}"
    assert checks["e"] < tol_e, f"Emissions mismatch too large: {checks['e']}"
    assert checks["B"] < tol_B, f"Bank mismatch too large: {checks['B']}"
    assert checks["MSR"] < tol_MSR, f"MSR mismatch too large: {checks['MSR']}"

    print("\n✅ PASS: Espen baseline matches replication within tolerance.")
    return df_espen, m, checks


if __name__ == "__main__":
    # Point to your frozen replication output
    # e.g. output/data/msr_replication.csv in the repo
    replication_csv = Path("output/data/msr_replication.csv")

    run_step1_equivalence_test(replication_csv=replication_csv, T=53)