import numpy as np
import pandas as pd
from pathlib import Path

from ets_msr_inner import (
    build_ets_msr_inner,
    solve_and_print,
    dump_solution_to_df,
)

def compute_Z_supply_over_Frem(df_sol, T: int, F_rem: np.ndarray):
    """
    Unsmoothed scarcity signal based on market supply relative to
    remaining-horizon exogenous issuance.

    Z_t = max(0, 1 - S_mkt_t / F_rem_t)

    F_rem[t] must be > 0 for all t < T.
    """
    Z = np.zeros(T + 1)

    S_mkt = df_sol["Supply_to_market"].to_numpy()
    t_vals = df_sol["t"].to_numpy().astype(int)

    for t, s in zip(t_vals, S_mkt):
        if 0 <= t <= T:
            denom = F_rem[t]
            if denom > 0:
                Z[t] = max(0.0, 1.0 - s / denom)
            else:
                Z[t] = 0.0

    return Z
    
def compute_Frem_from_S_issue(df, T: int) -> np.ndarray:
    """
    Remaining-horizon exogenous issuance (TOTAL, not average).

    Frem[t] = sum_{j=t..T} S_issue[j]   (inclusive)

    Returns array length T+1 with indices 0..T.
    Sets Frem[0] = Frem[1] for convenience (t=0 is pre-decision period in our setup).
    """
    import numpy as np

    Frem = np.zeros(T + 1, dtype=float)

    # Map issuance by t (robust to ordering)
    t_vals = df["t"].to_numpy().astype(int)
    S_issue = df["S_issue"].to_numpy().astype(float)

    S_by_t = np.zeros(T + 1, dtype=float)
    for t, s in zip(t_vals, S_issue):
        if 0 <= t <= T:
            S_by_t[t] = s

    # Reverse cumulative sum to get remaining totals
    Frem = np.cumsum(S_by_t[::-1])[::-1]

    # match your convention
    if T >= 1:
        Frem[0] = Frem[1]

    return Frem



def compute_F_remaining_from_S_issue(df, T: int) -> np.ndarray:
    """
    F_rem[t] = average remaining-horizon exogenous issuance from t..T (inclusive),
    built from df['S_issue'] (assumed exogenous).
    Returns array length T+1 with indices 0..T. Sets F_rem[0]=F_rem[1].
    """
    d = df.sort_values("t")
    s_issue = d["S_issue"].to_numpy(dtype=float)
    F_rem = np.zeros(T + 1, dtype=float)

    for t in range(1, T + 1):
        F_rem[t] = float(np.mean(s_issue[t:T+1]))

    F_rem[0] = F_rem[1]
    return F_rem


def compute_Z_tnac_over_Frem(df, T: int, F_rem: np.ndarray) -> np.ndarray:
    """
    Z[t] = max(0, 1 - TNAC[t]/F_rem[t]), with F_rem forward-looking (remaining-horizon issuance average).
    Returns array length T+1 indexed by t.
    """
    d = df.sort_values("t")
    tnac = d["TNAC"].to_numpy(dtype=float)

    Z = np.zeros(T + 1, dtype=float)
    eps = 1e-12
    for t in range(0, T + 1):
        denom = max(float(F_rem[t]), eps)
        Z[t] = max(0.0, 1.0 - float(tnac[t]) / denom)
    return Z


def compute_Z(df: pd.DataFrame) -> pd.Series:
    """Z_t = max(0, Intake_t - Reinj_t + Cancel_t)."""
    z = df["Intake"] - df["Reinj"] + df["Cancel"]
    return z.clip(lower=0.0)
    
    
def compute_Z_abatement(df_sol, u: float):
    """
    Normalized abatement effort with constant BAU emissions u:
        Z_t = (u - e_t) / u
    """
    e = df_sol["e"].to_numpy()
    Z = (u - e) / u
    return pd.Series(Z, index=df_sol.index)

def compute_Fbar_from_S_issue(df: pd.DataFrame, T: int) -> float:
    if "S_issue" not in df.columns:
        raise KeyError("Missing column 'S_issue' needed for exogenous Fbar.")

    d = df.loc[df["t"].between(1, T), "S_issue"].astype(float).to_numpy()
    if d.size == 0:
        raise ValueError("No rows for t=1..T when computing Fbar.")

    Fbar = float(np.mean(d))
    if not np.isfinite(Fbar) or Fbar <= 0.0:
        raise ValueError(f"Invalid Fbar={Fbar} computed from S_issue.")
    return Fbar
    


def compute_Z_policy_flow_Fbar(df: pd.DataFrame, T: int, Fbar: float) -> np.ndarray:
    for c in ["t", "Intake", "Reinj", "Cancel"]:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' needed for Z.")

    Z = np.zeros(T + 1, dtype=float)

    sub = df.loc[df["t"].between(1, T), ["t", "Intake", "Reinj", "Cancel"]].copy()
    sub["t"] = sub["t"].astype(int)

    intake = sub["Intake"].to_numpy(dtype=float)
    reinj  = sub["Reinj"].to_numpy(dtype=float)
    cancel = sub["Cancel"].to_numpy(dtype=float)

    zvals = (intake - reinj + cancel) / Fbar

    for tt, zz in zip(sub["t"].to_numpy(), zvals):
        Z[int(tt)] = float(zz)

    return Z   
     

def update_delta_from_Z(Z: np.ndarray, r: float, eta: float) -> np.ndarray:
    """
    Lagged hazard:
      psi_{t+1} = 1 - exp(-eta * Z_t)
      delta_{t+1} = (1 - psi_{t+1})/(1+r) = exp(-eta*Z_t)/(1+r)
    Returns delta_path of length T+1 with delta[0]=nan unused.
    """
    T = len(Z) - 1  # if Z indexed by t=0..T
    delta = np.empty(T + 1, dtype=float)
    delta[0] = np.nan
    # delta[t] links (t-1)->t, so delta[t] uses Z_{t-1}
    for t in range(1, T + 1):
        delta[t] = np.exp(-eta * Z[t - 1]) / (1.0 + r)
    return delta

def solve_inner(T: int, delta_path, use_hotelling_regime=True):
    """Solve ETS+MSR inner model given delta_path; return solution df and params."""
    # Specific bounds for T=22
    backstop = 150.
    bigM = {
        "P": backstop,     # 150
        "B": 5000.0,       # bank/TNAC scale in Mt; generous for 2017–2039
        "TNAC": 5000.0,
        "MSR": 5000.0,
        "MSRpre": 5000.0,
        "S": 2500.0,       # annual supply/auction volume scale
        "I": 2500.0,       # intake
        "R": 2500.0,       # reinjection
        "C": 2500.0,       # cancellation (one-off can be ~2000 in your replication)
    }    
    
    
    cpx, idx, params = build_ets_msr_inner(
        T=T,
        use_hotelling_regime=use_hotelling_regime,
        delta_path=delta_path,
        bigM=bigM,
        eps=1e-6,     
    )
    solve_and_print(cpx)
    df = dump_solution_to_df(cpx, idx, params=params, T=T)
    return df, params

def solve_routeA(
    T: int,
    eta: float,
    omega: float = 0.2,
    tol: float = 1e-6,            # fixed-point tolerance on Z
    max_iter: int = 200,
    Z_mode: str = "legacy",
    out_csv: Path | None = None,
    verbose: bool = True,
):
    import numpy as np
    import pandas as pd

    # --- initialization ---
    df0, params0 = solve_inner(T=T, delta_path=None, use_hotelling_regime=True)
    r = float(params0["r"])
    delta = np.array([np.nan] + [1.0 / (1.0 + r)] * T, dtype=float)

    # Option B: normalization constant (exogenous issuance average)
    # Uses df0 (no MSR feedback) by construction.
    Fbar_exo = compute_Fbar_from_S_issue(df0, T=T)
    
    # Option C and D: normalization variable (exogenous remaining issuance) 
    F_rem_exo = compute_Frem_from_S_issue(df0, T=T)
    
    

    eps_regime = 1e-9  # on/off threshold for signature (negatives clipped anyway)

    history = []
    df_sol = None
    params = None

    Z_prev1 = None
    stable_Z_count = 0
    K_stable = 5

    # store discrete policy signature history (for cycle detection)
    sig_hist = []
    df_hist = []
    t0_year_hist = []
    t_star = None

    # cycle detection parameters
    Pmax = 10
    detected_cycle = None  # (p, k_end)

    for k in range(max_iter):
        if verbose:
            print(f"iter= {k}")

        df_sol, params = solve_inner(T=T, delta_path=delta.tolist(), use_hotelling_regime=True)

        # clean policy variables (important for signature stability)
        for col in ["Intake", "Reinj", "Cancel"]:
            df_sol[col] = df_sol[col].clip(lower=0.0)

        tau_up = float(params["tau_up"])
        tau_low = float(params["tau_low"])
        t0_year = int(params.get("t0_year", 2017))

        # --- lock focal t_star at k=0 (diagnostic only) ---
        if k == 0:
            t_vals = df_sol["t"].to_numpy().astype(int)
            tnac = df_sol["TNAC"].to_numpy()

            mask = t_vals >= 1
            d_up = np.abs(tnac[mask] - tau_up)
            d_low = np.abs(tnac[mask] - tau_low)

            idx = int(np.argmin(np.minimum(d_up, d_low)))
            t_star = int(t_vals[mask][idx])

            if verbose:
                print(f"  [RouteA] locked focal t_star={t_star} (year={t0_year + t_star})")

        row_star = df_sol.loc[df_sol["t"] == t_star].iloc[0]

        # --- compute Z according to mode ---
        if Z_mode == "abatement":
            Z = compute_Z_abatement(df_sol, u=params["u"]).to_numpy()

        elif Z_mode in ("policy_flow_Fbar", "optionB"):
            # Option B: Z_t = (Intake_t - Reinj_t + Cancel_t) / Fbar_exo
            Z = compute_Z_policy_flow_Fbar(df_sol, T=T, Fbar=Fbar_exo)          

        elif Z_mode in ("scarcity_tnac_over_Frem", "tnac_over_Frem", "optionC"):
            # Option C: Z_t = max(0, 1 - TNAC_t / Frem_t)
            Z = compute_Z_tnac_over_Frem(df_sol, T=T, F_rem=F_rem_exo)
        
        elif Z_mode in ("scarcity_supply_over_Frem", "supply_over_Frem", "optionD"):
            # Option D: Z_t = max(0, 1 - S_mkt_t / Frem_t)
            Z = compute_Z_supply_over_Frem(df_sol, T=T, F_rem=F_rem_exo)


            
        else:
            # legacy
            Z = compute_Z(df_sol).to_numpy()

        # --- update delta ---
        delta_new = update_delta_from_Z(Z=Z, r=r, eta=eta)
        delta_next = delta.copy()
        for t in range(1, T + 1):
            delta_next[t] = (1.0 - omega) * delta[t] + omega * delta_new[t]

        # --- compute max_dZ_1 for printing + fixed-point check ---
        max_dZ_1 = None
        j1 = None
        if Z_prev1 is not None:
            d1 = Z - Z_prev1
            max_dZ_1 = float(np.max(np.abs(d1)))
            j1 = int(np.argmax(np.abs(d1)))

        history.append({"iter": k, "max_abs_dZ_1": max_dZ_1})

        if verbose and (max_dZ_1 is not None):
            print(f"  max_dZ_1={max_dZ_1:.3e} at t={j1} (year={t0_year + j1})")
            print(
                f"  [t={t_star} | year={t0_year+t_star}] "
                f"TNAC={row_star['TNAC']:.2f} "
                f"Intake={row_star['Intake']:.2f} "
                f"Reinj={row_star['Reinj']:.2f} "
                f"Cancel={row_star['Cancel']:.2f} "
                f"MSR={row_star['MSR']:.2f}"
            )

        # --- fixed point detection ---
        if max_dZ_1 is not None:
            stable_Z_count = stable_Z_count + 1 if (max_dZ_1 < tol) else 0
            if stable_Z_count >= K_stable:
                print(f"[RouteA] Converged (fixed point): max|ΔZ|<{tol} for {K_stable} iters.")
                delta = delta_next
                break

        # --- build discrete regime signature across ALL t (cycle detection) ---
        t_grid = df_sol["t"].to_numpy().astype(int)
        intake_on = (df_sol["Intake"].to_numpy() > eps_regime).astype(int)
        reinj_on  = (df_sol["Reinj"].to_numpy()  > eps_regime).astype(int)
        cancel_on = (df_sol["Cancel"].to_numpy() > eps_regime).astype(int)

        order = np.argsort(t_grid)
        intake_on = intake_on[order]
        reinj_on  = reinj_on[order]
        cancel_on = cancel_on[order]

        sig = []
        for a, b, c in zip(intake_on, reinj_on, cancel_on):
            sig.extend([int(a), int(b), int(c)])
        sig = tuple(sig)

        sig_hist.append(sig)
        df_hist.append(df_sol.copy())
        t0_year_hist.append(t0_year)

        # keep memory bounded
        keep_last = 4 * Pmax
        if len(sig_hist) > keep_last:
            sig_hist = sig_hist[-keep_last:]
            df_hist  = df_hist[-keep_last:]
            t0_year_hist = t0_year_hist[-keep_last:]

        # --- generic p-cycle detection on signatures ---
        L = len(sig_hist)
        if L >= 4:  # minimum for 2-cycle
            for p in range(2, Pmax + 1):
                if L >= 2 * p:
                    block1 = sig_hist[-2 * p : -p]
                    block2 = sig_hist[-p:]
                    if block1 == block2 and len(set(block2)) > 1:
                        detected_cycle = (p, k)
                        break

            if detected_cycle is not None:
                p, k_end = detected_cycle
                print(f"[RouteA] Detected persistent policy cycle of length p={p} (by regime signature).")

                # save cycle evidence immediately
                if out_csv is not None:
                    out_dir = out_csv.parent / f"{out_csv.stem}_cycle_p{p}"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # last p iterates are one full cycle
                    for i in range(p):
                        df_hist[-p + i].to_csv(out_dir / f"cycle_{i+1:02d}.csv", index=False)

                    # cycle summary at focal t_star (easy to cite)
                    rows = []
                    for i in range(p):
                        df_i = df_hist[-p + i]
                        row_star_i = df_i.loc[df_i["t"] == t_star].iloc[0]
                        rows.append({
                            "cycle_member": i + 1,
                            "iter": (k_end - (p - 1) + i),
                            "year_t_star": int(t0_year + t_star),
                            "TNAC_t_star": float(row_star_i["TNAC"]),
                            "Intake_t_star": float(row_star_i["Intake"]),
                            "Reinj_t_star": float(row_star_i["Reinj"]),
                            "Cancel_t_star": float(row_star_i["Cancel"]),
                            "MSR_t_star": float(row_star_i["MSR"]),
                        })
                    pd.DataFrame(rows).to_csv(out_dir / "cycle_summary.csv", index=False)

                    # dump history as well
                    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

                delta = delta_next
                break

        # advance
        Z_prev1 = Z.copy()
        delta = delta_next

    else:
        print("[RouteA] WARNING: did not converge within max_iter.")

    hist_df = pd.DataFrame(history)

    # --- attach paper-trace columns to df_sol (Z, psi_next, delta_next) ---
    if df_sol is not None:
        # store Z as a per-row series, consistent with mode
        if Z_mode == "abatement":
            df_sol["Z"] = (params["u"] - df_sol["e"]) / params["u"]
        elif Z_mode in ("policy_flow_Fbar", "optionB"):
            Z_store = compute_Z_policy_flow_Fbar(df_sol, T=T, Fbar=Fbar_exo)
            df_sol["Z"] = df_sol["t"].astype(int).map(
                lambda tt: float(Z_store[int(tt)]) if 0 <= int(tt) <= T else np.nan
            )

        elif Z_mode in ("scarcity_tnac_over_Frem", "tnac_over_Frem", "optionC"):
            Z_store = compute_Z_tnac_over_Frem(df_sol, T=T, F_rem=F_rem_exo)
            df_sol["Z"] = df_sol["t"].astype(int).map(
                lambda tt: float(Z_store[int(tt)]) if 0 <= int(tt) <= T else np.nan
            )
    
        elif Z_mode in ("scarcity_supply_over_Frem", "supply_over_Frem", "optionD"):
            Z_store = compute_Z_supply_over_Frem(df_sol, T=T, F_rem=F_rem_exo)
            df_sol["Z"] = df_sol["t"].astype(int).map(
                lambda tt: float(Z_store[int(tt)]) if 0 <= int(tt) <= T else np.nan
            )
            
                                   
            
        else:
            df_sol["Z"] = compute_Z(df_sol)

        # IMPORTANT: lag structure — collapse probability at t+1 depends on Z_t
        psi_next = np.zeros(T + 1)
        for t in range(T):
            zt = float(df_sol.loc[df_sol["t"] == t, "Z"].iloc[0])
            psi_next[t + 1] = 1.0 - np.exp(-eta * zt)
        df_sol["psi_next"] = psi_next

        # record delta used to generate the NEXT iterate
        df_sol["delta_next"] = delta

    # --- final save (always, regardless of converge/cycle/fail) ---
    if out_csv is not None and df_sol is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_sol.to_csv(out_csv, index=False)
        hist_df.to_csv(out_csv.with_name(out_csv.stem + "_history.csv"), index=False)

    return df_sol, hist_df    


if __name__ == "__main__":
    # Example run
    project_root = Path(__file__).resolve().parents[2]
#     out = project_root / "output" / "data" / "simulation"/ "discarded_specs"/"2_cycle"/ "routeA_eta_0p0001.csv"
    
    out = project_root / "output" / "data" / "simulation"/ "supply_over_Frem"/ "routeA_eta_0p01.csv"

    
    
    # Choose T so that t=0 corresponds to 2017 and t=T corresponds to 2039
    # If 2017..2039 inclusive => 23 years => T=22
    # BUT your replication seems to use longer horizons; set T to match what you want to solve.
    solve_routeA(T=22, eta=0.01, omega=0.1, max_iter=200,Z_mode="supply_over_Frem",out_csv=out)