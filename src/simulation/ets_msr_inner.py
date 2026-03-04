import cplex
from cplex.exceptions import CplexError

PROJECT_ROOT ='/Users/irodr/Dropbox/Research (iro@fek.su.se)/Espen/EU_ETS_project/'
REPLICATION_DATA_DIR = PROJECT_ROOT + "output/data/"



def build_ets_msr_inner(
    T=20,
    r=0.08,
    u=2000.0,      # paper baseline u (MtCO2e)
    c=None,         # MAC slope parameter in p = c(u-e) (compute from backstop by default)
    S0=1000.0,     # initial supply baseline (2010 issuance in their slides)
    backstop=150.0, # costs associated with a costly but inexhaustible abatement option
    a=None,        # list/array length T+1: LRF term a_t, used in S_t recursion
    t0_year=None,
    auction_share=0.57,
    tau_up=833.0,
    tau_low=400.0,
    S0_2010=None,
    intake_rate=None,  # list length T+1 with c_t (24% then 12% etc)
    R=100.0,
    TNAC0=1645.0,
    MSR0=0.0,
    S_init=None,       # initial S_0 value for recursion (can equal S0 or set separately)
    bigM=None,
    eps=1e-3,
    use_hotelling_regime=False,
    delta_path=None,   # NEW: list length T+1, delta_path[t] links (t-1)->t
):
    """
    Builds a MILP approximating the equilibrium with MSR and cancellation mechanism.

    Key constraints:
    - e_t = u - p_t/c
    - B_t = B_{t-1} + S_t - e_t, B_t >= 0
    - S_t = S_{t-1} - a_t*S0 - Intake_t + Reinj_t
    - MSR_t = MSR_{t-1} + Intake_t - Reinj_t - Cancel_t
    - MSR/CM rules via binaries + big-M
    - Optional: Hotelling regime via binaries (bank positive -> equality; else inequality)

    Returns: (cpx, var_index_dict)
    """
    
    
    # --- Adding default values ---
    # def build_ets_msr_milp(..., T=53, t0_year=2017, S0_2010=2199.0, S_init=None, a=None, auction_share=0.57, ...):
    
    # --- 0) Defaults / calendar mapping ---
    if t0_year is None:
        t0_year = 2017
    
    if S0_2010 is None:
        S0_2010 = 2199.0  # paper: issued allowances in 2010 (million)
    
    # --- 1) LRF schedule a[t] (paper: 1.74% before 2021, 2.2% from 2021 onward) ---
    # We keep a[0]=0.0 by convention; for t>=1 a[t] depends on calendar year (t0_year+t).
    if a is None:
        a = [0.0] * (T + 1)
        for t in range(1, T + 1):
            year = t0_year + t
            a[t] = 0.0174 if year <= 2020 else 0.022
    
    # --- 2) Compute issuance in year t=0 implied by 2010 baseline, if S_init not provided ---
    # If t0_year=2017, there are 7 steps from 2010->2017: 2011..2017 inclusive.
    # Paper simplification: apply the pre-2021 LRF (1.74%) for all those pre-2021 years.
    if S_init is None:
        years_from_2010_to_t0 = t0_year - 2010
        pre_LRF = 0.0174
        S_init = S0_2010 - years_from_2010_to_t0 * pre_LRF * S0_2010
        S_init = max(0.0, S_init)
    
    # --- 3) Build issued allowances path S_issue[t] using the paper formula, clipped at 0 ---
    # S_issue[t] = max(0, S_issue[t-1] - a[t]*S0_2010)
    S_issue = [0.0] * (T + 1)
    S_issue[0] = S_init
    for t in range(1, T + 1):
        S_issue[t] = max(0.0, S_issue[t-1] - a[t] * S0_2010)
    
    # --- 4) Gross auctions path G[t] = auction_share * S_issue[t] ---
    G = [auction_share * S_issue[t] for t in range(T + 1)]
    
    # --- 5) Paper baseline: set cost slope so that p_max = c*u = backstop (150 EUR/t)
    if c is None:
        c = backstop / u
    
    # ---- 6) Default: full credibility (Hotelling)
    if delta_path is None:
        delta_path = [None] + [1.0 / (1.0 + r)] * T
    assert len(delta_path) == T + 1
    
    
    # --- 7) (optional) store params for debugging/DF dumps ---
    # Make sure your function returns params (cpx, idx, params)
    params = params if "params" in locals() and params is not None else {}
    params.update({
        "t0_year": t0_year,
        "S0_2010": S0_2010,
        "S_init": S_init,
        "a": a,
        "S_issue": S_issue,
        "G": G,
        "auction_share": auction_share,
        "u": u,
        "c": c,
        "r": r,
        "tau_up": tau_up,
        "tau_low": tau_low,
        "R": R,
    })

    # ---- 8) Big-M defaults (tighten later once you have paper calibration ranges)
    if bigM is None:
        bigM = {
            "TNAC": 1e6,
            "I": 1e6,
            "R": 1e6,
            "C": 1e6,
            "P": 1e6,
            "B": 1e6,
            "MSR": 1e6,
            "MSRpre": 1e6,
            "S": 1e6,
        }
        
    
    # exogenous MSR endowments (paper calibration)
    # t=0 is 2017  =>  t=2 is 2019, t=3 is 2020
    msr_endow = {2: 900.0, 3: 600.0} 
        

    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    
    # helper to create variable names by time
    def vname(prefix, t):
        return f"{prefix}_{t}"



    # ---- Decision variables
    # Continuous vars (all years 0..T)
    cont_prefixes = ["p", "e", "B", "S", "MSR", "Intake", "Reinj", "Cancel", "TNAC","MSRpre"]
    cont_vars = [vname(pref, t) for pref in cont_prefixes for t in range(T + 1)]

    # Binaries for policy rules (years 1..T typically depend on t-1)
    bin_vars = []
    for t in range(1, T + 1):
        bin_vars += [
            vname("z_up", t),    # TNAC_{t-1} >= tau_up
            vname("z_low", t),   # TNAC_{t-1} < tau_low
            vname("z_R", t),     # MSR_t >= R
            vname("z_CM", t),    # MSR_t >= auction_{t-1}
        ]
        if use_hotelling_regime:
            bin_vars += [vname("z_bank", t)]  # banking regime indicator (for Hotelling)

    all_vars = cont_vars + bin_vars

    # Variable types
    types = ""
    lb = []
    ub = []

    # Continuous bounds
    for name in cont_vars:
        if name.startswith("p_"):
            types += "C"; lb.append(0.0); ub.append(backstop)
        elif name.startswith("B_"):
            types += "C"; lb.append(0.0); ub.append(bigM["B"])
        elif name.startswith("MSR_"):
            types += "C"; lb.append(0.0); ub.append(bigM["MSR"])
        elif name.startswith("MSRpre_"):
            types += "C"; lb.append(0.0); ub.append(bigM["MSRpre"])           
        elif name.startswith("S_"):
            types += "C"; lb.append(0.0); ub.append(bigM["S"])
        elif name.startswith("Intake_") or name.startswith("Reinj_") or name.startswith("Cancel_"):
            types += "C"; lb.append(0.0); ub.append(max(bigM["I"], bigM["R"], bigM["C"]))
        elif name.startswith("TNAC_"):
            types += "C"; lb.append(0.0); ub.append(bigM["TNAC"])
        elif name.startswith("e_"):
            # emissions can be bounded [0, u] for realism
            types += "C"; lb.append(0.0); ub.append(u)
        else:
            types += "C"; lb.append(-cplex.infinity); ub.append(cplex.infinity)

    # Binary bounds
    for name in bin_vars:
        types += "B"; lb.append(0.0); ub.append(1.0)

    cpx.variables.add(names=all_vars, types=types, lb=lb, ub=ub)

    # Build a quick name->index map
    idx = {name: j for j, name in enumerate(all_vars)}

    # ---- Objective
    # Important: for equilibrium-as-feasibility, any small objective is fine.
    # Here: minimize sum of prices (or emissions) just to select a point. You can change later.
    obj = [(idx[vname("p", t)], 1.0) for t in range(T + 1)]
    cpx.objective.set_linear(obj)
    
    
    # ----- Objective: MIQP tie-breaker (abatement cost)
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    
    n = cpx.variables.get_num()
    
    # Clear linear objective first
    cpx.objective.set_linear([(j, 0.0) for j in range(n)])
    
    # Linear coefficients and diagonal Q
    lin = [0.0] * n
    Q = [([], []) for _ in range(n)]  # each row: (ind_list, val_list)
    
    discount = 1.0
    for t in range(T + 1):
        et = idx[f"e_{t}"]
        w = discount  # (1/(1+r))^t
    
        # linear term: w * (-c*u) * e_t
        lin[et] += w * (-c * u)
    
        # quadratic term: (1/2) * Q_ii * e_t^2 with Q_ii = w*c
        Q[et] = ([et], [w * c])
    
        discount /= (1.0 + r)
    
    # Apply
    cpx.objective.set_linear(list(enumerate(lin)))
    cpx.objective.set_quadratic(Q)  
    
    # helper to add equations
    def add_eq(lhs, rhs, sense="E", name=None):
        # lhs: list of (varname, coeff); rhs is float
        ind = [idx[v] for v, _ in lhs]
        val = [coef for _, coef in lhs]
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=ind, val=val)],
            senses=[sense],
            rhs=[rhs],
            names=[name] if name else None,
        )
       

    # ---- Constraints
    # 1) Define TNAC_t (closer to paper than TNAC=B):
    # TNAC_t = B_t  (replication-consistent; TNAC excludes MSR stock)
    for t in range(T + 1):
        add_eq([(vname("TNAC", t), 1.0), (vname("B", t), -1.0)],
               0.0, "E", f"TNAC_def_{t}")               
    # Initial TNAC (paper: 2017 = 1645)
    add_eq([(vname("TNAC", 0), 1.0)], TNAC0, "E", "TNAC0")            

    # 2) MAC condition: e_t = u - p_t/c  <=>  e_t + (1/c)p_t = u
    invc = 1.0 / c
    for t in range(T + 1):
        add_eq([(vname("e", t), 1.0), (vname("p", t), invc)], u, "E", f"MAC_{t}")

    # 3) Banking law (TOTAL supply to market, not auctions)
    # B_t = B_{t-1} + S_issue[t] - Intake_t + Reinj_t - e_t
    B0 = TNAC0
    add_eq([(vname("B", 0), 1.0)], B0, "E", "B0")
    
    for t in range(1, T + 1):
        add_eq(
            [
                (vname("B", t), 1.0),
                (vname("B", t-1), -1.0),
                (vname("Intake", t), 1.0),
                (vname("Reinj", t), -1.0),
                (vname("e", t), 1.0),
            ],
            S_issue[t],
            "E",
            f"Bank_{t}",
        )
    
    add_eq([(vname("B", T), 1.0)], 0.0, "E", "B_terminal")


    # 4a) MSR recursion: MSR_t = MSRpre_t - Cancel_t 
    add_eq([(vname("MSR", 0), 1.0)], MSR0, "E", "MSR0")
    for t in range(1, T + 1):
        add_eq([(vname("MSR", t), 1.0),
                (vname("MSRpre", t), -1.0),
                (vname("Cancel", t), 1.0)],
               0.0, "E", f"MSR_post_{t}")
                   
    # 4a) MSRpre_t = MSR_{t-1} + Intake_t - Reinj_t
    # Introducing MSR initial endowments in 2019 and 2020 
    MSR_endow = [0.0] * (T + 1)
    MSR_endow[2019 - t0_year] = 900.0   # t=2 if t0_year=2017
    MSR_endow[2020 - t0_year] = 600.0   # t=3 if t0_year=2017   
    for t in range(1, T + 1):
#         add_eq([(vname("MSRpre", t), 1.0),
#                 (vname("MSR", t-1), -1.0),
#                 (vname("Intake", t), -1.0),
#                 (vname("Reinj", t),  1.0)],
#                0.0, "E", f"MSRpre_def_{t}")
        rhs = MSR_endow[t]
        add_eq([(vname("MSRpre", t), 1.0),
                (vname("MSR", t-1), -1.0),
                (vname("Intake", t), -1.0),
                (vname("Reinj", t),  1.0)],
               rhs, "E", f"MSRpre_def_{t}")
       
          
        
    # 4c) Replace your auction supply constraints with: S_t + Intake_t - Reinj_t = G[t] ---
    # IMPORTANT: delete/disable any old recursion like S_t = S_{t-1} - a[t]*S0 - Intake + Reinj
    add_eq([(vname("S", 0), 1.0)], G[0], "E", "S0")
    for t in range(1, T + 1):
        add_eq([(vname("S", t), 1.0),
                (vname("Intake", t), 1.0),
                (vname("Reinj", t), -1.0)],
               G[t],
               "E",
               f"S_def_{t}")
        
            
        

    # 5) MSR Intake rule with binary z_up_t:
    # If TNAC_{t-1} >= tau_up => Intake_t = intake_rate[t] * TNAC_{t-1}, else Intake_t = 0.
    # Default intake rate schedule (paper): 24% until 2023, 12% from 2024 onward
    if intake_rate is None:
        intake_rate = [0.0] * (T + 1)
        for t in range(1, T + 1):
            year = t0_year + t   # since t=0 is t0_year
            intake_rate[t] = 0.24 if year <= 2023 else 0.12 
    M_tau = bigM["TNAC"]
    M_I = bigM["I"]    
    # First effective MSR intake is in 2019 (t=2), so force 2018 intake to zero
    add_eq([(vname("Intake", 1), 1.0)], 0.0, "E", "Intake_1_zero")
    add_eq([(vname("z_up", 1), 1.0)], 0.0, "E", "zup_1_zero")    
    for t in range(2, T + 1):
        z = vname("z_up", t)
        TN = vname("TNAC", t-1)
        I = vname("Intake", t)
        ct = intake_rate[t]

        # Threshold encoding for z_up (tight):
        # z=1 => TN >= tau_up
        # z=0 => TN <= tau_up - eps
        
        # If z=1 => TN >= tau_up; if z=0 relaxed
        # TN >= tau_up - M*(1-z)  <=>  TN - M*z >= tau_up - M
        add_eq([(TN, 1.0), (z, -M_tau)],
               tau_up - M_tau, "G", f"up_thr_L_{t}")
        
        # If z=0 => TN <= tau_up - eps; if z=1 relaxed
        # TN <= tau_up - eps + M*z  <=>  TN - M*z <= tau_up - eps
        add_eq([(TN, 1.0), (z, -M_tau)],
               tau_up - eps, "L", f"up_thr_U_{t}")

        # Intake = ct*TN when z=1; Intake=0 when z=0
        # Intake <= M_I z
        add_eq([(I, 1.0), (z, -M_I)], 0.0, "L", f"Intake_cap_{t}")

        # Intake - ct*TN <= M_I(1-z)  -> Intake - ct*TN + M_I z <= M_I
        add_eq([(I, 1.0), (TN, -ct), (z, M_I)], M_I, "L", f"Intake_link_U_{t}")
        # ct*TN - Intake <= M_I(1-z)  -> ct*TN - Intake + M_I z <= M_I
        add_eq([(TN, ct), (I, -1.0), (z, M_I)], M_I, "L", f"Intake_link_L_{t}")
        
        # Intake cannot exceed gross auctions
        add_eq(
            [(vname("Intake", t), 1.0)],
            G[t],
            "L",
            f"Intake_le_G_{t}",
        )        

    # 6) Reinjection rule:
    # If TNAC_{t-1} < tau_low AND MSR_t >= R => Reinj_t = R
    # If TNAC_{t-1} < tau_low AND MSR_t <  R => Reinj_t = MSR_t
    # Else Reinj_t = 0
    M_R = bigM["R"]
    M_MSR = bigM["MSR"]
    for t in range(1, T + 1):
        zlow = vname("z_low", t)
        zR = vname("z_R", t)
        TN = vname("TNAC", t-1)
        MSR = vname("MSR", t)
        Re = vname("Reinj", t)

        # low-threshold encoding for z_low:
        # If zlow=1 => TN <= tau_low - eps
        # If zlow=0 => TN >= tau_low
        # TN - tau_low <= -eps + M(1-zlow)  <=> TN + M*zlow <= tau_low - eps + M
        add_eq([(TN, 1.0), (zlow, M_tau)], tau_low - eps + M_tau, "L", f"low_thr_U_{t}")
        
        # TN - tau_low >= -M*zlow  <=> TN - M*zlow >= tau_low - M
        add_eq([(TN, 1.0), (zlow, M_tau)], tau_low, "G", f"low_thr_L_{t}")        
               

        # MSR >= R indicator (tight)
        # If zR=1 => MSR >= R
        # If zR=0 => MSR <= R - eps
        
        add_eq([(MSR, 1.0), (zR, -M_MSR)], R - M_MSR, "G", f"msrR_thr_L_{t}")
        add_eq([(MSR, 1.0), (zR, -M_MSR)], R - eps,   "L", f"msrR_thr_U_{t}")        
        

        # If not low => Reinj = 0 enforced by Reinj <= M_R * zlow
        add_eq([(Re, 1.0), (zlow, -M_R)], 0.0, "L", f"Reinj_cap_low_{t}")

        # Case low & zR=1 => Reinj = R
        # |Re - R| <= M*( (1-zlow) + (1-zR) )  implemented with two inequalities
        add_eq([(Re, 1.0), (zlow, M_R), (zR, M_R)], R + 2*M_R, "L", f"Reinj_R_U_{t}")
        add_eq([(Re, -1.0), (zlow, M_R), (zR, M_R)], -R + 2*M_R, "L", f"Reinj_R_L_{t}")

        # Case low & zR=0 => Reinj = MSR
        # |Re - MSR| <= M*( (1-zlow) + zR )
        add_eq([(Re, 1.0), (MSR, -1.0), (zlow, M_R), (zR, -M_R)],  M_R, "L", f"Reinj_MSR_U_{t}")
        add_eq([(Re, -1.0), (MSR,  1.0), (zlow, M_R), (zR, -M_R)], M_R, "L", f"Reinj_MSR_L_{t}")

    # 7) Cancellation mechanism (net auctions last year)
    # Cancel_t = max(0, MSRpre_t - S_{t-1})
    M = bigM["C"]
    t_cancel_start = 6  # 2023
    # --- Early years: cancellation fully off ---
    for t in range(1, t_cancel_start):
        add_eq([(vname("Cancel", t), 1.0)], 0.0, "E", f"Cancel_off_{t}")
        add_eq([(vname("z_CM", t), 1.0)], 0.0, "E", f"zCM_off_{t}")
    
    # --- Cancellation active from 2023 onward ---
    for t in range(t_cancel_start, T + 1):
        z  = vname("z_CM", t)
        MP = vname("MSRpre", t)
        Cn = vname("Cancel", t)
        S_prev = vname("S", t-1)
    
        # gap = MP − S_prev
        # z=1 => gap ≥ 0
        # z=0 => gap ≤ 0   (NO eps here!)
    
        # gap ≥ -M(1−z)
        add_eq([(MP, 1.0), (S_prev, -1.0), (z, M)],
               -M, "G", f"CM_gap_L_{t}")
    
        # gap ≤ M z
        add_eq([(MP, 1.0), (S_prev, -1.0), (z, -M)],
               0.0, "L", f"CM_gap_U_{t}")
    
        # Cancel = gap when z=1, else 0
        add_eq([(Cn, 1.0), (MP, -1.0), (S_prev, 1.0)],
               0.0, "G", f"Cancel_ge_gap_{t}")
    
        add_eq([(Cn, 1.0), (MP, -1.0), (S_prev, 1.0), (z, M)],
               M, "L", f"Cancel_le_gap_{t}")
    
        add_eq([(Cn, 1.0), (z, -M)],
               0.0, "L", f"Cancel_cap_{t}")                            
               

    # 8) Optional: Hotelling regime logic (bank>0 => equality; bank=0 => p_t <= (1+r)p_{t-1})
    if use_hotelling_regime:
        M_P = bigM["P"]
        for t in range(1, T + 1):
            zb = vname("z_bank", t)  # interpret as: banking active at t-1
            B_prev = vname("B", t-1)
            p = vname("p", t)
            pprev = vname("p", t-1)

            # Link B_prev and zb: if zb=1 then B_prev >= eps ; if zb=0 then B_prev can be 0
            add_eq([(B_prev, 1.0), (zb, -bigM["B"])], 0.0, "L", f"B_zb_U_{t}")     # B_prev <= M*zb
            add_eq([(B_prev, 1.0), (zb, -eps)], 0.0, "G", f"B_zb_L_{t}")           # B_prev >= eps*zb

            # Hazard-adjusted growth factor for link (t-1)->t
            # p_t = (1/delta_t) * p_{t-1} in the interior banking regime
            growth_t = 1.0 / float(delta_path[t])

            pmax = backstop  # since p is bounded by backstop in your model
            M_t = (1.0 + growth_t) * pmax   # safe big-M for this period
            
            # Enforce equality when zb=1: p - growth_t*pprev = 0
            add_eq([(p, 1.0), (pprev, -growth_t), (zb, M_t)], M_t, "L", f"Hot_U_{t}")
            add_eq([(p, -1.0), (pprev, growth_t), (zb, M_t)], M_t, "L", f"Hot_L_{t}")
            
            # Always enforce p <= growth_t*pprev when zb=0
            add_eq([(p, 1.0), (pprev, -growth_t), (zb, -M_t)], 0.0, "L", f"Hot_ineq_{t}")                  

    # 9) Mutually exclusive up/low triggers (optional but helpful): z_up + z_low <= 1
    for t in range(1, T + 1):
        add_eq([(vname("z_up", t), 1.0), (vname("z_low", t), 1.0)], 1.0, "L", f"z_excl_{t}")
        
        
    # 10) Add explicit nonnegativity constraints for bank (Even though the bounds should do it, 
    #     adding explicit constraints helps prevent presolve from loosening things)
    for t in range(T+1):
        add_eq([(vname("B", t), 1.0)], 0.0, "G", f"B_nonneg_{t}")
        add_eq([(vname("p", t), 1.0)], 0.0, "G", f"p_nonneg_{t}")
        
        
    params.update({
        "intake_rate": intake_rate,
    })
    return cpx, idx, params

import pandas as pd

def dump_solution_to_df(cpx, idx, params=None, T=None):
    """
    Build a pandas DataFrame with all key time series in one shot.
    Assumes variable names like 'p_0', 'TNAC_0', 'MSR_1', etc.
    Saves constants like G[t] if provided in params["G"].
    """
    sol = cpx.solution

    # Infer T if not provided
    if T is None:
        # look for largest t among S_t indices
        ts = []
        for k in idx.keys():
            if "_" in k:
                name, t = k.rsplit("_", 1)
                if t.isdigit():
                    ts.append(int(t))
        T = max(ts) if ts else 0
    
    # Pull mapping info
    t0_year = None
    if params is not None:
        t0_year = params.get("t0_year", None)

    # Helper to fetch value safely
    def get(name, t):
        key = f"{name}_{t}"
        if key in idx:
            return sol.get_values(idx[key])
        return None

    G = None
    if params is not None and "G" in params:
        G = params["G"]
        
    S_issue = None
    if params is not None and "S_issue" in params:
        S_issue = params["S_issue"]        

    rows = []
    for t in range(0, T + 1):
        row = {"t": t}
        
        # Add calendar year if available
        row["year"] = (t0_year + t) if t0_year is not None else None

        # Exogenous gross auctions
        row["G"] = G[t] if G is not None and t < len(G) else None
        
        # Total Issued allowances
        row["S_issue"] = S_issue[t] if S_issue is not None and t < len(S_issue) else None

        # Core variables
        for v in ["S", "p", "e", "B", "TNAC", "MSR", "Intake", "Reinj", "Cancel"]:
            row[v] = get(v, t)

        # Binaries (might not exist for all t)
        for z in ["z_up", "z_low", "z_R", "z_CM"]:
            row[z] = get(z, t)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Some helpful derived columns
    if "G" in df.columns and "S" in df.columns:
        df["G_minus_S"] = df["G"] - df["S"]
    if "Intake" in df.columns and "Reinj" in df.columns and "Cancel" in df.columns:
        df["Net_MSR_flow"] = df["Intake"].fillna(0) - df["Reinj"].fillna(0) - df["Cancel"].fillna(0)        
    if {"S_issue", "Intake", "Reinj"}.issubset(df.columns):
        df["Supply_to_market"] = (
            df["S_issue"]
            - df["Intake"].fillna(0)
            + df["Reinj"].fillna(0)
        )

    return df


        

def solve_and_print(cpx):
    try:
        # Silence solver streams
        cpx.set_log_stream(None)
        cpx.set_error_stream(None)
        cpx.set_warning_stream(None)
        cpx.set_results_stream(None) 
        # Set tolerances       
        cpx.parameters.simplex.tolerances.feasibility.set(1e-9)
        cpx.parameters.mip.tolerances.integrality.set(1e-9)
        cpx.parameters.mip.tolerances.mipgap.set(1e-12)
        cpx.parameters.barrier.convergetol.set(1e-10)
        # Solve               
        cpx.solve()
    except CplexError as e:
        print("CPLEX error:", e)


if __name__ == "__main__":
    cpx, idx, params = build_ets_msr_milp(T=53, use_hotelling_regime=True)
    solve_and_print(cpx)
    df = dump_solution_to_df(cpx, idx, params=params, T=53)
    # --- parameters (use the ones already in your model) ---
    u = params["u"]          # e.g. 2000
    c = params["c"]          # e.g. 0.075
    r = params["r"]          # e.g. 0.08 (whatever you use for discounting / Hotelling)    
         
    # Interior-FOC mapping: p = c (u - e)  <=>  e = u - (1/c) p
    beta = 1.0 / c
    df["e_implied"] = u - beta * df["p"]
    df["foc_resid"] = df["e"] - df["e_implied"]
    df["foc_abs"] = df["foc_resid"].abs()
    df["is_interior_like"] = (df["foc_abs"] < 1e-8)
    
    # Price growth
    df["p_lag"] = df["p"].shift(1)
    df["p_gross_growth"] = df["p"] / df["p_lag"]
    df["p_net_growth"] = df["p_gross_growth"] - 1.0
    
    # Hotelling deviation (net)
    df["hotelling_net"] = r
    df["hotelling_dev"] = df["p_net_growth"] - r
    # csv  
    df.to_csv(REPLICATION_DATA_DIR+"msr_replication.csv", index=False)
    # excel
    df.to_excel(REPLICATION_DATA_DIR+"msr_replication.xlsx", index=False, engine="openpyxl")    
    print(df.head())

