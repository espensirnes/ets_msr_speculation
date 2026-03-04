"""
Simple dynamic simulation of an EU ETS + MSR model in Python.

This is NOT a calibrated model, but a transparent sandbox that
implements the core feedback:

  banking (low emissions) -> higher TNAC
  -> higher MSR intake -> more invalidation
  -> lower future effective cap -> higher price growth incentive.

You can tweak parameters and functional forms as you like.

Author: ChatGPT (based on Bocklet-style structure)
"""

import numpy as np
from dataclasses import dataclass

# ======================
# 1. PARAMETERS
# ======================

# Time settings
START_YEAR = 2024
END_YEAR = 2050
T = END_YEAR - START_YEAR + 1  # number of periods (annual)

DISCOUNT_RATE = 0.03  # r in Hotelling condition
INITIAL_PRICE = 80.0  # EUR per ton at START_YEAR

# Cap path (very stylized: linear decline)
# Think of this as million allowances per year.
INITIAL_CAP = 1500.0  # e.g. 1.5 billion allowances in 2024
CAP_DECLINE_RATE = 0.043  # 4.3% per year decline

# Auction share (simplification: fixed fraction of cap is auctioned)
AUCTION_SHARE = 0.6

# MSR parameters (Phase 4-ish, simplified)
TNAC_LOWER_THRESHOLD = 833.0   # million allowances
TNAC_UPPER_THRESHOLD = 1096.0  # million allowances
MSR_INTAKE_RATE = 0.12         # 24% of TNAC when TNAC > lower threshold
MSR_CAP_LEVEL = 400.0          # million allowances kept in MSR; above is invalidated

# Initial TNAC and MSR (in million allowances)
INITIAL_TNAC = 1400.0
INITIAL_MSR = 400.0

# Demand curve parameters: e_t = a - b * p_t
# (Very stylized; you can replace with anything.)
DEMAND_INTERCEPT = 1700.0  # a
DEMAND_SLOPE = 3.0         # b  (so 1 EUR more reduces demand by 5 million t)

MSR_INTAKE_RATE = 0.12
INITIAL_TNAC = 1400.0
INITIAL_MSR = 400.0
DEMAND_INTERCEPT = 1700.0  # lower demand overall
DEMAND_SLOPE = 3.0


# ======================
# 2. DATA STRUCTURES
# ======================

@dataclass
class State:
    year: int
    price: float
    cap: float
    auction_volume: float
    emissions: float
    tnac: float
    msr: float
    msr_intake: float
    invalidation: float


# ======================
# 3. MODEL FUNCTIONS
# ======================

def cap_path(t: int) -> float:
    """
    Cap_t as a function of t (0-based index).
    Units: million allowances.
    """
    cap_t = INITIAL_CAP * ((1 - CAP_DECLINE_RATE) ** t)
    return cap_t


def auction_volume(cap_t: float) -> float:
    """
    Auction volume A_t.
    Simplified as a fixed share of cap.
    """
    return AUCTION_SHARE * cap_t


def demand_for_allowances(price_t: float) -> float:
    """
    Emissions (demand for allowances) e_t = D(p_t).
    Simple linear demand with truncation at zero.
    """
    e_t = DEMAND_INTERCEPT - DEMAND_SLOPE * price_t
    return max(e_t, 0.0)


def msr_intake(tnac_t: float, auction_t: float) -> float:
    """
    MSR intake I_t as a function of TNAC_t.
    Follows the stylized EU rule:
      - 0 if TNAC < lower threshold
      - alpha * TNAC if between thresholds (capped at auction volume)
      - auction_t if TNAC > upper threshold
    """
    if tnac_t < TNAC_LOWER_THRESHOLD:
        return 0.0
    elif tnac_t <= TNAC_UPPER_THRESHOLD:
        intake = MSR_INTAKE_RATE * tnac_t
        return min(intake, auction_t)
    else:
        # When TNAC above upper threshold, all auction volume can be diverted
        return auction_t


def msr_invalidation(msr_before_invalidation: float) -> float:
    """
    Invalidation S_t = max(0, MSR_t - MSR_CAP_LEVEL)
    Everything above MSR_CAP_LEVEL is cancelled.
    """
    return max(0.0, msr_before_invalidation - MSR_CAP_LEVEL)


def update_price(prev_price: float) -> float:
    """
    Hotelling condition: p_{t} = p_{t-1} * (1 + r)
    (No political risk / shocks here; purely mechanical.)
    """
    return prev_price * (1 + DISCOUNT_RATE)


def simulate_step(prev_state: State, t_index: int) -> State:
    """
    Simulate one period (year t) from previous state.
    t_index = 0 for START_YEAR, 1 for START_YEAR+1, etc.
    """
    year = START_YEAR + t_index

    # 1. Price (Hotelling forward)
    if t_index == 0:
        price_t = prev_state.price  # first period uses initial price
    else:
        price_t = update_price(prev_state.price)

    # 2. Cap and auction volume
    cap_t = cap_path(t_index)
    auction_t = auction_volume(cap_t)

    # 3. Emissions from demand (bounded above by cap_t)
    emissions_t = min(demand_for_allowances(price_t), cap_t)

    # 4. MSR intake based on TNAC_t-1
    intake_t = msr_intake(prev_state.tnac, auction_t)

    # 5. Update MSR (before invalidation)
    msr_temp = prev_state.msr + intake_t

    # 6. Invalidation S_t
    invalidation_t = msr_invalidation(msr_temp)
    msr_t = msr_temp - invalidation_t

    # 7. Update TNAC:
    # TNAC_{t} = TNAC_{t-1} + Cap_t - Emissions_t - Intake_t
    # (invalidation removes allowances already in MSR, so does not change TNAC)
    tnac_t = prev_state.tnac + cap_t - emissions_t - intake_t

    return State(
        year=year,
        price=price_t,
        cap=cap_t,
        auction_volume=auction_t,
        emissions=emissions_t,
        tnac=tnac_t,
        msr=msr_t,
        msr_intake=intake_t,
        invalidation=invalidation_t,
    )


# ======================
# 4. MAIN SIMULATION
# ======================

def main():
    # Initial state at START_YEAR (t_index = 0)
    initial_state = State(
        year=START_YEAR,
        price=INITIAL_PRICE,
        cap=cap_path(0),
        auction_volume=auction_volume(cap_path(0)),
        emissions=demand_for_allowances(INITIAL_PRICE),
        tnac=INITIAL_TNAC,
        msr=INITIAL_MSR,
        msr_intake=0.0,
        invalidation=0.0,
    )

    states = [initial_state]

    # Simulate years START_YEAR+1 ... END_YEAR
    prev_state = initial_state
    for t_index in range(1, T):
        new_state = simulate_step(prev_state, t_index)
        states.append(new_state)
        prev_state = new_state

    # Simple textual output (you can replace with plotting / CSV export)
    print(f"{'Year':>4}  {'Price':>8}  {'Cap':>8}  {'Emiss':>8}  {'TNAC':>10}  {'MSR':>10}  {'Intake':>8}  {'Invalid':>8}")
    for s in states:
        print(f"{s.year:4d}  {s.price:8.1f}  {s.cap:8.1f}  {s.emissions:8.1f}  {s.tnac:10.1f}  {s.msr:10.1f}  {s.msr_intake:8.1f}  {s.invalidation:8.1f}")

    # If you want to work further:
    # - return `states` instead of just printing
    # - or write to CSV / plot with matplotlib.


if __name__ == "__main__":
    main()
