import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
# Change to where the two cycle files to be compared are
out = project_root / "output" / "data" / "simulation"/ "policy_flow_Fbar" / "routeA_eta_0p2_cycle_p2"



csv1 = out / "cycle_01.csv"
csv2 = out / "cycle_02.csv"

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

# sanity check
assert (df1["t"].values == df2["t"].values).all(), "t grids do not match"

cols_to_compare = [
    "B", "TNAC", "e", "p",
    "Intake", "Reinj", "Cancel",
    "MSR", "S"
]

rows = []
for col in cols_to_compare:
    diff = df1[col] - df2[col]
    rows.append({
        "variable": col,
        "max_abs_diff": np.max(np.abs(diff)),
        "mean_abs_diff": np.mean(np.abs(diff)),
    })

comparison_table = pd.DataFrame(rows).sort_values("max_abs_diff", ascending=False)
print(comparison_table)


comparison_table.to_csv(out/"cycle_numeric_comparison.csv", index=False)

