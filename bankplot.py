import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# -----------------------------
# Solve for T >= 0 in:
# (1-phi)^T = eta(1-phi) / (A - phi(1-eta)^T)
# -----------------------------

def main():
    plot_mesh()  # <-- set A here
    dT_deta_a, dT_dphi_a= [], []
    for A in [1, 5, 10, 20]:  # <-- set A values here
        p, e, t, _, _ = calc_mesh(A)
        print(f"A={A}: phi range: {np.nanmin(p):.2f} to {np.nanmax(p):.2f}")
        print(f"A={A}: eta range: {np.nanmin(e):.2f} to {np.nanmax(e):.2f}")
        dT_deta = numerical_slopes(p, e, t, which="eta")
        dT_dphi = numerical_slopes(p, e, t, which="phi")
        dT_deta_a.extend(dT_deta)
        dT_dphi_a.extend(dT_dphi)

    slope_quadrant_plot(dT_deta_a, dT_dphi_a)  # also shows the slope-vs-slope scatter plot
    a=0





@dataclass
class Params:
    A: float = 1.0       # <-- set this
    Tmin: float = 0.0
    Tmax: float = 200.0  # increase if you expect very large T
    n_scan: int = 4000   # scan points to find sign changes robustly

def f_T(T, phi, eta, A):
    """Residual f(T)=LHS-RHS for the equation."""
    if T < 0:
        return np.nan

    x = 1.0 - phi
    y = 1.0 - eta

    # LHS: x^T with careful handling for x=0 and T=0
    if x == 0.0:
        lhs = 1.0 if T == 0.0 else 0.0
    else:
        lhs = x**T

    # Denominator: A - phi*y^T
    if y == 0.0:
        yT = 1.0 if T == 0.0 else 0.0
    else:
        yT = y**T

    denom = A - phi * yT
    if denom <= 0 or not np.isfinite(denom):
        return np.nan

    rhs = eta * x / denom
    return lhs - rhs

def solve_T(phi, eta, p: Params):
    """
    Find a root T in [Tmin, Tmax] if it exists.
    We scan for the first sign change among valid points, then use brentq.
    """
    # skip endpoints (phi or eta exactly 0/1 can create edge behaviour)
    if not (0.0 < phi < 1.0 and 0.0 < eta < 1.0):
        return np.nan

    Ts = np.linspace(p.Tmin, p.Tmax, p.n_scan)
    vals = np.array([f_T(T, phi, eta, p.A) for T in Ts], dtype=float)

    finite = np.isfinite(vals)
    if finite.sum() < 2:
        return np.nan

    # Find first sign change among consecutive finite points
    idx = np.where(finite[:-1] & finite[1:] & (np.sign(vals[:-1]) != np.sign(vals[1:])))[0]
    if len(idx) == 0:
        return np.nan

    i = idx[0]
    a, b = Ts[i], Ts[i + 1]
    fa, fb = vals[i], vals[i + 1]

    if not (np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0):
        return np.nan

    try:
        return brentq(lambda T: f_T(T, phi, eta, p.A), a, b, maxiter=200)
    except Exception:
        return np.nan
    
import numpy as np
import matplotlib.pyplot as plt

def numerical_slopes(PHI, ETA, Tsol, which="phi"):
    PHI, ETA, Tsol = np.asarray(PHI), np.asarray(ETA), np.asarray(Tsol)
    dphi = PHI[0, 1] - PHI[0, 0]
    deta = ETA[1, 0] - ETA[0, 0]

    slopes = np.full_like(Tsol, np.nan, dtype=float)

    if which == "phi":
        slopes[:, 1:-1] = (Tsol[:, 2:] - Tsol[:, :-2]) / (2 * dphi)
        slopes[:, 0]    = (Tsol[:, 1] - Tsol[:, 0]) / dphi
        slopes[:, -1]   = (Tsol[:, -1] - Tsol[:, -2]) / dphi

    elif which == "eta":
        slopes[1:-1, :] = (Tsol[2:, :] - Tsol[:-2, :]) / (2 * deta)
        slopes[0, :]    = (Tsol[1, :] - Tsol[0, :]) / deta
        slopes[-1, :]   = (Tsol[-1, :] - Tsol[-2, :]) / deta

    else:
        raise ValueError("which must be 'phi' or 'eta'")

    return slopes.flatten()  # keep 2D

def slope_quadrant_plot(dT_deta, dT_dphi):
    

    x = np.array(dT_deta)
    y = np.array(dT_dphi)

    # Quick diagnostics
    share_third = np.mean((x < 0) & (y < 0))
    share_x_neg = np.mean(x < 0)
    share_y_neg = np.mean(y < 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(x, y, c=x, edgecolors="none")  # colour by dT/deta (arbitrary)

    # Quadrant guides
    ax.axvline(0, linewidth=1)
    ax.axhline(0, linewidth=1)

    ax.set_xlabel(r"$\partial T / \partial \eta$")
    ax.set_ylabel(r"$\partial T / \partial \phi$")
    ax.set_title("Slope-vs-slope scatter (want all points in 3rd quadrant)")
    fig.colorbar(sc, ax=ax, label=r"$\partial T / \partial \eta$")

    # Annotate shares
    ax.text(
        0.02, 0.98,
        f"Share in 3rd quadrant: {share_third:.3f}\n"
        f"Share with dT/dη < 0: {share_x_neg:.3f}\n"
        f"Share with dT/dφ < 0: {share_y_neg:.3f}\n"
        f"N points: {len(x)}",
        transform=ax.transAxes,
        va="top"
    )

    plt.tight_layout()
    plt.show()

    # Return arrays too, in case you want to inspect extremes
    return x, y


    
def calc_mesh(A=1.0):
    # -----------------------------
    # Grid + plots
    # -----------------------------
    p = Params(A=1.0, Tmax=200.0, n_scan=4000)   # <-- set A here
    n = 10

    # Use interior grid to avoid phi/eta at exactly 0 or 1
    phis = np.linspace(0.02, 0.98, n)
    etas = np.linspace(0.02, 0.98, n)

    PHI, ETA = np.meshgrid(phis, etas, indexing="xy")
    Tsol = np.full_like(PHI, np.nan, dtype=float)

    for i in range(PHI.shape[0]):
        for j in range(PHI.shape[1]):
            Tsol[i, j] = solve_T(PHI[i, j], ETA[i, j], p)

    return PHI, ETA, Tsol, phis, etas



def plot_mesh(A=1.0):
    PHI, ETA, Tsol, phis, etas = calc_mesh(A)  # <-- set A here
    
    # 3D surface: x=phi, y=T, z=eta (T on y-axis; phi & eta are x/z)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(PHI, ETA, Tsol, rstride=1, cstride=1, linewidth=0.3, antialiased=True)


    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\eta$")
    ax.set_zlabel(r"$T$")
    ax.set_title(rf"Solution surface for $T(\phi,\eta)$   (A = {A})")
    plt.tight_layout()
    plt.show()

    # 2D heatmap: eta vs phi
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    im = ax2.imshow(
        Tsol,
        origin="lower",
        aspect="auto",
        extent=[phis.min(), phis.max(), etas.min(), etas.max()],
    )
    ax2.set_xlabel(r"$\phi$")
    ax2.set_ylabel(r"$\eta$")
    ax2.set_title(rf"Heatmap of $T(\phi,\eta)$   (A = {A:g})")
    fig2.colorbar(im, ax=ax2, label=r"$T$")
    plt.tight_layout()
    plt.show()
    a=0


main()