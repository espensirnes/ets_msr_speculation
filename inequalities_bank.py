import numpy as np
import matplotlib.pyplot as plt

# Define RHS function safely
def rhs_T(phi, eta):
    if phi <= 0 or phi >= 1 or eta <= 0 or eta >= 1:
        return np.nan
    num = (eta/phi) * (1 - phi) * (np.log(1 - phi) / np.log(1 - eta))
    den = np.log((1 - eta) * (1 - phi))
    if not np.isfinite(num) or not np.isfinite(den) or num <= 0 or den == 0:
        return np.nan
    return np.log(num) / den

# Create grid



a = np.linspace(0.0001, 0.9999, 50)
plt.plot(a, np.exp(a)-1)
plt.plot(a, a)
#plt.plot(a, 1-np.exp(-a))
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.ylim(0, 1.8)

plt.show()
n = 50
phis = np.linspace(0.02, 0.98, n)
etas = np.linspace(0.02, 0.98, n)
PHI, ETA = np.meshgrid(phis, etas, indexing="xy")

# Evaluate RHS on grid
vec_rhs = np.vectorize(rhs_T, otypes=[float])
T_rhs = vec_rhs(PHI, ETA)

# Plot 3D surface: x=phi, y=T_rhs (vertical), z=eta
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(PHI, ETA, T_rhs, rstride=1, cstride=1, linewidth=0.3, antialiased=True)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\eta$")
ax.set_zlabel(r"RHS bound on $T$")
ax.set_title(r"Surface of RHS bound for $T(\phi,\eta)$")

plt.tight_layout()
plt.show()
a=0
