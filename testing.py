import numpy as np
import matplotlib.pyplot as plt


def main():

    T = 30
    t = np.arange(1, T + 1, dtype=float)
    # ----------------------------
    # Parameters (set arbitrarily)
    # ----------------------------
    class Parameters:
        def __init__(self):
            self.r = 0.05
            self.eta = 1e-10         # must satisfy eta != 0 and eta != 1 for these formulas
            self.e_bar = 10.0
            self.alphaS = 2.0       # this is αS (allowances issued per period)
            self.phi = 0.1          # must satisfy phi != 0
            self.b0 = 40
            self.T = T
            self.theta = p(T, self)/self.e_bar  # making sure emission is zero at the terminal date (t=T) by setting theta = P(T)/e_bar

    params = Parameters()
    # ----------------------------
    # Time grid                 
    # ----------------------------


    plot(t, params)
    a=0



def P(t, pa):
    Pt = np.exp(p(t, pa))
    return Pt
    
def p(t, pa):
    pt = (pa.r / pa.eta) * ((1 - pa.eta) ** (-t) - 1)
    return pt

def e(t, pa):
    bt = b(t, pa)
    et = pa.e_bar - (pa.r / (pa.theta * pa.eta)) * ((1 - pa.eta) ** (-t) - 1)
    return et

def b(t, pa):
    D = pa.phi + pa.eta - pa.phi * pa.eta
    bt = (
        (1-pa.phi) ** t * pa.b0+
        (pa.alphaS - pa.e_bar - pa.r / (pa.theta * pa.eta)) * (1 - (1 - pa.phi) ** t) / pa.phi
        + (pa.r / (pa.theta * pa.eta)) * (((1 - pa.eta) ** (-t) - (1 - pa.phi) ** t) / D)
    )
    bt = np.maximum(bt, 0)  # ensure non-negativity
    return bt

def pi(t, pa):
    pit = 1-np.exp(-pa.eta * p(t, pa))
    return pit


def plot(t, params):

    # ----------------------------
    # Plot (subplots, same figure)
    # ----------------------------
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(t, P(t, params))
    axes[0].set_ylabel(r"$P_t$")
    axes[0].set_title(r"$P_t = \exp(\frac{r}{\eta}\left((1-\eta)^{-t}-1\right))$")
    axes[0].grid(True)
    axes[0].set_ylim(0, 30)

    axes[1].plot(t, e(t, params))
    axes[1].set_ylabel(r"$e_t$")
    axes[1].set_title(r"$e_t = \bar e - \frac{r}{\theta\eta}\left((1-\eta)^{-t}-1\right)$")
    axes[1].grid(True)

    axes[2].plot(t, b(t, params))
    axes[2].set_ylabel(r"$b_{t+1}$")
    axes[2].set_title(
        r"$b_{t+1} = \left(\alpha S-\bar e-\frac{r}{\theta\eta}\right)\frac{1-(1-\phi)^t}{\phi}"
        r" + \frac{r}{\theta\eta}\frac{(1-\eta)^{-t}-(1-\phi)^t}{\phi+\eta-\phi\eta}$"
    )
    axes[2].set_xlabel(r"$t$")
    axes[2].grid(True)


    axes[3].plot(t, pi(t, params))
    axes[3].set_ylabel(r"$\pi(P_t)$")
    axes[3].set_title(r"$\pi(P_t) = P_t^{-\eta}$")
    axes[3].grid(True)


    plt.tight_layout()
    plt.show()

main()