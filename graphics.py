import numpy as np
import matplotlib.pyplot as plt



def main():
	# =========================
	# Resetting some parameters
	# =========================
	p = params
	T_bank_termination = 35
	denominator = 0.01
	
	p['alpha'] = 1/p["T_zero_emissions"]
	p['s0'] = (p["omega"] * p["e0_bar"]+(hotelling_adj(p) - denominator))/p['alpha']
	
	phi = p['alpha']*p['s0'] - p["omega"] * p["e0_bar"]
	denominator = hotelling_adj(p) - phi
	params['b0'] = denominator * T_bank_termination**2/2


	# =========================
	# TIME GRID (beyond T_price)
	# =========================
	T_START = 2005
	T = compute_T_min(params)
	t = np.linspace(0,params["T_zero_emissions"], 200)  # 30% beyond T_price



	# =========================
	# COMPUTE SERIES
	# =========================
	b_t = banking(t, T, params)
	b_t_actual = b_t*1  # Bank cannot be negative
	nz = np.nonzero(b_t_actual <= 1e-2)[0]
	if len(nz):
		b_t_actual[nz[0]:] = 0
	e_t = emissions(t, T, params)
	p_t = prices(t, T, params)
	s_t = release(params, t)
	p_T = prices(T, T, params)

	plot(t + T_START, p_t, b_t_actual, b_t, e_t, s_t, p_T, T+T_START)
	a=0


# =========================
# PARAMETERS
# =========================
params = {
	"theta": 0.9,
	"r": 0.05,
	"eta": 0.1,
	"omega": 0.00,
	"e0_bar": 100.0,#Sets the price level, as the price at t=0 is kappa + theta*e0_bar
	"kappa": 1.0,
	"T_zero_emissions": 45
}


# =========================
# FUNCTIONS
# =========================
def hotelling_adj(p):
	h =  (1 / p["theta"]) * (p["r"] / (1 - p["eta"]))
	return h


def e_bar(p, t):
	return p["e0_bar"] * (1 - p["omega"] * t)


def release(p, t):
	return p["s0"] * (1 - p["alpha"] * t)


def banking(t, T, p):
	A = hotelling_adj(p)
	slope = A - p["s0"] * p["alpha"] + p["omega"] * p["e0_bar"]
	b = p["b0"] - slope * t * (T - t / 2)
	return b

	import math

import math

def compute_T_min(params):

	b0 = params["b0"]
	s0 = params["s0"]
	alpha = params["alpha"]
	omega = params["omega"]
	e0_bar = params["e0_bar"]

	A = hotelling_adj(params)
	denominator = A - s0 * alpha + omega * e0_bar
	numerator = 2 * b0
	T_min = (numerator / denominator)**0.5

	return T_min


def emissions(t, T, p):
	A = hotelling_adj(p)
	st = release(p, t)
	e_t = st + (A + p["omega"] * p["e0_bar"] - p["alpha"] * p["s0"]) * (T-0.5 - t)
	if type(t)==np.ndarray:
		bank_ends = np.nonzero(t>= T)[0][0]
		e_t[bank_ends:]= release(p, t)[bank_ends:]
	return e_t


def prices(t, T, p):
	ebart = e_bar(p,t)
	return p["kappa"] + p["theta"] * (ebart - emissions(t, T, p)) 


def plot(t, p, b_actual, b_latent, e, s_t, pT,root):
	fig, ax = plt.subplots(figsize=(9, 6))
	ax.plot(t, p) 
	ax.plot(t, pT*np.ones_like(t), color="black", linewidth=0.1)
	if root is not None and 0.0 <= root <= max(t):
		ax.axvline(root, linestyle="--", label=f"Smallest root t* = {root:.4g}")
	ax.set_xlabel("Time")
	ax.set_ylabel("Price (EUR/t)")

	ax2 = ax.twinx()    
	ax2.plot(t, s_t, color="red", linestyle="--")
	ax2.plot(t, b_latent, color="grey", label="Banking (Latent)")
	ax2.plot(t, b_actual, color="orange", label="Banking (Actual)") 
	ax2.plot(t, e, color="green") 
	
	ax2.set_ylabel("Emission (banking, emission, release)")

	for spine in ['top']:
		ax.spines[spine].set_visible(False)
		ax2.spines[spine].set_visible(False)

	fig.legend(["Price","$p_T$", "Bank termination", "Release", "Banking latent", "Banking", "Emission"],  frameon=False, 
			loc="upper right", bbox_to_anchor=(0.6, 1.0))
	#plt.ylim(-0.5, 1.1)
	plt.savefig("output/inequalities_bank.png", dpi=300)
	plt.show()
	a=0
	 

main()

