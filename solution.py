
from logging import root

from matplotlib import pyplot as plt
import numpy as np
from sympy import root
import mpmath as mp


P0 = 0
R = 0.05
E_BAR = 8
ALPHA = 0.05
THETA = 1
PHI = 0.0001
S = 20
ETA = 0.2
T = 40
B0 = 20
PRESTART = 20

def main():
	#plot_price()
	plot_bank(E_BAR, ALPHA, S, THETA, ETA, R)

def plot_bank(e_bar, alpha, S, theta, eta, r):

	DeltaT = delta_T_sol(e_bar, S, alpha, PHI, B0, eta, theta, r)
	if DeltaT is None:	
		DeltaT = 0
	
	total_p_increase =(e_bar - alpha*S)*theta
	pT = P0 + total_p_increase
	T_max = total_p_increase*(1-eta)/r - DeltaT
	T_max_int = int(T_max)
	t = np.arange(0, T_max_int+PRESTART+10)-PRESTART

	p = price_function(pT, P0, R, eta, t, T_max)
	e = emission_function(P0, p, e_bar, theta)
	b = banking(t, B0, PHI, ALPHA, S, e_bar, theta, eta, R, DeltaT)		

	b = b * (t<=T_max)  # Bank is zero after T_max

	plot(t, p, b, e, alpha*S, pT, root=T_max)
	a=0
	
def plot(t, p, b, e, alpha_S, pT,root=None):
	fig, ax = plt.subplots(figsize=(9, 6))
	ax.plot(t, p) 
	ax.plot(t, pT*np.ones_like(t), color="black", linewidth=0.1)
	if root is not None and 0.0 <= root <= max(t):
		ax.axvline(root, linestyle="--", label=f"Smallest root t* = {root:.4g}")
	ax.set_xlabel("Time")
	ax.set_ylabel("Price (EUR/t)")

	ax2 = ax.twinx()    
	ax2.plot(t, b, color="orange") 
	ax2.plot(t, e, color="green") 
	ax2.plot(t, alpha_S*np.ones_like(t), color="red", linestyle="--")
	ax2.set_ylabel("Emission (banking, emission, issuance)")

	for spine in ['top']:
		ax.spines[spine].set_visible(False)
		ax2.spines[spine].set_visible(False)

	fig.legend(["Price","$p_T$", "Bank termination", "Banking", "Emissions", "Issuance"],  frameon=False, 
			loc="upper right", bbox_to_anchor=(0.4, 0.8))
	#plt.ylim(-0.5, 1.1)
	plt.savefig("output/inequalities_bank.png", dpi=300)
	plt.show()
	a=0




def banking(t, b0, phi, alpha, S, e_bar, theta, eta, r, DeltaT):
	"""
	Discrete-time banking-matching solution.
	b_{t} = (1-phi)*b_{t-1} + alpha*S - e_t
	with e_t = e_bar - (1/theta)*(r/(1-eta))*(t)
	"""
	t = t*(t>0)
	b = []
	e_t, deposit = 0, 0
	for ti in t:
		b0_phi = (1 - phi)**ti * b0 
		b_innov = 0
		for tau in range(1, ti+1):
			e_t = e_bar - (1/theta)*(r/(1-eta))*(tau + DeltaT)
			deposit = (1 - phi)**(ti - tau) * (alpha * S - e_t) * (ti > 0)
			b_innov += deposit
		delta_b = b0_phi + b_innov
		b.append(delta_b)	
		#print(f"t={ti}, b0_phi={b0_phi:.4f}, delta_b={delta_b:.4f}, b={b[-1]:.4f}, e_t={e_t:.4f}, deposit={deposit:.4f}, b_innov={b_innov:.4f}")
	b = np.array(b)
	return b

def banking_comprehension(t, b0, phi, alpha, S, e_bar, theta, eta, r, DeltaT):
	"""
	Discrete-time banking-matching solution.
	b_{t} = (1-phi)*b_{t-1} + alpha*S - e_t
	with e_t = e_bar - (1/theta)*(r/(1-eta))*(t)
	"""
	t = t*(t>0)
	b = [
		(1 - phi)**ti * b0 
		+ sum(
			 (1 - phi)**(ti - tau) * (alpha * S -(
				 e_bar - (1/theta)*(r/(1-eta))*(tau + DeltaT)
			 ) )*(ti>0)
			 for tau in range(1, ti+1)
			)
		 for ti in t
	]
	b = np.array(b)
	return b*(b>0)

def emission_function(p0,p, e_bar,theta):
	e = e_bar - (1/theta)*(p-p0)
	return e

def price_function(pT, p0, r, eta, t, T):
	p = pT-(T-t)*r/(1-eta)
	# Price equals the non intertemporal trading price when the bank is empty:
	p = p * (t<=T) + pT*(t>T)
	# Price equals zero before p0, because then by definition there is no abatement:
	p = p*(t>=0)+p0*(t<0)
	return p


def solve_DeltaT_for_terminal(t, phi, b0, theta, r, eta, ebar, alpha, S):
	import numpy as np

	lam = -np.log(1.0 - phi)
	kappa = lam / phi
	A = (1.0 / theta) * (r / (1.0 - eta))

	E = np.exp(-lam * t)

	num = (
		E * b0
		+ kappa * A * (t / lam + (E - 1.0) / (lam ** 2))
		+ kappa * (alpha * S - ebar) * ((1.0 - E) / lam)
	)
	den = kappa * A * ((1.0 - E) / lam)

	return -num / den

def solve_DeltaT_match_roof(T_max, phi, b0, theta, r, eta, ebar, alpha, S):

	lam = -np.log(1.0 - phi)
	A = (1.0 / theta) * (r / (1.0 - eta))
	B = alpha * S - ebar

	lam = mp.mpf(lam)
	A = mp.mpf(A)
	B = mp.mpf(B)
	T = mp.mpf(T_max)
	b0 = mp.mpf(b0)
	phi = mp.mpf(phi)

	m = mp.e**(lam * T) * (lam * T - 1 + lam * (B / A))
	d = -1 + lam * (B / A) - lam * phi * (b0 / A)

	z = m * mp.e**d

	sols = []
	for branch in (0, -1):
		W = mp.lambertw(z, branch)
		if mp.im(W) == 0:
			DT = (mp.log(m) - mp.log(W)) / lam
			sols.append(float(mp.re(DT)))

	return min(sols) if sols else None

def solve_b_zero_times_phi0(b0, A, C):
	a2 = A / 2.0
	a1 = C
	a0 = b0

	disc = a1**2 - 4*a2*a0
	if disc < 0:
		return []

	sqrt_disc = np.sqrt(disc)
	t1 = (-a1 + sqrt_disc) / (2*a2)
	t2 = (-a1 - sqrt_disc) / (2*a2)

	sols = sorted([t1, t2])
	unique = []
	for s in sols:
		if not unique or abs(s - unique[-1]) > 1e-8:
			unique.append(s)
	return min(unique) if unique else None

def solve_b_zero_times(phi, b0, theta, r, eta, DeltaT, ebar, alpha, S):
	"""
	Real roots of b(t)=0 for the banking-matching continuous b(t),
	using Lambert W branches 0 and -1.
	"""

	a = 1.0 - phi
	lam = -np.log(a)
	kappa = lam / phi

	A = (1.0 / theta) * (r / (1.0 - eta))
	C = A * DeltaT - (ebar - alpha * S)

	lam = mp.mpf(lam)
	A = mp.mpf(A)
	C = mp.mpf(C)
	b0 = mp.mpf(b0)
	kappa = mp.mpf(kappa)

	if phi == 0:
		return solve_b_zero_times_phi0(float(b0), float(A), float(C))
	
	K = b0 + kappa * A / (lam**2) - kappa * C / lam
	M = kappa * A / lam
	N = -kappa * A / (lam**2) + kappa * C / lam

	arg = -(lam / M) * K * mp.e ** (lam * N / M)

	sols = []
	for branch in (0, -1):
		w = mp.lambertw(arg, branch)
		if mp.im(w) == 0:
			t = -N / M + (1 / lam) * w
			sols.append(float(t))

	if len(sols) == 0:
		return None
	return min(sols)

import numpy as np


def t_terminal(phi, b0, theta, r, eta, ebar, alpha, S, DeltaT=0):
	"""
	Returns t solving b'(t)=0 
	"""

	lam = -np.log(1.0 - phi)
	A = (1.0 / theta) * (r / (1.0 - eta))
	B = alpha * S - ebar
	C = B + A * DeltaT  # shifted constant term

	numerator = lam**2 * b0 + A - lam * C
	denominator = A

	value = numerator / denominator

	t = (1.0 / lam) * np.log(value)

	return t

def delta_T_sol(e_bar, S, alpha, phi, b0, eta, theta, r):
	lam = -np.log(1.0 - phi)
	A = (1.0 / theta) * (r / (1.0 - eta))

	sols = []
	for branch in (0, -1):
		w = mp.lambertw(-np.exp(-1 - lam**2 * b0 / A), branch)
		if mp.im(w) == 0:
			dt = ((e_bar - alpha*S)/A
				+ lam*b0/A
				+ 1/lam
				+ (1/lam) * w)
			sols.append(float(dt))

	return min(sols) if sols else None

main()