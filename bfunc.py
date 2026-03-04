import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

def main():
	params = dict(
		phi=0.15,
		b0=5.0,
		theta=2.0,
		r=0.05,
		eta=0.2,
		DeltaT=0,
		ebar=1.0,
		alpha=0.1,
		S=5,
	)

	out = plot_discrete_vs_continuous(T=30, dense=2000, **params)
	print("All real roots:", out["roots"])
	print("Smallest nonnegative root:", out["smallest_nonneg_root"])



def banking(t, b0, phi, alpha, S, e_bar, theta, eta, r, DeltaT):
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


def b_continuous(t, phi, b0, theta, r, eta, DeltaT, ebar, alpha, S):
	"""
	Banking-matching continuous solution.
	"""
	import numpy as np

	a = 1.0 - phi
	lam = -np.log(a)
	kappa = lam / phi

	A = (1.0 / theta) * (r / (1.0 - eta))
	C = A * DeltaT - (ebar - alpha * S)

	t = np.asarray(t, dtype=float)
	e = np.exp(-lam * t)

	return (
		e * b0
		+ kappa * A * (t / lam + (e - 1.0) / (lam ** 2))
		+ kappa * C * ((1.0 - e) / lam)
	)



def solve_b_zero_times(phi, b0, theta, r, eta, DeltaT, ebar, alpha, S):
	"""
	Real roots of b(t)=0 for the banking-matching continuous b(t),
	using Lambert W branches 0 and -1.
	"""
	import numpy as np
	import mpmath as mp

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




def plot_discrete_vs_continuous(
	T,
	phi, b0, theta, r, eta, DeltaT, ebar, alpha, S,
	dense=2000
):
	# discrete
	t_int = np.arange(0, T+1)
	b_bank = banking(t_int, b0, phi, alpha, S, ebar, theta, eta, r, 0)
	# continuous curve
	b = b_continuous(t_int, phi, b0, theta, r, eta, DeltaT, ebar, alpha, S)

	# solve for b(t)=0
	root = solve_b_zero_times(phi, b0, theta, r, eta, DeltaT, ebar, alpha, S)
	# plot
	plt.figure()
	plt.plot(t_int, b, label="Continuous b(t)")
	plt.plot(t_int, b_bank, "o", label="Banking b(t)", color="green")

	if root is not None and 0.0 <= root <= T:
		plt.axvline(root, linestyle="--", label=f"Smallest root t* = {root:.4g}")

	plt.xlabel("t")
	plt.ylabel("b / x")
	plt.title("Discrete vs continuous analogue (λ = -log(1-φ))")
	plt.legend()
	plt.tight_layout()
	print(f"b cont:{b}")
	print(f"b bank:{b_bank}")
	print(f"t for all:{t_int}")
	plt.savefig("output/discrete_vs_continuous.png", dpi=300)
	plt.show()



	return {"roots": roots, "smallest_nonneg_root": t_star}


main()