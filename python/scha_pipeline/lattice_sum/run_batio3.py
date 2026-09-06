import argparse

from lattice_sum import expansion_coefficients
from reciprocal_sum import lrp_quadratic_invariants, g0_quadratic_correction
from materials import MATERIALS

parser = argparse.ArgumentParser(description="Dipolar lattice-sum small-k coefficients.")
parser.add_argument("--material", choices=sorted(MATERIALS), default="nishimatsu_bto")
parser.add_argument("--tol", type=float, default=1e-4)
args = parser.parse_args()

m = MATERIALS[args.material]
a0, Z_star, eps_inf = m["a0_bohr"], m["Z_star"], m["epsilon_inf"]
eta = 2.0 / a0  # Ewald splitting parameter; the final (SR+LR+G0) result is eta-independent
tol = args.tol

sr = expansion_coefficients(a0, eta, tol=tol)
lr = lrp_quadratic_invariants(a0, eta, tol=tol)
g0 = g0_quadratic_correction(a0, eta)  # exact, no lattice sum -> no error term

invariants = ["k2_delta", "k_a k_b", "delta_k_a2"]
labels = {"k2_delta": "k^2 delta_ab", "k_a k_b": "k_a k_b", "delta_k_a2": "delta_ab k_a^2"}

geometric_total = {inv: sr[inv] + lr[inv] + g0[inv] for inv in invariants}
# g0 has no error (closed form); SR and LR errors add (triangle inequality).
geometric_err = {inv: sr[f"err_{inv}"] + lr[f"err_{inv}"] for inv in invariants}

print(f"material = {args.material}")
print(f"a0 = {a0} bohr, eta = {eta:.4f} bohr^-1, convergence tol = {tol}")
print(f"SR converged after {sr['n_shells']} shells, LR converged after {lr['n_shells']} shells")
print()
print(f"{'invariant':<16}{'SR':>14}{'LR (G!=0)':>14}{'G=0 corr.':>14}  {'total (geom.)':<24}")
for inv in invariants:
    total_str = f"{geometric_total[inv]:.4e} +/- {geometric_err[inv]:.1e}"
    print(f"{labels[inv]:<16}{sr[inv]:>14.4e}{lr[inv]:>14.4e}{g0[inv]:>14.4e}  {total_str:<24}")

# The k^0-order self-term -4*eta^3/(3*sqrt(pi))*delta_ab does not contribute at
# order k^2 (it is a constant), so it has no entry in the table above.

prefactor = Z_star**2 / eps_inf
physical_total = {inv: prefactor * geometric_total[inv] for inv in invariants}
physical_err = {inv: prefactor * geometric_err[inv] for inv in invariants}

print()
print(f"Physical prefactor Z*^2/eps_inf = {prefactor:.4f}")
print(f"{'invariant':<16}{'geometric (1/bohr)':<24}{'physical':<24}")
for inv in invariants:
    geom_str = f"{geometric_total[inv]:.4e} +/- {geometric_err[inv]:.1e}"
    phys_str = f"{physical_total[inv]:.4e} +/- {physical_err[inv]:.1e}"
    print(f"{labels[inv]:<16}{geom_str:<24}{phys_str:<24}")
