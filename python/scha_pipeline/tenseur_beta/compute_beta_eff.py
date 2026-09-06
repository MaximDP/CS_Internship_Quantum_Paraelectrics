"""Compute the cubic coefficients of beta_eff for a supported material.

Run from the repository root with:
    python3 "python /tenseur_beta/compute_beta_eff.py"
"""

from __future__ import annotations

import argparse

import numpy as np

from constants import (
    MATERIAL_SETS,
    bare_beta_invariants,
    cubic_invariants_from_tensor,
    elastic_matrix,
    electrostrictive_tensor_vanderbilt,
    strain_quartic_tensor,
)


def fmt(value: float) -> str:
    return f"{value: .10f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute beta_eff cubic invariants.")
    parser.add_argument("--material", choices=sorted(MATERIAL_SETS), default="nishimatsu_bto")
    args = parser.parse_args()

    material = MATERIAL_SETS[args.material]
    e_matrix = elastic_matrix(material.elastic)
    e_inv = np.linalg.inv(e_matrix)

    b_vanderbilt = electrostrictive_tensor_vanderbilt(material.electrostrictive)

    beta_bare = np.array(bare_beta_invariants(material.onsite))

    # Vanderbilt writes
    #   E_el  = 1/2 eta_l C_lm eta_m
    #   E_int = 1/2 eta_l B^V_l_ab u_a u_b.
    # Eliminating eta gives Delta E_4 = -1/8 (B^V C^-1 B^V)_abcd u_a u_b u_c u_d.
    # Since untilted.tex writes E_4 = 1/4 beta_abcd u_a u_b u_c u_d,
    # Delta beta = -1/2 sym(B^V C^-1 B^V).
    t_vanderbilt = strain_quartic_tensor(b_vanderbilt, e_matrix)
    t_vanderbilt_invariants = np.array(cubic_invariants_from_tensor(t_vanderbilt))
    beta_delta = -0.5 * t_vanderbilt_invariants
    beta_eff = beta_bare + beta_delta

    print(f"Source values: {args.material}")
    print("Elastic constants:", material.elastic)
    print("Elastic-mode constants:", material.electrostrictive)
    print("Bare on-site quartic constants:", material.onsite)
    print()

    print("Elastic matrix E:")
    print(e_matrix)
    print()
    print("Elastic inverse E^-1:")
    print(e_inv)
    print()

    print("Bare beta invariants in untilted.tex convention:")
    print("b1_0 =", fmt(beta_bare[0]))
    print("b2_0 =", fmt(beta_bare[1]))
    print()

    print("Symmetrized invariants of T = B^V E^-1 B^V:")
    print("T_b1 =", fmt(t_vanderbilt_invariants[0]))
    print("T_b2 =", fmt(t_vanderbilt_invariants[1]))
    print()

    print("Delta beta = -1/2 T from Vanderbilt prefactors:")
    print("Delta b1 =", fmt(beta_delta[0]))
    print("Delta b2 =", fmt(beta_delta[1]))
    print("b1_eff =", fmt(beta_eff[0]))
    print("b2_eff =", fmt(beta_eff[1]))


if __name__ == "__main__":
    main()
