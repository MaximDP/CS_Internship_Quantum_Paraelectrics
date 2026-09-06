"""Constants for the short-range gradient tensor C of cubic perovskites.

Sources:
- Zhong, Vanderbilt and Rabe, Phys. Rev. B 52, 6301 (1995), Table II.
- Kornev note Continuum_LimitV0.pdf, Secs. 5.5--5.7.
"""

from __future__ import annotations

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903


def ev_per_angstrom_power_to_hartree_per_bohr_power(value: float, power: int) -> float:
    return value / HARTREE_TO_EV * (BOHR_TO_ANGSTROM**power)


# ZVR-1995 BaTiO3 parameters. Energies are Hartree, lengths are bohr.
SHORT_RANGE_J = {
    "j1": -0.02734,
    "j2": 0.04020,
    "j3": 0.00927,
    "j4": -0.00815,
    "j5": 0.00580,
    "j6": 0.00370,
    "j7": 0.00185,
}

DIPOLE = {
    "Z_star": 9.956,
    "epsilon_inf": 5.24,
    "a0": 7.46,
}

NISHIMATSU_BTO_SHORT_RANGE_EV_A2 = {
    "j1": -2.0840250430,
    "j2": -1.1290411983,
    "j3": +0.6894579816,
    "j4": -0.6113408159,
    "j5": 0.0,
    "j6": +0.2768966803,
    "j7": 0.0,
}

NISHIMATSU_STO_SHORT_RANGE_EV_A2 = {
    # Nishimatsu et al., JPSJ 85, 114714 (2016), Table I.
    "j1": -2.012,
    "j2": -1.815,
    "j3": +0.590,
    "j4": -0.567,
    "j5": 0.0,
    "j6": +0.238,
    "j7": 0.0,
}

MATERIAL_SETS = {
    "zvr_bto": {
        "short_range_j": SHORT_RANGE_J,
        "dipole": DIPOLE,
    },
    "nishimatsu_bto": {
        "short_range_j": {
            key: ev_per_angstrom_power_to_hartree_per_bohr_power(value, 2)
            for key, value in NISHIMATSU_BTO_SHORT_RANGE_EV_A2.items()
        },
        "dipole": {
            "Z_star": 10.33,
            "epsilon_inf": 6.8691464565,
            "a0": 3.98596 / BOHR_TO_ANGSTROM,
        },
    },
    "nishimatsu_sto": {
        "short_range_j": {
            key: ev_per_angstrom_power_to_hartree_per_bohr_power(value, 2)
            for key, value in NISHIMATSU_STO_SHORT_RANGE_EV_A2.items()
        },
        "dipole": {
            "Z_star": 9.28,
            "epsilon_inf": 6.46,
            "a0": 3.901 / BOHR_TO_ANGSTROM,
        },
    },
}


def short_range_cartesian_coefficients(material: str = "zvr_bto") -> tuple[float, float, float]:
    """Return (G11, G12, G44) for the short-range part only.

    The coefficients are defined by

      K_xx = G11 k_x^2 + G44 (k_y^2 + k_z^2),
      K_xy = G12 k_x k_y.

    Directly expanding the 26-neighbour FERAM/ZVR matrix gives:

      G11^SR = -a0^2 (j2 + 4 j3 + 4 j6)
      G12^SR = -4 a0^2 (j5 + 2 j7)
      G44^SR = -a0^2 (j1 + 2 j3 + 2 j4 + 4 j6)

    Earlier versions interchanged the last two expressions and returned half
    of the off-diagonal coefficient.  The definitions above are also obtained
    independently from C_abgd = -1/2 sum_R J_ab(R) R_g R_d.
    """
    j = MATERIAL_SETS[material]["short_range_j"]
    a0 = MATERIAL_SETS[material]["dipole"]["a0"]
    a02 = a0 * a0
    g11 = -a02 * (j["j2"] + 4.0 * j["j3"] + 4.0 * j["j6"])
    g12 = -4.0 * a02 * (j["j5"] + 2.0 * j["j7"])
    g44 = -a02 * (j["j1"] + 2.0 * j["j3"] + 2.0 * j["j4"] + 4.0 * j["j6"])
    return g11, g12, g44


def c_invariants_from_cartesian(g11: float, g12: float, g44: float) -> tuple[float, float, float]:
    """Map Kornev (G11,G12,G44) to untilted.tex (c1,c2,c3).

    untilted.tex uses
      C_abgd k_g k_d = c2 k^2 delta_ab + 2 c3 k_a k_b
                       + c1 delta_ab k_a^2  (no sum on a).

    Comparing with K_xx=G11 k_x^2+G44(k_y^2+k_z^2) and
    K_xy=G12 k_x k_y gives:
      c2 = G44,  2 c3 = G12,  c1 = G11-G12-G44.
    """
    c1 = g11 - g12 - g44
    c2 = g44
    c3 = 0.5 * g12
    return c1, c2, c3
