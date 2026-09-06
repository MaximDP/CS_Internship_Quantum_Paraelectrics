"""Cubic lattice constants of perovskite materials, for use with lattice_sum.py."""

BOHR_TO_ANGSTROM = 0.529177210903

MATERIALS = {
    "zvr_bto": {
        # LDA-relaxed cubic lattice constant, Zhong & Vanderbilt, PRB 52, 6301 (1995), Fig. 3 caption.
        "a0_bohr": 7.46,
        # Mode effective charge and optical dielectric constant, Table II of the same reference.
        "Z_star": 9.956,
        "epsilon_inf": 5.24,
    },
    "nishimatsu_bto": {
        # Nishimatsu et al. WC-GGA/FERAM BaTiO3 parametrization.
        "a0_bohr": 3.98596 / BOHR_TO_ANGSTROM,
        "Z_star": 10.33,
        "epsilon_inf": 6.8691464565,
    },
    "nishimatsu_sto": {
        # Nishimatsu et al., JPSJ 85, 114714 (2016), Table I.
        "a0_bohr": 3.901 / BOHR_TO_ANGSTROM,
        "Z_star": 9.28,
        "epsilon_inf": 6.46,
    },
}

MATERIALS["BaTiO3"] = MATERIALS["nishimatsu_bto"]
MATERIALS["SrTiO3"] = MATERIALS["nishimatsu_sto"]
