"""Compute the three cubic coefficients of C_abgd for BaTiO3.

Run from the repository root with:
    python3 "python /tenseur_C/compute_C_tensor.py"
"""

from __future__ import annotations

import argparse

from constants import (
    MATERIAL_SETS,
    c_invariants_from_cartesian,
    short_range_cartesian_coefficients,
)


def fmt(value: float) -> str:
    return f"{value: .10f}"


def print_block(title: str, g_values: tuple[float, float, float]) -> None:
    g11, g12, g44 = g_values
    c1, c2, c3 = c_invariants_from_cartesian(g11, g12, g44)
    print(title)
    print("G11 =", fmt(g11))
    print("G12 =", fmt(g12))
    print("G44 =", fmt(g44))
    print("c1  =", fmt(c1))
    print("c2  =", fmt(c2))
    print("c3  =", fmt(c3))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute C tensor invariants.")
    parser.add_argument("--material", choices=sorted(MATERIAL_SETS), default="nishimatsu_bto")
    args = parser.parse_args()

    material = MATERIAL_SETS[args.material]
    print(f"Source: {args.material} short-range tensor")
    print("Short-range j_i:", material["short_range_j"])
    print()

    sr = short_range_cartesian_coefficients(args.material)

    print_block("Pure short-range tensor C:", sr)


if __name__ == "__main__":
    main()
