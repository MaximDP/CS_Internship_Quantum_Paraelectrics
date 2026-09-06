"""Symbolic diagonalization helper for the cubic SCHA kernel.

Matrix studied:

    A = [[Cx,        B*kx*ky, B*kx*kz],
         [B*kx*ky,  Cy,      B*ky*kz],
         [B*kx*kz,  B*ky*kz, Cz]]

The fully general eigenvalues are the roots of a cubic and are usually not the
most readable object to paste in the manuscript.  This script therefore prints:

1. the matrix A;
2. the characteristic polynomial in invariant form;
3. the formal symbolic eigenvalues;
4. simple high-symmetry direction checks.
"""

from __future__ import annotations

import argparse

import sympy as sp


def print_block(title: str, expr) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    sp.pprint(expr, use_unicode=True)


def print_latex(title: str, expr) -> None:
    print("\n" + "-" * 80)
    print(title + " (LaTeX)")
    print("-" * 80)
    print(sp.latex(expr))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Symbolic diagonalization helper for the 3x3 SCHA kernel."
    )
    parser.add_argument(
        "--full-eigenvalues",
        action="store_true",
        help="Also print the full cubic-form symbolic eigenvalues. This is very verbose.",
    )
    args = parser.parse_args()

    kx, ky, kz, B = sp.symbols("kx ky kz B", real=True)
    Cx, Cy, Cz = sp.symbols("C_x C_y C_z", real=True)
    lam = sp.symbols("lambda", real=True)

    A = sp.Matrix(
        [
            [Cx, B * kx * ky, B * kx * kz],
            [B * kx * ky, Cy, B * ky * kz],
            [B * kx * kz, B * ky * kz, Cz],
        ]
    )

    print_block("Matrix A", A)
    print_latex("Matrix A", A)

    charpoly = sp.factor(A.charpoly(lam).as_expr())
    print_block("Characteristic polynomial det(lambda I - A)", charpoly)
    print_latex("Characteristic polynomial det(lambda I - A)", charpoly)

    I1 = sp.factor(sp.trace(A))
    I2 = sp.factor(
        Cx * Cy
        + Cx * Cz
        + Cy * Cz
        - B**2 * (kx**2 * ky**2 + kx**2 * kz**2 + ky**2 * kz**2)
    )
    I3 = sp.factor(A.det())
    invariant_poly = sp.factor(lam**3 - I1 * lam**2 + I2 * lam - I3)

    print_block("Invariant coefficients: I1 = tr(A)", I1)
    print_block("Invariant coefficients: I2 = sum principal minors", I2)
    print_block("Invariant coefficients: I3 = det(A)", I3)
    print_latex("I1", I1)
    print_latex("I2", I2)
    print_latex("I3", I3)
    print_latex("lambda^3 - I1 lambda^2 + I2 lambda - I3 = 0", invariant_poly)

    if args.full_eigenvalues:
        print_block("Formal symbolic eigenvalues", A.eigenvals())
    else:
        print(
            "\nFormal symbolic eigenvalues are the three roots of the cubic above. "
            "Run with --full-eigenvalues to print the raw cubic formula."
        )

    k = sp.symbols("k", real=True)

    A_100 = sp.simplify(A.subs({kx: k, ky: 0, kz: 0}))
    print_block("[100] direction matrix", A_100)
    print_block("[100] eigenvalues", A_100.eigenvals())
    print_latex("[100] eigenvalues", list(A_100.eigenvals().keys()))

    A_110 = sp.simplify(A.subs({kx: k / sp.sqrt(2), ky: k / sp.sqrt(2), kz: 0}))
    print_block("[110] direction matrix", A_110)
    print_block("[110] eigenvalues", A_110.eigenvals())
    print_latex("[110] eigenvalues", list(A_110.eigenvals().keys()))

    A_111 = sp.simplify(
        A.subs({kx: k / sp.sqrt(3), ky: k / sp.sqrt(3), kz: k / sp.sqrt(3)})
    )
    print_block("[111] direction matrix", A_111)
    print_block("[111] characteristic polynomial", sp.factor(A_111.charpoly(lam).as_expr()))
    print_latex("[111] characteristic polynomial", sp.factor(A_111.charpoly(lam).as_expr()))

    C = sp.symbols("C", real=True)
    A_111_iso = A_111.subs({Cx: C, Cy: C, Cz: C})
    print_block("[111], Cx=Cy=Cz=C eigenvalues", A_111_iso.eigenvals())
    print_latex("[111], Cx=Cy=Cz=C eigenvalues", list(A_111_iso.eigenvals().keys()))


if __name__ == "__main__":
    main()
