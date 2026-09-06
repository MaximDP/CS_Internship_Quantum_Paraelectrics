#!/usr/bin/env python3
"""One-shot projected-local quartic correction around the Mayer SCHA branch.

This auxiliary calculation evaluates the two-vertex quartic sunset with the
full quantum SCHA propagator. The momentum convolution is integrated with a
scrambled Sobol rule on the periodic Brillouin zone and the two Matsubara sums
are replaced by one Gauss-Legendre integral in imaginary time.

The calculation is deliberately a diagnostic, not the complete post-SCHA
theory documented in Mayer.tex:

* the quartic vertex uses the local homogeneous-strain projection already
  stored in b1_eff and b2_eff;
* contractions of the local sextic and octic tensors dress that quartic
  vertex;
* the distinct dynamic inhomogeneous-strain vertex and the remaining
  quartic-sextic/sextic-sextic/... skeletons are not evaluated;
* the lattice-matched pressure is the existing SCHA background pressure and
  is not refitted after adding the second-order free-energy correction.

All outputs remain isolated in the outputs/mayer directory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc


MODULE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = MODULE_DIR.parent
PARAELECTRIC_DIR = PIPELINE_DIR / "SCHA_paraelc"
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(PARAELECTRIC_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

import nishimatsu_harmonic as harmonic
import parameters as mayer
import run_mayer_bto as mayer_scan
import scha_paraelc as scha


DEFAULT_INPUT = MODULE_DIR.parent / "outputs" / "reference" / "mayer" / "mayer_bto_400_800K_n40.csv"
DEFAULT_OUTPUT_CSV = (
    MODULE_DIR.parent / "outputs" / "mayer" / "mayer_post_scha_quartic_one_shot_400_800K_n40.csv"
)
DEFAULT_OUTPUT_JSON = (
    MODULE_DIR.parent / "outputs" / "mayer" / "mayer_post_scha_quartic_one_shot_400_800K_n40.json"
)
DEFAULT_OUTPUT_PNG = (
    MODULE_DIR.parent / "outputs" / "mayer" / "mayer_post_scha_quartic_one_shot_400_800K_n40.png"
)


@dataclass(frozen=True)
class SunsetGrid:
    """Three harmonic eigensystems for Q1, Q2 and Q3=-Q1-Q2."""

    sample_power: int
    seed: int
    eigenvalues_1: np.ndarray
    eigenvectors_1: np.ndarray
    eigenvalues_2: np.ndarray
    eigenvectors_2: np.ndarray
    eigenvalues_3: np.ndarray
    eigenvectors_3: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.eigenvalues_1.shape[0])


def wrap_to_bz(wavevectors: np.ndarray, cutoff: float) -> np.ndarray:
    """Wrap vectors componentwise into [-cutoff, cutoff)."""

    period = 2.0 * cutoff
    return (np.asarray(wavevectors) + cutoff) % period - cutoff


def _batch_raw_harmonic_kernel(
    wavevectors: np.ndarray,
    coefficients: scha.SchaCoefficients,
    chunk_size: int = 256,
) -> np.ndarray:
    """Vectorized raw harmonic kernel for auxiliary QMC nodes."""

    neighbours = harmonic._short_range_neighbours(coefficients)
    ewald = harmonic._ewald_terms(coefficients)
    neighbour_vectors, neighbour_matrices = neighbours
    real_vectors, real_tensors, reciprocal_vectors, eta, self_term = ewald
    identity = np.eye(3)
    output = np.empty((wavevectors.shape[0], 3, 3), dtype=float)

    for start in range(0, wavevectors.shape[0], chunk_size):
        stop = min(start + chunk_size, wavevectors.shape[0])
        current = wavevectors[start:stop]
        short_range = np.broadcast_to(
            2.0 * coefficients.kappa2 * identity,
            (current.shape[0], 3, 3),
        ).copy()
        short_range += np.einsum(
            "nr,rij->nij",
            np.cos(current @ neighbour_vectors.T),
            neighbour_matrices,
            optimize=True,
        )

        dipolar = np.broadcast_to(
            self_term, (current.shape[0], 3, 3)
        ).copy()
        dipolar += np.einsum(
            "nr,rij->nij",
            np.cos(current @ real_vectors.T),
            real_tensors,
            optimize=True,
        )
        shifted = reciprocal_vectors[None, :, :] + current[:, None, :]
        shifted_norm2 = np.einsum(
            "nri,nri->nr", shifted, shifted, optimize=True
        )
        reciprocal_weights = np.zeros_like(shifted_norm2)
        nonzero = shifted_norm2 > 1.0e-24
        reciprocal_weights[nonzero] = (
            np.exp(-shifted_norm2[nonzero] / (4.0 * eta**2))
            / shifted_norm2[nonzero]
        )
        dipolar += (
            4.0
            * math.pi
            / coefficients.omega0
            * np.einsum(
                "nr,nri,nrj->nij",
                reciprocal_weights,
                shifted,
                shifted,
                optimize=True,
            )
        )
        dipolar *= coefficients.zstar**2 / coefficients.eps_inf
        output[start:stop] = short_range + dipolar
    return output


def harmonic_offsets_at(
    wavevectors: np.ndarray,
    coefficients: scha.SchaCoefficients,
) -> np.ndarray:
    """Return the periodic harmonic kernel with its cubic Gamma mass removed."""

    gamma = harmonic.raw_harmonic_kernel(np.zeros(3), coefficients)
    gamma_scalar = float(np.trace(gamma) / 3.0)
    offsets = _batch_raw_harmonic_kernel(wavevectors, coefficients)
    offsets -= gamma_scalar * np.eye(3)[None, :, :]
    return offsets


def build_sunset_grid(
    coefficients: scha.SchaCoefficients,
    sample_power: int,
    seed: int,
) -> SunsetGrid:
    """Build deterministic scrambled-Sobol momentum pairs and eigensystems."""

    if sample_power < 5:
        raise ValueError("sample_power must be at least 5")
    sampler = qmc.Sobol(d=6, scramble=True, seed=seed)
    unit_points = sampler.random_base2(sample_power)
    momenta = (2.0 * unit_points - 1.0) * coefficients.kmax
    q1 = momenta[:, :3]
    q2 = momenta[:, 3:]
    q3 = wrap_to_bz(-(q1 + q2), coefficients.kmax)

    offsets = [
        harmonic_offsets_at(points, coefficients)
        for points in (q1, q2, q3)
    ]
    eigenpairs = [np.linalg.eigh(value) for value in offsets]

    # Guard the vectorized auxiliary kernel against a convention drift.
    gamma = harmonic.raw_harmonic_kernel(np.zeros(3), coefficients)
    gamma_scalar = float(np.trace(gamma) / 3.0)
    for index in range(min(3, q1.shape[0])):
        reference = harmonic.raw_harmonic_kernel(q1[index], coefficients)
        reference -= gamma_scalar * np.eye(3)
        if not np.allclose(
            offsets[0][index], reference, rtol=2.0e-12, atol=2.0e-12
        ):
            raise RuntimeError("batch harmonic kernel disagrees with reference")

    return SunsetGrid(
        sample_power=sample_power,
        seed=seed,
        eigenvalues_1=eigenpairs[0][0],
        eigenvectors_1=eigenpairs[0][1],
        eigenvalues_2=eigenpairs[1][0],
        eigenvectors_2=eigenpairs[1][1],
        eigenvalues_3=eigenpairs[2][0],
        eigenvectors_3=eigenpairs[2][1],
    )


@lru_cache(maxsize=8)
def _rank4_tensor(coefficients: scha.SchaCoefficients) -> np.ndarray:
    return np.asarray(
        [
            scha.beta_eff_component(a, b, c, d, coefficients)
            for a in range(3)
            for b in range(3)
            for c in range(3)
            for d in range(3)
        ],
        dtype=float,
    ).reshape((3,) * 4)


@lru_cache(maxsize=8)
def _rank6_tensor(coefficients: scha.SchaCoefficients) -> np.ndarray:
    return np.asarray(
        [
            scha.gamma_component(indices, coefficients)
            for indices in np.ndindex((3,) * 6)
        ],
        dtype=float,
    ).reshape((3,) * 6)


@lru_cache(maxsize=8)
def _rank8_tensor(coefficients: scha.SchaCoefficients) -> np.ndarray:
    return np.asarray(
        [
            scha.rho_component(indices, coefficients)
            for indices in np.ndindex((3,) * 8)
        ],
        dtype=float,
    ).reshape((3,) * 8)


def effective_quartic_vertex(
    component_variance: float,
    coefficients: scha.SchaCoefficients,
) -> np.ndarray:
    """Return the Gaussian average of the fourth local-potential derivative."""

    if component_variance <= 0.0 or not math.isfinite(component_variance):
        raise ValueError("component_variance must be positive and finite")
    covariance = component_variance * np.eye(3)
    beta = _rank4_tensor(coefficients)
    gamma = _rank6_tensor(coefficients)
    rho = _rank8_tensor(coefficients)

    vertex = 6.0 * beta
    vertex += 60.0 * np.einsum(
        "abcdef,ef->abcd", gamma, covariance, optimize=True
    )
    vertex += 210.0 * (
        np.einsum(
            "abcdefgh,ef,gh->abcd",
            rho,
            covariance,
            covariance,
            optimize=True,
        )
        + np.einsum(
            "abcdefgh,eg,fh->abcd",
            rho,
            covariance,
            covariance,
            optimize=True,
        )
        + np.einsum(
            "abcdefgh,eh,fg->abcd",
            rho,
            covariance,
            covariance,
            optimize=True,
        )
    )
    return vertex


def _imaginary_time_propagator(
    offset_eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    a1: float,
    temperature: float,
    coefficients: scha.SchaCoefficients,
    tau: np.ndarray,
) -> np.ndarray:
    """Return C_ab(q,tau)=kBT sum_n exp(-i wn tau) G_ab(q,wn)."""

    static_eigenvalues = offset_eigenvalues + a1
    if np.any(static_eigenvalues <= 1.0e-14):
        raise ValueError("the one-shot propagator is not positive definite")
    kbt = scha.KB_HARTREE_PER_K * temperature
    beta_thermodynamic = 1.0 / kbt
    omega = np.sqrt(static_eigenvalues / coefficients.mass_au)
    denominator = (
        2.0
        * coefficients.mass_au
        * omega
        * (-np.expm1(-beta_thermodynamic * omega))
    )
    forward = np.exp(-omega[:, None, :] * tau[None, :, None])
    backward = np.exp(
        -omega[:, None, :]
        * (beta_thermodynamic - tau)[None, :, None]
    )
    mode_correlation = (forward + backward) / denominator[:, None, :]
    return np.einsum(
        "nai,nti,nbi->ntab",
        eigenvectors,
        mode_correlation,
        eigenvectors,
        optimize=True,
    )


def quartic_sunset_mass(
    a1: float,
    temperature: float,
    component_variance: float,
    coefficients: scha.SchaCoefficients,
    grid: SunsetGrid,
    tau_order: int,
) -> tuple[float, float, np.ndarray]:
    """Return the cubic mass projection of the one-shot quartic sunset."""

    if tau_order < 8:
        raise ValueError("tau_order must be at least 8")
    nodes, weights = np.polynomial.legendre.leggauss(tau_order)
    kbt = scha.KB_HARTREE_PER_K * temperature
    beta_thermodynamic = 1.0 / kbt
    tau = 0.5 * beta_thermodynamic * (nodes + 1.0)
    tau_weights = 0.5 * beta_thermodynamic * weights

    propagators = [
        _imaginary_time_propagator(
            eigenvalues,
            eigenvectors,
            a1,
            temperature,
            coefficients,
            tau,
        )
        for eigenvalues, eigenvectors in (
            (grid.eigenvalues_1, grid.eigenvectors_1),
            (grid.eigenvalues_2, grid.eigenvectors_2),
            (grid.eigenvalues_3, grid.eigenvectors_3),
        )
    ]
    vertex = effective_quartic_vertex(component_variance, coefficients)

    # Trace over external Cartesian indices. The uniform BZ average equals
    # Omega0/(2*pi)^3 int_BZ because Omega0=a0^3.
    trace_integrand = np.einsum(
        "acde,afgh,ntcf,ntdg,nteh->nt",
        vertex,
        vertex,
        propagators[0],
        propagators[1],
        propagators[2],
        optimize=True,
    )
    per_sample_mass = (
        -np.einsum("t,nt->n", tau_weights, trace_integrand, optimize=True)
        / 18.0
    )
    estimate = float(np.mean(per_sample_mass))
    iid_standard_error_diagnostic = float(
        np.std(per_sample_mass, ddof=1) / math.sqrt(grid.sample_count)
    )
    return estimate, iid_standard_error_diagnostic, vertex


def load_scha_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def evaluate_protocol(
    row: dict[str, float],
    prefix: str,
    coefficients: scha.SchaCoefficients,
    grid: SunsetGrid,
    tau_order: int,
) -> dict[str, float | bool]:
    a1 = row[f"{prefix}_A1_Ha_per_bohr2"]
    variance = row[f"{prefix}_mean_u_component_variance_bohr2"]
    delta_a1, iid_standard_error_diagnostic, vertex = quartic_sunset_mass(
        a1,
        row["temperature_K"],
        variance,
        coefficients,
        grid,
        tau_order,
    )
    corrected_mass = a1 + delta_a1
    stable = bool(corrected_mass > 0.0)
    dyson_chi = (
        coefficients.zstar**2 / (coefficients.omega0 * corrected_mass)
        if stable
        else math.nan
    )
    scha_chi = row[f"{prefix}_chi_soft"]
    # Strictly at the same perturbative order as delta_a1, expand the
    # observable instead of partially resumming its denominator.  This formal
    # curve remains drawable when A1+delta_A1 is negative; the ratio panel is
    # therefore indispensable for diagnosing loss of perturbative control.
    linearized_chi = scha_chi * (1.0 - delta_a1 / a1)
    return {
        f"{prefix}_post_delta_A1": delta_a1,
        f"{prefix}_post_delta_A1_iid_se_diagnostic": (
            iid_standard_error_diagnostic
        ),
        f"{prefix}_post_delta_over_A1": delta_a1 / a1,
        f"{prefix}_post_A1": corrected_mass,
        f"{prefix}_post_dyson_stable": stable,
        f"{prefix}_post_chi_dyson": dyson_chi,
        f"{prefix}_post_chi_linearized": linearized_chi,
        f"{prefix}_effective_vertex_xxxx": float(vertex[0, 0, 0, 0]),
        f"{prefix}_effective_vertex_xxyy": float(vertex[0, 0, 1, 1]),
    }


def run_one_shot(
    input_rows: list[dict[str, float]],
    sample_power: int,
    tau_order: int,
    seed: int,
) -> tuple[list[dict[str, float | bool]], dict[str, object]]:
    coefficients = mayer.MAYER_ANHARMONIC_COEFFICIENTS
    grid = build_sunset_grid(coefficients, sample_power, seed)
    output_rows: list[dict[str, float | bool]] = []
    for row in input_rows:
        output: dict[str, float | bool] = {
            "temperature_K": row["temperature_K"],
            "pointwise_pressure_GPa": row["pointwise_pressure_GPa"],
            "zero_scha_A1": row["zero_A1_Ha_per_bohr2"],
            "zero_scha_chi_soft": row["zero_chi_soft"],
            "lattice_matched_scha_A1": row[
                "lattice_matched_A1_Ha_per_bohr2"
            ],
            "lattice_matched_scha_chi_soft": row[
                "lattice_matched_chi_soft"
            ],
        }
        output.update(
            evaluate_protocol(
                row, "zero", coefficients, grid, tau_order
            )
        )
        output.update(
            evaluate_protocol(
                row, "lattice_matched", coefficients, grid, tau_order
            )
        )
        output_rows.append(output)

    summary: dict[str, object] = {
        "status": "provisional one-shot projected-local quartic post-SCHA",
        "model": coefficients.name,
        "sample_method": "scrambled Sobol on two independent BZ momenta",
        "sample_power": sample_power,
        "sample_count": grid.sample_count,
        "sobol_seed": seed,
        "imaginary_time_quadrature": "Gauss-Legendre",
        "tau_order": tau_order,
        "included": [
            "full quantum SCHA propagator including periodic dipolar kernel",
            "two-vertex projected-local quartic sunset",
            "local sextic contractions dressing the quartic vertex",
            "local octic contractions dressing the quartic vertex",
            "zero-pressure SCHA background",
            "existing lattice-matched-pressure SCHA background",
        ],
        "excluded": [
            "separate dynamic inhomogeneous-strain sunset vertices",
            "quartic-sextic, sextic-sextic, sextic-octic and octic-octic skeletons",
            "post-SCHA refit of the lattice-matching pressure",
            "self-consistent Dyson or 2PI resummation",
        ],
        "sign_convention": (
            "Gamma^(2)=A_SCHA+Sigma_post; negative delta_A1 softens the mode"
        ),
        "stability": {
            prefix: {
                "stable_points": int(
                    sum(
                        bool(row[f"{prefix}_post_dyson_stable"])
                        for row in output_rows
                    )
                ),
                "total_points": len(output_rows),
                "maximum_abs_delta_over_A1": float(
                    max(
                        abs(float(row[f"{prefix}_post_delta_over_A1"]))
                        for row in output_rows
                    )
                ),
            }
            for prefix in ("zero", "lattice_matched")
        },
        "numerical_error_note": (
            "The per-row iid_se_diagnostic is the dispersion of integrand "
            "samples divided by sqrt(N); it is not a rigorous randomized-"
            "QMC confidence interval. Independent scrambled-Sobol convergence "
            "must be used for a production uncertainty."
        ),
    }
    return output_rows, summary


def write_csv(rows: list[dict[str, float | bool]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, float | bool]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temperatures = np.asarray([float(row["temperature_K"]) for row in rows])
    mayer_t, _, mayer_chi = mayer_scan.load_mayer_experimental_response()

    fig, (response_axis, ratio_axis) = plt.subplots(
        2,
        1,
        figsize=(8.6, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    response_axis.scatter(
        mayer_t,
        mayer_chi,
        s=18,
        color="tab:cyan",
        edgecolor="black",
        linewidth=0.35,
        alpha=0.9,
        label="Expérience de Mayer (numérisée)",
        zorder=5,
    )

    protocols = (
        ("zero", "tab:blue", r"$p_{\rm eff}=0$"),
        (
            "lattice_matched",
            "tab:orange",
            r"$p_a(T)$ imposée par la maille",
        ),
    )
    for prefix, color, label in protocols:
        scha_values = np.asarray(
            [float(row[f"{prefix}_scha_chi_soft"]) for row in rows]
        )
        post_values = np.asarray(
            [float(row[f"{prefix}_post_chi_linearized"]) for row in rows]
        )
        response_axis.plot(
            temperatures,
            scha_values,
            linestyle="--",
            color=color,
            linewidth=1.5,
            alpha=0.75,
            label=f"SCHA, {label}",
        )
        response_axis.plot(
            temperatures,
            post_values,
            "o-",
            color=color,
            markersize=3.2,
            linewidth=1.8,
            label=f"one-shot linéarisé, {label}",
        )

        softening_ratio = -np.asarray(
            [float(row[f"{prefix}_post_delta_over_A1"]) for row in rows]
        )
        ratio_axis.plot(
            temperatures,
            softening_ratio,
            "o-",
            color=color,
            markersize=3.0,
            linewidth=1.5,
            label=label,
        )

    response_axis.set_yscale("log")
    response_axis.set_ylabel(r"Susceptibilité du mode mou $\chi_{\rm soft}$")
    response_axis.grid(alpha=0.22)
    response_axis.legend(frameon=False, fontsize=7.5, ncol=2)
    response_axis.set_title(
        "BaTiO$_3$ de Mayer : quartique post-SCHA one-shot projeté-local"
    )

    ratio_axis.axhline(1.0, color="tab:red", linestyle=":", linewidth=1.2)
    ratio_axis.axhline(0.0, color="0.45", linewidth=0.8)
    ratio_maximum = max(
        -float(row[f"{prefix}_post_delta_over_A1"])
        for prefix in ("zero", "lattice_matched")
        for row in rows
    )
    ratio_axis.axhspan(
        1.0,
        1.08 * ratio_maximum,
        color="tab:red",
        alpha=0.07,
        label="régime one-shot non contrôlé",
    )
    ratio_axis.set_ylim(0.0, 1.08 * ratio_maximum)
    ratio_axis.set_xlabel("Température (K)")
    ratio_axis.set_ylabel(r"$-\delta A_1^{(2)}/A_1$")
    ratio_axis.grid(alpha=0.22)
    ratio_axis.legend(
        loc="lower right", frameon=True, framealpha=0.92, fontsize=8
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--sample-power", type=int, default=12)
    parser.add_argument("--tau-order", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20220812)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_rows = load_scha_rows(args.input)
    rows, summary = run_one_shot(
        input_rows,
        sample_power=args.sample_power,
        tau_order=args.tau_order,
        seed=args.seed,
    )
    write_csv(rows, args.output_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_plot(rows, args.output_png)
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_png}")
    print(json.dumps(summary["stability"], indent=2))


if __name__ == "__main__":
    main()
