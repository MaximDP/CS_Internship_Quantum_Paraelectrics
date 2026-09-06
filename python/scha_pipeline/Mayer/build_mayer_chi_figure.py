#!/usr/bin/env python3
"""Build the publication-ready Mayer susceptibility benchmark figure.

The numerical scans are read from the frozen Mayer result tables. This script
only redraws the susceptibility panel of ``run_mayer_bto.py`` and does not
rerun or modify the production SCHA calculation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
RESULTS = MODULE_DIR.parent / "outputs" / "reference" / "mayer" / "mayer_bto_400_800K_n40.csv"
PREVIOUS = MODULE_DIR.parent / "outputs" / "reference" / "mayer" / "nishimatsu_previous_predictions_400_800K_n40.csv"
MAYER_EXPERIMENT = MODULE_DIR.parent / "inputs" / "mayer" / "mayer2022_batio3_fig3a_experiment_digitized.csv"
WIECZOREK_EXPERIMENT = MODULE_DIR.parent / "inputs" / "mayer" / "wieczorek2006_batio3_fig1a_digitized.csv"
OUTPUT = MODULE_DIR.parent / "outputs" / "mayer" / "mayer_chi_temperature_300_800K_final.png"
EPS_INF_MAYER = 6.847


def read_columns(path: Path, *columns: str) -> tuple[np.ndarray, ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return tuple(
        np.asarray([float(row[column]) for row in rows], dtype=float)
        for column in columns
    )


def permittivity_to_soft_chi(epsilon_r: np.ndarray) -> np.ndarray:
    return (epsilon_r - EPS_INF_MAYER) / (4.0 * np.pi)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temperatures, zero_chi, matched_chi, affine_chi = read_columns(
        RESULTS,
        "temperature_K",
        "zero_chi_soft",
        "lattice_matched_chi_soft",
        "affine_pressure_chi_soft",
    )
    old_t, old_lattice_chi, old_response_fit_chi, old_zero_chi = read_columns(
        PREVIOUS,
        "temperature_K",
        "nishimatsu_lattice_matched_chi",
        "nishimatsu_response_fitted_chi",
        "nishimatsu_zero_pressure_chi",
    )
    mayer_t, mayer_epsilon = read_columns(
        MAYER_EXPERIMENT, "temperature_K", "epsilon_r"
    )
    wieczorek_t, wieczorek_epsilon = read_columns(
        WIECZOREK_EXPERIMENT, "temperature_K", "epsilon_r"
    )
    mayer_chi = permittivity_to_soft_chi(mayer_epsilon)
    wieczorek_chi = permittivity_to_soft_chi(wieczorek_epsilon)

    fit_start = float(np.min(wieczorek_t))
    fit_stop = float(np.max(wieczorek_t))
    fit_t = np.unique(
        np.concatenate(
            ([fit_start], old_t[(old_t > fit_start) & (old_t < fit_stop)], [fit_stop])
        )
    )
    fit_chi = np.interp(fit_t, old_t, old_response_fit_chi)
    extrapolated_t = np.unique(np.concatenate(([fit_stop], old_t[old_t > fit_stop])))
    extrapolated_chi = np.interp(extrapolated_t, old_t, old_response_fit_chi)
    barrett_t = old_t[old_t >= fit_start]
    barrett_chi = (1.5e5 / (barrett_t - 390.0) - 6.0) / (4.0 * np.pi)

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 1.7,
        }
    )
    fig, axis = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)

    axis.plot(
        temperatures,
        zero_chi,
        "D-.",
        color="tab:purple",
        markersize=3.8,
        label=r"Mayer SCHA: $p_{\rm eff}=0$",
    )
    axis.plot(
        temperatures,
        matched_chi,
        "o-",
        color="tab:blue",
        markersize=3.8,
        label="Mayer SCHA: lattice match",
    )
    axis.plot(
        temperatures,
        affine_chi,
        ":",
        color="tab:orange",
        linewidth=2.0,
        label="Mayer SCHA: affine lattice fit",
    )
    axis.plot(
        old_t,
        old_zero_chi,
        "--",
        color="tab:purple",
        linewidth=1.45,
        label=r"Nishimatsu: $p_{\rm eff}=0$",
    )
    axis.plot(
        old_t,
        old_lattice_chi,
        "--",
        color="tab:blue",
        linewidth=1.45,
        label="Nishimatsu: lattice match",
    )
    axis.plot(
        fit_t,
        fit_chi,
        "-",
        color="tab:red",
        linewidth=1.8,
        label=r"Nishimatsu: fitted $p_\chi(T)$",
    )
    axis.plot(
        extrapolated_t,
        extrapolated_chi,
        "--",
        color="tab:red",
        linewidth=1.45,
        label=r"Same $p_\chi(T)$, extrapolated",
    )
    axis.plot(
        barrett_t,
        barrett_chi,
        "--",
        color="tab:green",
        linewidth=1.45,
        label="Barrett empirical Curie-Weiss",
    )
    axis.plot(
        mayer_t,
        mayer_chi,
        linestyle="none",
        marker="o",
        color="tab:cyan",
        markeredgecolor="0.25",
        markeredgewidth=0.35,
        markersize=3.2,
        zorder=5,
        label="Mayer et al., experiment (digitized)",
    )
    axis.plot(
        wieczorek_t,
        wieczorek_chi,
        linestyle="none",
        marker="s",
        color="black",
        markerfacecolor="white",
        markeredgewidth=0.8,
        markersize=4.6,
        zorder=6,
        label="Wieczorek et al., 1 kHz (digitized)",
    )

    axis.set_xlim(300.0, 800.0)
    axis.set_yscale("log")
    axis.set_xlabel("Temperature (K)")
    axis.set_ylabel(r"Soft-mode susceptibility $\chi_T$")
    axis.grid(which="major", alpha=0.24, linewidth=0.8)
    axis.grid(which="minor", alpha=0.10, linewidth=0.55)
    axis.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        columnspacing=1.25,
        handlelength=2.7,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
