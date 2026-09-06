#!/usr/bin/env python3
"""Build the uncluttered Mayer/SCHA comparison used on slide 22."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MAYER_DIR = ROOT / "python" / "scha_pipeline"
RESULTS = MAYER_DIR / "outputs" / "reference" / "mayer" / "mayer_bto_400_800K_n40.csv"
EXPERIMENT = MAYER_DIR / "inputs" / "mayer" / "mayer2022_batio3_fig3a_experiment_digitized.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "mayer_scha_slide_simplified.png"
EPS_INF_MAYER = 6.847


def read_columns(path: Path, *columns: str) -> tuple[np.ndarray, ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return tuple(
        np.asarray([float(row[column]) for row in rows], dtype=float)
        for column in columns
    )


def main() -> None:
    temperature, zero_chi, matched_chi = read_columns(
        RESULTS,
        "temperature_K",
        "zero_chi_soft",
        "lattice_matched_chi_soft",
    )
    experiment_temperature, experiment_epsilon = read_columns(
        EXPERIMENT,
        "temperature_K",
        "epsilon_r",
    )
    experiment_chi = (experiment_epsilon - EPS_INF_MAYER) / (4.0 * np.pi)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 15,
            "legend.fontsize": 14.0,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "lines.linewidth": 2.6,
        }
    )
    figure, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)

    axis.plot(
        experiment_temperature,
        experiment_chi,
        linestyle="none",
        marker="o",
        color="#18A7C3",
        markeredgecolor="#174956",
        markeredgewidth=0.55,
        markersize=5.4,
        zorder=5,
        label="Expérience Mayer et al.",
    )
    axis.plot(
        temperature,
        zero_chi,
        "D--",
        color="#8E63BE",
        markersize=5.2,
        label=r"SCHA : $p_{\rm eff}=0$",
    )
    axis.plot(
        temperature,
        matched_chi,
        "o-",
        color="#2477B3",
        markersize=5.0,
        label="SCHA : maille expérimentale",
    )

    axis.set_xlim(300.0, 520.0)
    axis.set_ylim(7.0, 230.0)
    axis.set_yscale("log")
    axis.set_xlabel("Température (K)")
    axis.set_ylabel(r"Susceptibilité du mode mou $\chi_T$")
    axis.grid(which="major", color="#6E7781", alpha=0.22, linewidth=0.9)
    axis.grid(which="minor", color="#6E7781", alpha=0.09, linewidth=0.65)
    axis.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
        loc="upper right",
        handlelength=2.5,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=320, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
