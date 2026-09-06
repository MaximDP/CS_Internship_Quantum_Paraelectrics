#!/usr/bin/env python3
"""Compute one paraelectric SCHA state from a JSON input file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'SCHA_paraelc'))
import scha_paraelc as scha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'inputs/examples/batio3_500K.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/example')
    args = parser.parse_args()
    config = json.loads(args.input.read_text(encoding='utf-8'))
    unknown = set(config) - {'material', 'temperature_K', 'ngrid', 'tolerance'}
    if unknown:
        parser.error('unknown input keys: ' + ', '.join(sorted(unknown)))
    coefficients = scha.MATERIALS[config['material']]
    result = scha.solve_a1(
        config['temperature_K'], config['ngrid'], coefficients.kmax,
        coefficients, tolerance=config.get('tolerance', 1e-12),
    )
    chi = coefficients.zstar**2 / (coefficients.omega0 * result['A1'])
    response = {
        'temperature_K': result['temperature_K'],
        'A1_u_Ha_per_bohr2': result['A1'],
        'chi_soft_gaussian': chi,
        'epsilon_r': coefficients.eps_inf + 4 * np.pi * chi,
        'covariance_u_bohr2': (
            scha.KB_HARTREE_PER_K * result['temperature_K'] * np.asarray(result['G'])
        ).tolist(),
        'closure': 'projected local paraelectric SCHA; zero pressure',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, payload in [('input.json', config), ('scha.json', result), ('response.json', response)]:
        (args.output / filename).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"A1 = {result['A1']:.12g} Ha/bohr^2; chi = {chi:.10g}; epsilon_r = {response['epsilon_r']:.10g}")
    print(f"Results: {args.output.resolve()}")


if __name__ == '__main__':
    main()
