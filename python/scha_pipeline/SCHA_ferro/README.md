# SCHA ferroelectric solver

Solves the direction-dependent ferroelectric SCHA stationarity equations
(159)--(160) in `rapportCS_stage_maxime/rapportCS.tex`.

The trial kernel contains a full symmetric positive local mass matrix,

```text
A_ab(k, omega_n; T) = M* omega_n^2 delta_ab
                    + A_loc,ab(T)
                    + A_lattice,ab(k) - A01 delta_ab.
```

The numerical root has nine unknowns and nine residuals:

```text
six independent entries of A_loc,ab
three Cartesian components (u_min,x, u_min,y, u_min,z)
```

Positive definiteness of `A_loc` is enforced through a Cholesky
parameterization.  The three background equations are divided by `|u_min|`,
which selects the nonzero ferroelectric branch without prescribing its
direction.  `--direction` is only an initial direction; the reported optimized
direction is obtained from the converged vector.

The Gamma-point mass and background equations contain the local quartic,
homogeneous-strain, and dynamic inhomogeneous-strain sectors separately.  The
sixth- and eighth-order local terms are also retained.  Brillouin-zone loops use
the full periodic Nishimatsu/FERAM harmonic kernel.  The polar and acoustic
mode diagonalizations and Matsubara products are vectorized over the complete
quadrature grid.

Run the default 100 K calculation with

```bash
python3 scha_ferro.py --temperature 100 --ngrid 12 \
  --json result_100K_direction_free_n12.json
```

The default initial direction is the generic vector `(1, 0.8, 0.6)`, avoiding
an artificial high-symmetry constraint.  Explicit initial directions can still
be tested:

```bash
python3 scha_ferro.py --temperature 100 --direction 001
python3 scha_ferro.py --temperature 100 --direction 110
python3 scha_ferro.py --temperature 100 --direction 111
```

For the Nishimatsu coefficients at 100 K and quadrature order 12, the generic
initial vector converges to the rhombohedral direction `[111]`:

```text
optimized direction = (0.57735027, 0.57735027, 0.57735027)
|p_min|             = 14.1092989241 bohr/sqrt(Hartree)
|u_min|             = 0.251082558652 bohr
A_T                 = 0.00307735242012 Hartree/bohr^2
A_L                 = 0.0269431959670 Hartree/bohr^2
minimum grid mass   = 0.00898586171332 Hartree
total residual norm = 2.4e-16
```

Here `u_min = sqrt(k_B T) p_min` is the physical local-mode displacement in
the normalization used by the script.  The centered paraelectric solution
`p_min=0` is deliberately excluded from this nonzero-branch solver.  Several
high-symmetry stationary branches can coexist; a residual root establishes a
self-consistent branch, while a variational-free-energy comparison is still
needed to prove which branch is the global thermodynamic minimum.

## Zero temperature and cryogenic susceptibility

Internally the solver uses the physical covariance
`Q = k_B T G = <delta u delta u>` and `u_min`, not the singular scaled
variables `G` and `p_min`.  It can therefore evaluate `T=0` exactly: the mode
covariance becomes `1/(2 M omega)` and the dynamic acoustic convolution uses
the analytic zero-temperature Matsubara limit.  For example,

```bash
python3 scha_ferro.py --temperature 0 --ngrid 12
```

The 0--50 K continuation scan is

```bash
python3 scan_chi_temperature.py
```

It writes into `../outputs/ferroelectric/`:

- `chi_ferro_0_50K_n12.csv`: masses, displacement, the longitudinal,
  transverse, and pseudocubic `[100]` susceptibilities and relative
  permittivities;
- `chi_ferro_0_50K_n12.png`: the temperature profile and the comparison to
  Wul (1946) and Holste, Lawless, and Samara (1976).

The conversion from the local-mode mass to the intrinsic single-domain
response is

```text
chi = Z*^2 / Omega0 A_loc^(-1)
epsilon_r = epsilon_infinity I + 4 pi chi.
```

The experiment comparison is necessarily component-sensitive.  The accessible
summary of the old absolute datum `epsilon_r(4.2 K) = 100` does not identify
the orientation, while the model distinguishes a small longitudinal and a
much larger transverse response.  Domain-wall and sample-history effects are
not part of the intrinsic SCHA curve.
