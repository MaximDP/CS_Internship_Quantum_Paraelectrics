# SCHA paraelectric solver

Solves the paraelectric SCHA mass equation in `rapportCS_stage_maxime/rapportCS.tex`.

The default calculation uses the Nishimatsu coefficients and keeps the quartic,
sixth-order, and eighth-order SCHA corrections:

```text
A_ab(k,T) = A0_ab(k)
          + 3 kBT beta_eff_abgd G_gd
          + 15 (kBT)^2 gamma_abgder G_gd G_er
          + 105 (kBT)^3 rho_abgderhl G_gd G_er G_hl
```

Only `A1(T)` is solved self-consistently. Brillouin-zone integrals use the full
periodic Nishimatsu harmonic kernel: the FERAM 26-neighbour short-range matrix
and the Ewald dipolar matrix. `A2`, `A3`, `A4`, and `A5` are the small-`k`
expansion coefficients and are not extrapolated to the zone boundary.

The Nishimatsu polynomial coefficients are converted to the document tensor
normalizations:

```text
E6 = (1/6) gamma_abcdef u_a...u_f
E8 = (1/8) rho_abcdefgh u_a...u_h
```

Run:

```bash
python3 scha_paraelc.py --temperature 500 --ngrid 40
```

Available coefficient sets:

```bash
python3 scha_paraelc.py --material nishimatsu --temperature 500
python3 scha_paraelc.py --material nishimatsu_sto --temperature 100
python3 scha_paraelc.py --material vanderbilt --temperature 500
```

`nishimatsu_sto` is the WC-GGA SrTiO3 set from Table I of Nishimatsu et al.,
J. Phys. Soc. Jpn. 85, 114714 (2016).  The 2010 paper contains the SrTiO3
energy surfaces but publishes a complete effective-Hamiltonian table only for
BaTiO3.

`ngrid` must be even so that the non-analytic Gamma point is approached without
placing a quadrature node exactly at `k=0`.

Negative static eigenvalues of the trial kernel are not allowed: a candidate
`A1` is discarded as soon as one integration-grid mode has `mu_i(k) <= 0`.
If several stable sign-changing brackets remain, the script picks the branch
whose bracket center is closest to `A01`. To force another detected branch:

```bash
python3 scha_paraelc.py --root-index 0 --temperature 500
```

The implementation follows the diagonalization/Matsubara procedure described
around Eqs. (89)--(96):

```text
mu(k; A1) = O(k) diag(mu_i(k; A1)) O(k)^T
omega_i(k; A1)^2 = mu_i(k; A1)/M*
sum_n 1/[M*(omega_n^2 + omega_i^2)]
  = coth[omega_i/(2 kBT)]/[2 M* kBT omega_i]
G_ab = Omega0/(2 pi)^3 int d^3k [O diag(sum_i) O^T]_ab
```

Non-positive static eigenvalues are excluded.

## Temperature scan of the static susceptibility

The zero-pressure SrTiO3 quantum-paraelectric profile is generated with

```bash
python3 scan_chi_temperature_srtio3.py --start 415 --stop 565 --step 5 --ngrid 40
```

This uses the same temperature interval and the same dynamic
`H^dagger G_w H` acoustic-strain calculation as the BaTiO3 profile.  The plot
also retains the former projected polar-only closure and the Mueller--Burkard
single-crystal Barrett fit for comparison.  The oxygen-rotation mode of the
105 K antiferrodistortive transition remains outside the Hamiltonian.

The paraelectric transverse susceptibility can be evaluated on a regular
temperature grid while reusing the full lattice eigensystem:

```bash
python3 scan_chi_temperature.py --start 415 --stop 515 --step 5 --ngrid 40
```

This produces a CSV table and a PNG plot in `../outputs/paraelectric/`.  The plotted dc
susceptibility uses only the zero Matsubara-frequency response,
`chi_T = Omega0/A1_P(T)`; the Matsubara sum remains inside the SCHA equation
that determines `A1(T)`.

Hydrostatic pressure can be included through the strain-elimination kernel

```text
P^(p,u) = -p a0^3 (B1xx + 2 B1yy)/(B11 + 2 B12),
```

using the Nishimatsu elastic and local-mode--strain parameters.  A constant
effective pressure and the thermal-expansion prescription used in the FERAM
molecular-dynamics paper are selected with

```bash
python3 scan_chi_temperature.py --pressure-model constant --pressure-gpa -0.6
python3 scan_chi_temperature.py --pressure-model thermal-expansion --skip-unstable
```

The second command uses `p(T)=-0.005*T GPa`.  `--skip-unstable` retains the
requested temperature grid and writes `NaN` where no stable centered
paraelectric SCHA root exists; without that flag the scan stops at the first
such point.

## Scan with the inhomogeneous acoustic strain

The alternative script

```bash
python3 scan_chi_temperature_inhomogeneous.py --start 400 --stop 565 --step 5 --ngrid 40
```

keeps the reference scan unchanged and adds the acoustic-strain correction

```text
Sigma_loc = 3 kBT beta G
Sigma_hom = -1/2 kBT Lambda_hom G  (thermodynamic limit)
Sigma_inh(K) = -kBT sum_Q M_inh(K-Q) G_P(Q)
M_inh(Q) = H(Q)^* G_w(Q) H(Q)
G_w(k, omega_n) = [m_w omega_n^2 + Phi_elastic(k)]^-1
```

The bare local tensor `beta`, the all-to-all homogeneous kernel `Lambda_hom`,
and the inhomogeneous kernel `M_inh` remain separate.  The elastic kernel
`Phi_elastic(k)` and mixed tensor `H(k)` reproduce Eqs. (17)--(23) of
Nishimatsu et al., Phys. Rev. B 78, 104104 (2008), including the shear factors
in the 3x6 matrix.  The acoustic mass is `46.44 amu`, and the product of the
acoustic and polar propagators is summed analytically over all bosonic
Matsubara frequencies.  For the dc mass, the direct Wick channel transfers
zero momentum and vanishes because `H(0)=0`; the code evaluates the two crossed
channels with transferred `(k, omega_n)`.

Equation (133) also displays a homogeneous exchange term proportional to
`A^-1(k=0)/N`.  Unlike the other loops it contains no extensive momentum sum,
so it vanishes for `N -> infinity`.  The Brillouin-zone quadrature implements
this thermodynamic limit; its integration order is not interpreted as a
finite-supercell size.

One run compares two mechanical protocols.  By default, the blue curve uses a
temperature-dependent pressure obtained by inverting the cubic SCHA equation
of state at every plotted temperature.  Its target is a weighted linear fit to
the single-crystal X-ray lattice parameters of Nakatani et al. over
`413--598 K`, stored in
`../inputs/paraelectric/nakatani2016_batio3_cubic_lattice.csv`:

```text
a_exp_fit(T) = 4.98674856788e-5*T + 3.98930432811 Angstrom.
```

The pressure required pointwise by this target is then compressed into

```text
p_a(T) = -0.00485972840897*T + 0.250696172704 GPa.
```

On the production `40^3` grid this affine pressure reproduces
`a_exp_fit(T)` with a maximum error of `1.92e-6 Angstrom` over `400--565 K`;
values below `413 K` use a short extrapolation of the cubic fit.  The second
curve uses an affine pressure fitted by minimizing the vertical least-squares
distance to the digitized Wieczorek susceptibility data,

```text
sum_i [chi_SCHA(T_i,p(T_i)) - chi_Wieczorek(T_i)]^2,
p(T) = 0.00634746406215*T - 8.19433710756 GPa.
```

The 14 digitized temperatures from `415.65 K` to `448.15 K` enter the fit
directly.  The lattice parameter is reconstructed from the centered SCHA
covariance through

```text
<u_x^2> = kBT Tr(G)/3
eta = -[(B1xx+2 B1yy)<u_x^2>/2 + p a0^3]/(B11+2 B12)
a(T,p) = a0 [1+eta].
```

The experimental lattice fit interval can be changed with `--lattice-fit-start`
and `--lattice-fit-stop`.  A legacy single-point calibration is selected by
providing both
`--lattice-calibration-target-angstrom` and
`--lattice-calibration-temperature`; `--constant-pressure-gpa` instead selects
an explicit constant pressure.  The previous susceptibility calibration remains available with
`--fit-pressure-temperature 500`.  The older affine fit to the empirical
Curie--Weiss parametrization reported by Barrett is available with
`--fit-linear-pressure`; an explicit line, including Nishimatsu's
`p(T)=-0.005*T GPa`, can be selected with
`--use-explicit-linear-pressure`, `--linear-pressure-slope`, and
`--linear-pressure-intercept`.  The dielectric-fitted orange curve is evaluated
at the 14 measured temperatures from `415.65 K` through the last Wieczorek
point at `448.15 K`.  The same affine pressure law is then extrapolated to
`560 K` and drawn as an orange dashed line, explicitly distinguishing that
continuation from the calibration interval.  The script writes one comparison CSV and PNG in
`../outputs/paraelectric/`, including the reconstructed lattice parameter and
isotropic strain for both pressure protocols.
Unstable centered solutions are retained as `NaN`, so failure of the
paraelectric branch is visible rather than silently discarded.
