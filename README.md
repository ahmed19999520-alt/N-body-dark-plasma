# N-body-dark-plasma: code_sim

Numerical librry for the SU(3)_dark sector, It implements the dark octet spectrum, the coupled Boltzmann hierarchy with co-annihilation, velocity-dependent self-interaction, the first-order confinement gravitational-wave background with LISA forecasts, dark acoustic oscillations, kinetic-mixing direct detection, FCC-ee and FCC-hh signatures, direct and Barnes-Hut N-body dynamics with SIDM scattering and a particle-in-cell dark plasma, and a coarse-grained Bayesian reconstruction pipeline.
Note :
The topological heavy sector is not part of the core library. It is provided as an optional complementary layer in N_bodyDP/topological and is enabled explicitly.
Installation
The topological heavy sector is outside the scope of this library.

## Installation

    pip install -e .[dev]
    pytest

Docker:

    docker compose run tests
    docker compose run pipeline

## Usage

    python scripts/build_data.py
    python scripts/run_full_pipeline.py --output results
    python scripts/run_full_pipeline.py --output results --quick
    python scripts/reproduce_paper_tables.py --output results
    python scripts/reproduce_paper_figures.py --output results

The pipeline writes JSON results, CSV tables, figures, and a verification report (`verification.md`) comparing every published number with the library value.

## Architecture

    N_bodyDP/
        constants.py                   parameters and published targets
        gmo/                           quantum numbers, mass spectrum, homotopy diagnostics
        boltzmann/                     cross sections, freeze-out, coupled solver, relic calibration
        self_interaction/              Sommerfeld enhancement and sigma_T/m
        gravitational_waves/           bubble, sound-wave and turbulence spectra, LISA SNR and Fisher
        dark_acoustic/                 sound horizon and power-spectrum modulation
        direct_detection/              kinetic-mixing nuclear scattering
        collider/                      FCC-ee and FCC-hh rates and endpoints
        nbody/                         direct summation, octree, SIDM, PIC
        bayesian/                      priors, likelihood, nested sampling, posterior geometry
        plotting/                      spectra, maps and corner plots
        verification.py                published-versus-library checks
        pipeline.py                    end-to-end driver

## Reproduced benchmark

| quantity | published | library |
|---|---|---|
| M(chi0) | 165.67 GeV | 165.67 GeV |
| r21, r31, r41 | 1.089, 1.163, 1.260 | 1.0887, 1.1630, 1.2596 |
| f_peak | 1.65 mHz | 1.65 mHz |
| Omega_GW h^2 peak | 1e-8 | 1.03e-8 |
| SNR (4 yr, 2 yr) | 8.5, 6.0 | 8.5, 6.0 |
| delta f/f, delta Omega/Omega | 15%, 25% | 15.0%, 24.7% |
| sigma_SI | 3.1e-48 cm^2 | 3.1e-48 cm^2 |
| r_DAO | 2.3 Mpc | 2.3 Mpc |
| Omega_DM h^2 | 0.120 | 0.12000 (calibrated solver) |

## Calibration policy

Quantities that cannot follow from the printed closed-form expressions are tied to the published benchmark through explicit, named calibration constants rather than hidden adjustments. These are the sound-wave and turbulence frequency prefactors, the composite peak skew, the SNR calibration, the foreground marginalisation factors, the annihilation normalisation, the Yukawa suppression factor, and the direct-detection normalisation. Raw printed formulas are evaluated separately and reported in `verification.md` with status `differs` where they do not reproduce the published value.

## Tests

The suite covers the spectrum and ratios, homotopy numerics, Boltzmann solver, self-interaction, gravitational-wave amplitudes and SNR, collider rates, N-body accuracy and conservation laws, PIC energy conservation, Bayesian recovery, and the verification pipeline.

## Citation
```bibtex
@article{Ali2026firstorder ,
  author  = {Ali, Ahmed},
  title   = {First-Order Confinement Dark Phase Transition in a Non-Abelian Dark Sector,
  year    = {2026},
  note    = {Arxiv_hep-ph},
  url     = {https://github.com/ahmed19999520-alt/N-body-dark-plasma}
  doi     = {https://doi.org/10.48550/arxiv.to appear}
}
---

## License

MIT © Ahmed Ali 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, subject to the condition that the above
copyright notice and this permission notice appear in all copies.