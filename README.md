# spectral-pideeponet-elasticity

## Requirements

Python 3.10+, `torch`, `numpy`, `scipy`, `pandas`, `matplotlib`.

```bash
pip install torch numpy scipy pandas matplotlib
```

## Entry points

| Script | Purpose |
|---|---|
| `continuum_elasticity_pideeponet_publication_study.py` | Original main study (PI vs data-only spectral DeepONet, ablations, seeds) |
| `cmame_extended_study.py` | Extended study (FNO baseline, FEM timing, noise, OOD) |
| `revision_new_benchmarks.py` | Revision experiments: non-sine forcings with FEM ground truth, ν sweep, capacity-matched FNO, accuracy–cost Pareto data, OOD capacity ablation. Start with `--selftest` (FEM convergence check). |
| `revision_final_additions.py` | LSFEM stress baseline (`--lsfem_selftest`, `--lsfem_stress_table`) and heterogeneous-inclusion benchmark (`--hetero_selftest`, `--hetero_gen_part`, `--hetero_merge`, `--hetero_fem_timing`) |
| `revision_chunk_runner.py` | Checkpoint/resume training driver for all revision models, including the least-squares-anchored branch and the rational-feature anchored variants. Run repeatedly until `ALL_JOBS_DONE`: `python revision_chunk_runner.py 3600` |
| `linear_baseline.py` | Closed-form linear read-out baseline (Table: linear-optimal calibration) |
| `hetero_linear.py` | Closed-form feature-basis baselines for the heterogeneous case (linear / bilinear / quadratic / rational-in-k) |
| `fair_fem_timing.py` | Amortized (factorization-reuse) FEM timing across mesh refinement |
| `aliasing_quantification.py` | Spectral aliasing of the equilibrium operator + conditioning of the coefficient-to-feature map |

## Strong-baseline benchmark (reframed study)

These scripts add the strong classical baselines and the heterogeneous
modulus-field benchmark used in the accuracy–speed–memory comparison.

| Script | Purpose |
|---|---|
| `rom_baseline.py` | POD–Galerkin reduced-order model on the fixed-operator (K=16) benchmark; sweeps POD rank, reports error, online cost, and on-disk basis size |
| `hetero_field.py` | Route-B physics: random log-normal `E(x)`-field Q4 FEM. `K(E)=Σ_e E_e K_e^{(1)}` re-factorized per query. `--selftest` verifies `E≡1` reproduces the homogeneous solver and checks mesh convergence |
| `hetero_field_gen.py` | Ground-truth generator for the `E(x)`-field benchmark (chunk-friendly for Slurm arrays; `--merge` to assemble) |
| `route_b.py` | Trains the `E(x)`-conditioned models: `spectral_e` (exact-BC sine trunk, branch ingests f + cosine projection of log E) and `fno_e` (5-channel FNO); variable-coefficient Navier–Cauchy residual |
| `eval_route_b.py` | Capstone eval of a trained checkpoint: field accuracy, UQ-propagation QoI distributions (strain energy, peak von Mises), and the resource ledger row |
| `rom_field.py` | POD–Galerkin ROM over the `E(x)` family (per-query assemble + project + solve); POD floor, ROM error, and the rank-vs-cost cliff |
| `ledger.py` | Master accuracy–time–RAM–disk ledger across a resolution sweep for FEM and ROM on both benchmarks (`--driver`); per-method RAM measured in isolated subprocesses |
| `make_figures.py` | Generates the benchmark figures (accuracy–cost Pareto, POD spectra, ROM cost cliff) from the ledger/ROM JSONs |
| `cluster/*.sh` | Slurm job scripts (RIT SPORC) for data generation, training, ROM/ledger runs, and evaluation |

## Typical reproduction order

```bash
python revision_new_benchmarks.py --selftest        # verify FEM (O(h^2) rates)
python revision_final_additions.py --lsfem_selftest # verify LSFEM
python revision_final_additions.py --hetero_selftest
python aliasing_quantification.py
python fair_fem_timing.py
python revision_new_benchmarks.py --exp pareto
python revision_final_additions.py --hetero_gen_part tr 1000 0 1000
python revision_final_additions.py --hetero_gen_part te 200 0 200
python revision_final_additions.py --hetero_merge
python revision_chunk_runner.py 100000              # all training jobs
python linear_baseline.py
python hetero_linear.py
python revision_final_additions.py --lsfem_stress_table
```
