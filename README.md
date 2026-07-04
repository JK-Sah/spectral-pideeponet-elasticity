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
