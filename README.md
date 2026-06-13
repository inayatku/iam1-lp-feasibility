# Paper: IAM-1 — An Artificial-Free Geometric Pivot Method for LP Feasibility

Code and data reproduction package for the paper *IAM-1: An Artificial-Free
Geometric Pivot Method for Linear Programming Feasibility*, following a full
recheck and regeneration of the paper's computational results.

**Repository:** https://github.com/inayatku/iam1-lp-feasibility
**Zenodo DOI:** https://doi.org/10.5281/zenodo.20678798

## Layout

The manuscript (LaTeX source + compiled PDF) is kept private until
publication and is not included in this public repository.

| Folder | Contents |
|---|---|
| `code/` | The verified solver/experiment stack: `iam1_original.py` (the IAM-1 implementation whose iteration counts match the archived logs), `simplex_phase_1.py`, `Pans_method_row_scale.py`, `iam1_random_comparison_with_highs.py` (random-LP driver), `iam1_netlib_comparison.py`, configs. Two June-2026 fixes are documented in-line: missing `MIN/MAX_ROW_SCALE_VALUE` constants appended to `config_simplex.py`, and the driver imports `iam1_original` instead of the later `iam1.py` (which counts one extra iteration per run). |
| `results/original_2025/` | The archived May-2025 CSVs that back the paper's tables, including `netlib_batch_maxiter6000_results.csv` (the authoritative NETLIB log; it contains two passes over the test set with identical values). |
| `results/regenerated_2026/` | First June-2026 reruns of all four random-LP configurations (feasible/infeasible x dense/sparse, seed 43, 30 instances per size, max 1000 iterations). Iteration counts and success rates match the 2025 logs exactly. |
| `results/regenerated_2026_pan_fixed/` | The canonical reruns backing the manuscript's random-LP tables. Identical instance set and solver behavior; Pan's iteration counts are recorded as pivots performed, the uniform convention used for all methods in the paper (the earlier logs record Pan's count including its initial feasibility check, i.e. one higher on solved runs). The dense configurations were run one at a time, and Table 4's timings come from those runs. |
| `verification/` | `verify_paper_tables.py` — checks every number in the manuscript's tables against a results directory (passes 248/248 on `original_2025`, 178/178 on `regenerated_2026` and on `regenerated_2026_pan_fixed`); it normalizes Pan's counts from the earlier logs to the paper's pivots-performed convention. `validate_netlib_subset.py` — re-solves selected preprocessed NETLIB instances and compares with the 2025 log. `make_performance_profile.py` — regenerates the Dolan–Moré figure from the NETLIB log. |
| `figures/` | `netlib_performance_profile.pdf` — Dolan–Moré performance profile over 37 nontrivial NETLIB instances (Figure 1 of the manuscript). Key values: at tau=2, IAM-1 solves 81% within twice the best method (HiGHS 76%, Pan 68%, simplex 30%); asymptotes 97/100/81/84%. |

## Repository

Published at https://github.com/inayatku/iam1-lp-feasibility and archived
on Zenodo at https://doi.org/10.5281/zenodo.20678798.

## How to reproduce

```
cd code
# random-LP experiments (the four menu answer sequences: seed 43; 1/2 = feasible/infeasible;
# 2 = 30 problems x 10 sizes; 1/3 = dense/sparse; n = no plots)
printf "43\n1\n2\n1\nn\n" | python iam1_random_comparison_with_highs.py
# verify everything
cd ../verification
python verify_paper_tables.py ../results/original_2025
python verify_paper_tables.py ../results/regenerated_2026_pan_fixed
python validate_netlib_subset.py AFIRO SC50A SCSD1 BRANDY
```

Requires Python 3.11+, numpy, scipy (>= 1.9, HiGHS backend), pandas, matplotlib.
The preprocessed NETLIB pickles are not stored here (large); they live in
`Research_work_augment_old/Compuations_AIM-1_random_and_netlib_Lps/Netlib_files/inequality_form_splited/`.
