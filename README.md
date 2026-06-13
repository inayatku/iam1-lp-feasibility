# Paper: IAM-1 — An Artificial-Free Geometric Pivot Method for LP Feasibility

Code and data reproduction package for the paper *IAM-1: An Artificial-Free
Geometric Pivot Method for Linear Programming Feasibility*, following a full
recheck and regeneration of the paper's computational results.

**Repository:** https://github.com/inayatku/iam1-lp-feasibility
**Zenodo DOI:** https://doi.org/10.5281/zenodo.20668374

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
on Zenodo at https://doi.org/10.5281/zenodo.20668374.

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

## Findings of the June-2026 recheck (what changed in the manuscript)

1. Random-LP tables (1-3, 5-6): every iteration count and success rate
   reproduced exactly from seed 43, including HiGHS under SciPy 1.17.
   Conventions now stated in the text: an iteration means one pivot
   operation (uniform across methods; an instance feasible at the initial
   basis counts as zero); Pan's averages cover successful runs only;
   IAM-1/simplex/HiGHS solved every instance.
2. Table 6 ratio column recomputed from full precision: dense 1.08/1.65/2.68,
   sparse 1.02/1.32/2.34.
3. Table 4 timings replaced with the regenerated (2026) measurements from
   the sequentially-run dense configurations; the IAM-1 vs Pan per-iteration
   cost ratio is 1.5-1.8x at the informative sizes.
4. NETLIB Table 7: two stale cells corrected to the authoritative
   maxiter-6000 log (WOODW simplex 381 -> 385, FFFFF800 simplex 360 -> 379);
   all other cells confirmed.
5. NETLIB Table 9 (overall) was **not reproducible** from any saved log and
   has been recomputed from the authoritative log over the 52-problem set
   (54 converted instances minus QAP15, memory error, and MAROS-R7,
   conversion artifact): IAM-1 98.1% (1 failure: DEGEN3), Pan 86.5%
   (7 failures), simplex 88.5% (6 failures), HiGHS 100%; geometric means
   reported over each method's successes and over the 42 commonly solved
   instances. Dependent prose updated (IAM-1 ~ half of simplex iterations on
   the common set, not "a quarter").
6. Appendix A added listing the 52 instances and the two exclusions.
7. The earlier draft's Pan failure list (5) and simplex failure count (4)
   could not be reproduced; the log shows 7 and 6. The paper now reports the
   log values. **Never reintroduce the old numbers without a new run.**

## Provenance notes

- `iam1.py` (May 2025, both variants) reports exactly one more iteration than
  `iam1_original.py` on every pivot-requiring run; the archived logs match
  `iam1_original.py`, which is the implementation used everywhere here.
- The `Correct_version_of_IAM1` snapshot's `simplex_phase_1.py` has an
  index-out-of-range bug in its ratio-test tie-breaking
  (`valid_ratio_indices[min_ratio_idx]` mixes two index spaces) and was NOT
  used for the paper's results. The root `simplex_phase_1.py` (in `code/`
  here) matches the logs.
- NETLIB was not fully rerun in 2026 (hours of compute; QAP15 needs >2 GB
  dense). Subset validation: AFIRO, SC50A exact; SCSD1, BRANDY exact for all
  three methods with `iam1_original`.
