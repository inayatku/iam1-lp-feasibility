# IAM-1 — An Artificial-Free Geometric Pivot Method for LP Feasibility

Code and data to reproduce the computational results of the paper
*IAM-1: An Artificial-Free Geometric Pivot Method for Linear Programming Feasibility*.

- **Repository:** https://github.com/inayatku/iam1-lp-feasibility
- **Zenodo:** https://doi.org/10.5281/zenodo.20678797

The manuscript is not included here; it is released on publication.

## Contents

| Folder | Contents |
|---|---|
| `code/` | `iam1.py` (IAM-1), `simplex_phase_1.py` (simplex Phase 1), `Pans_method_row_scale.py` (Pan's method), the random-LP comparison driver `iam1_random_comparison_with_highs.py`, shared utilities (`iam_norm_utils.py`, `mat2dict.py`), and configuration files. |
| `results/` | Per-instance result logs (CSV) for the random-LP and NETLIB experiments that back the paper's tables. |
| `verification/` | `verify_paper_tables.py` checks every value in the paper's tables against `results/`. `make_performance_profile.py` regenerates the Dolan–Moré performance profile (Figure 1). |
| `figures/` | `netlib_performance_profile.pdf` (Figure 1). |

## Reproduce

```bash
# Random-LP experiments (menu answers: seed 43; feasible; 30 problems x 10 sizes; dense; no plots)
cd code
printf "43\n1\n2\n1\nn\n" | python iam1_random_comparison_with_highs.py

# Check every table value in the paper against the result logs
python ../verification/verify_paper_tables.py ../results

# Regenerate Figure 1 from the NETLIB log
python ../verification/make_performance_profile.py
```

Requires Python 3.11+, numpy, scipy (>= 1.9, HiGHS backend), pandas, matplotlib.

## Conventions

Iteration counts are pivots performed, counted the same way for every method; an
instance feasible at the initial basis counts as zero. The NETLIB problems are
taken from the standard, publicly available NETLIB LP test collection.

## License

MIT (see `LICENSE`). If you use this code or data, please cite the paper (see `CITATION.cff`).
