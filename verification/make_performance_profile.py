"""Dolan-More performance profile over the 52-problem NETLIB set, from the
authoritative log. Performance measure: iteration counts; runs that hit the
6,000-iteration limit are treated as failures (ratio = infinity). The 15
instances solved at the initial slack basis by the best method (0 iterations)
are excluded, since performance ratios are undefined there.

Writes figures/netlib_performance_profile.pdf (vector) and prints key values.
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "results", "original_2025",
                   "netlib_batch_maxiter6000_results.csv")
OUT = os.path.join(HERE, "..", "figures", "netlib_performance_profile.pdf")
LIM = 6000

METHODS = [("IAM-1", "iam1_iterations", "-"),
           ("Pan's method", "pan_row_scaled_iterations", "--"),
           ("Simplex Phase 1", "simplex_iterations", "-."),
           ("HiGHS", "highs_simplex_iterations", ":")]

rows = list(csv.DictReader(open(LOG, newline="")))
last = {}
for r in rows:
    last[r["problem"].replace("NAME", "").strip()] = r
probs = sorted(n for n in last if n not in ("QAP15", "MAROS-R7"))

# performance data; failures -> inf
#
# The archived log records Pan's count including the initial feasibility
# check on solved runs; subtract 1 to express it as pivots performed, the
# uniform convention used for all methods in the paper (capped runs record
# the pivot count and are left unchanged).
def value(p, col):
    v = float(last[p][col])
    if col == "pan_row_scaled_iterations" and v < LIM:
        v -= 1
    return v

perf = {}
for name, col, _ in METHODS:
    perf[name] = {p: (math.inf if float(last[p][col]) >= LIM else value(p, col))
                  for p in probs}

# exclude instances whose best count is 0 (ratios undefined)
nontrivial = [p for p in probs if min(perf[m][p] for m, _, _ in METHODS) >= 1]
print(f"profile over {len(nontrivial)} of {len(probs)} instances "
      f"({len(probs)-len(nontrivial)} solved at the initial basis excluded)")

ratios = {}
for m, _, _ in METHODS:
    ratios[m] = {}
    for p in nontrivial:
        best = min(perf[mm][p] for mm, _, _ in METHODS)
        ratios[m][p] = perf[m][p] / best

TAU_MAX = 64.0
fig, ax = plt.subplots(figsize=(6.0, 4.0))
for m, _, ls in METHODS:
    rs = sorted(r for r in ratios[m].values() if not math.isinf(r))
    n = len(nontrivial)
    # step function points
    xs, ys = [1.0], [sum(1 for r in rs if r <= 1.0) / n]
    for r in rs:
        if r > 1.0 and r <= TAU_MAX:
            xs += [r, r]
            ys += [ys[-1], sum(1 for q in rs if q <= r) / n]
    xs.append(TAU_MAX)
    ys.append(ys[-1])
    ax.plot(xs, ys, ls, lw=1.6, label=m)
    print(f"{m:16s} rho(1)={ys[0]:.3f}  rho(2)={sum(1 for r in rs if r <= 2)/n:.3f}  "
          f"rho(4)={sum(1 for r in rs if r <= 4)/n:.3f}  rho(inf)={len(rs)/n:.3f}")

ax.set_xscale("log", base=2)
ax.set_xlim(1, TAU_MAX)
ax.set_ylim(0, 1.02)
ax.set_xlabel(r"performance ratio $\tau$ (iterations relative to best method, log scale)")
ax.set_ylabel(r"fraction of instances with ratio $\leq \tau$")
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT)
print("written:", OUT)
