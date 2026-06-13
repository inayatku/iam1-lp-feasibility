"""Cross-check every numeric table in the manuscript against a results
directory (the archived 2025 CSVs or the regenerated 2026 CSVs).

Conventions established during the June-2026 recheck:
- Pan's method averages are over its SUCCESSFUL runs only (Avg_Iterations);
  IAM-1 / simplex / HiGHS solved every random instance, so for them
  Avg_Iterations == Avg_Iterations_All.
- Wall-clock timings are machine-dependent and are reported, not asserted.
- All NETLIB values come from netlib_batch_maxiter6000_results.csv over the
  52-problem set (54 distinct minus QAP15 [memory error] and MAROS-R7
  [conversion artifact]).
- Iteration counts in the paper mean pivots performed, uniformly for all
  methods. In the original_2025 and regenerated_2026 logs, Pan's recorded
  count additionally includes the initial feasibility check on solved runs
  (one more than pivots performed; capped runs record the pivot count).
  When checking against those directories this script subtracts 1 from
  Pan's logged values on solved runs to express them in the paper's
  convention; logs in regenerated_2026_pan_fixed already use it.

Usage:  python verify_paper_tables.py <results_dir>
Exit code 0 = all checks pass.
"""
import sys
import math
import os
import csv

RES = sys.argv[1] if len(sys.argv) > 1 else "../results/original_2025"

# Pan counting offset (see docstring): logs in these directories record
# Pan's count including the initial feasibility check on solved runs.
PAN_LEGACY_DIRS = {"original_2025", "regenerated_2026"}
PAN_OFF = 1 if os.path.basename(os.path.normpath(RES)) in PAN_LEGACY_DIRS else 0

FILES = {
    ("feasible", 0.0): [
        "lp_feasibility_comparison_feasible_sparsity0.0_tol1e-08_maxiter1000_seed43_problems30_results.csv",
        "lp_feasible_sparsity0.0_tol1e-08_seed43_resultspaper.csv",
    ],
    ("infeasible", 0.0): [
        "lp_feasibility_comparison_infeasible_sparsity0.0_tol1e-08_maxiter1000_seed43_problems30_results.csv",
    ],
    ("feasible", 0.5): [
        "lp_feasibility_comparison_feasible_sparsity0.5_tol1e-08_maxiter1000_seed43_problems30_results.csv",
        "lp_feasible_sparsity0.5_tol1e-08_seed43_resultspaper.csv",
    ],
    ("infeasible", 0.5): [
        "lp_feasibility_comparison_infeasible_sparsity0.5_tol1e-08_maxiter1000_seed43_problems30_results.csv",
        "lp_infeasible_sparsity0.5_tol1e-08_seed43_resultspaper.csv",
    ],
}

IAM, PAN = "IAM-1 Standard", "Pan's Method with Row Scaling"

def load(ptype, sparsity):
    for name in FILES[(ptype, sparsity)]:
        path = os.path.join(RES, name)
        if os.path.exists(path):
            with open(path, newline="") as f:
                rows = {r["Size"]: r for r in csv.DictReader(f)}
            return rows, name
    raise FileNotFoundError(f"no CSV for {ptype} sparsity {sparsity} in {RES}")

failures = []
checks = [0]

def check(label, expected, actual, tol=0.006):
    checks[0] += 1
    if abs(expected - actual) > tol * max(1.0, abs(expected)):
        failures.append(f"  MISMATCH {label}: paper={expected} csv={round(actual, 4)}")

SIZES = ["5x5","10x10","15x15","20x20","25x25","30x30","40x40","50x50","50x60","60x60"]

# Tables 2 & 3: {size: (iamF,iamI,panF,panI,simF,simI)}; Pan = success-only avg
# Pan values use the corrected (pivots-performed) convention.
T_ITERS = {
 0.0: {"5x5":(2.27,2.07,1.93,2.20,3.63,3.00), "10x10":(5.63,6.07,4.90,5.60,8.27,9.17),
  "15x15":(9.30,9.20,8.13,7.90,14.40,12.83), "20x20":(14.03,13.77,12.10,12.23,22.13,22.03),
  "25x25":(17.87,17.50,15.83,16.33,30.80,31.23), "30x30":(22.07,21.77,18.87,21.27,40.17,44.03),
  "40x40":(33.87,30.87,35.60,33.30,69.30,68.87), "50x50":(47.00,42.27,78.88,144.78,101.87,98.57),
  "50x60":(49.53,55.00,62.42,200.30,116.10,129.20), "60x60":(62.00,61.97,72.00,117.15,155.23,152.10)},
 0.5: {"5x5":(2.33,0.83,2.17,0.90,4.00,2.77), "10x10":(5.67,3.37,5.07,3.63,8.23,6.70),
  "15x15":(10.00,7.77,7.97,7.23,13.37,13.93), "20x20":(13.87,13.33,11.63,11.47,20.70,20.37),
  "25x25":(17.50,14.93,15.10,13.60,29.00,25.53), "30x30":(24.13,23.07,20.40,21.27,39.67,39.13),
  "40x40":(33.37,30.67,30.33,29.37,63.87,61.97), "50x50":(48.87,44.60,54.67,51.97,99.77,96.03),
  "50x60":(46.50,49.27,41.30,58.80,91.57,115.90), "60x60":(58.83,56.93,64.30,86.41,129.53,137.20)},
}
# Table 1 Pan success rates (dense) + sparse 60x60 from Sec. 5.2 text
T_SUCCESS_PAN = {0.0: {"50x50":(86.7,90.0), "50x60":(86.7,76.7), "60x60":(46.7,43.3)},
                 0.5: {"60x60":(90.0,90.0)}}
# Table 6 HiGHS 60x60: (dense F, dense I, sparse F, sparse I)
T_HIGHS_6060 = (56.40, 58.23, 55.17, 58.73)
# Table 6 ratios (recomputed June 2026, corrected Pan convention)
T_RATIOS = {0.0: (1.08, 1.65, 2.68), 0.5: (1.02, 1.32, 2.34)}

print(f"Results dir: {RES}\n")
for sp in (0.0, 0.5):
    for ptype, col in (("feasible", 0), ("infeasible", 1)):
        rows, used = load(ptype, sp)
        print(f"[{ptype} sparsity {sp}] using {used}")
        for size in SIZES:
            if size not in rows:
                failures.append(f"  MISSING row {size} in {used}")
                continue
            r = rows[size]
            exp = T_ITERS[sp][size]
            check(f"{size} {ptype} sp{sp} IAM iters", exp[0+col], float(r[f"{IAM}_Avg_Iterations_All"]))
            check(f"{size} {ptype} sp{sp} Pan iters", exp[2+col], float(r[f"{PAN}_Avg_Iterations"]) - PAN_OFF)
            check(f"{size} {ptype} sp{sp} Simplex iters", exp[4+col], float(r["Simplex_Avg_Iterations"]))
            check(f"{size} {ptype} sp{sp} IAM success", 100.0, float(r[f"{IAM}_Success_Rate"]))
        for size, pr in T_SUCCESS_PAN[sp].items():
            check(f"{size} {ptype} sp{sp} Pan success", pr[col], float(rows[size][f"{PAN}_Success_Rate"]))
        idx = (0 if sp == 0.0 else 2) + col
        check(f"HiGHS 60x60 {ptype} sp{sp}", T_HIGHS_6060[idx], float(rows["60x60"]["HiGHS_Avg_Iterations"]))

# Table 6 ratios from full precision (Pan success-only, consistent with table entries)
for sp in (0.0, 0.5):
    fr, _ = load("feasible", sp); ir, _ = load("infeasible", sp)
    h = (float(fr["60x60"]["HiGHS_Avg_Iterations"]) + float(ir["60x60"]["HiGHS_Avg_Iterations"])) / 2
    for (label, fcol), expect in zip(
            (("IAM", f"{IAM}_Avg_Iterations_All"), ("Pan", f"{PAN}_Avg_Iterations"),
             ("Simplex", "Simplex_Avg_Iterations")), T_RATIOS[sp]):
        off = PAN_OFF if label == "Pan" else 0
        m = (float(fr["60x60"][fcol]) + float(ir["60x60"][fcol])) / 2 - off
        check(f"Table 6 ratio {label} sp{sp}", expect, m / h, tol=0.005)

# Timing (Table 4) - informational only
fr, _ = load("feasible", 0.0); ir, _ = load("infeasible", 0.0)
print("\nTable 4 (ms/iter, dense) from this results dir - informational, machine-dependent:")
for size in ("20x20","30x30","40x40","50x50","60x60"):
    print(f"  {size}: IAM F={1000*float(fr[size][f'{IAM}_Time_Per_Iter']):.2f} I={1000*float(ir[size][f'{IAM}_Time_Per_Iter']):.2f} | "
          f"Pan F={1000*float(fr[size][f'{PAN}_Time_Per_Iter']):.2f} I={1000*float(ir[size][f'{PAN}_Time_Per_Iter']):.2f}")

# ----- NETLIB (Tables 7-9) -------------------------------------------------
NB = os.path.join(RES, "netlib_batch_maxiter6000_results.csv")
if os.path.exists(NB):
    with open(NB, newline="") as f:
        nrows = list(csv.DictReader(f))
    last = {}
    for r in nrows:
        last[r["problem"].replace("NAME", "").strip()] = r
    probs = sorted(n for n in last if n not in ("QAP15", "MAROS-R7"))
    check("NETLIB set size", 52, len(probs), tol=0)

    LIM = 6000

    # The archived NETLIB log records Pan's count including the initial
    # feasibility check on solved runs (see docstring); normalize to the
    # paper's pivots-performed convention here.
    def nval(name, col):
        v = float(last[name][col])
        if col == "pan_row_scaled_iterations" and v < LIM:
            v -= 1
        return v

    T7 = {"SCSD1":(10,70,71,25), "SHIP08S":(19,20,85,22), "BRANDY":(22,19,98,41),
          "SHIP12S":(37,24,157,28), "WOODW":(30,37,385,172), "FFFFF800":(54,7,379,196),
          "BNL1":(51,47,1128,92), "SCFXM3":(178,90,1140,104), "BNL2":(294,23,1606,590)}
    T8 = {"SCAGR7":(6000,98,113,118), "LOTFI":(6000,358,410,178), "SCAGR25":(6000,498,517,356),
          "BANDM":(6000,443,494,251), "DEGEN2":(1256,527,6000,582)}
    for name, (i1, pn, sx, hg) in T7.items():
        check(f"NETLIB {name} IAM-1", i1, nval(name, "iam1_iterations"), tol=0)
        check(f"NETLIB {name} Pan", pn, nval(name, "pan_row_scaled_iterations"), tol=0)
        check(f"NETLIB {name} Simplex", sx, nval(name, "simplex_iterations"), tol=0)
        check(f"NETLIB {name} HiGHS", hg, nval(name, "highs_simplex_iterations"), tol=0)
    for name, (pn, i1, sx, hg) in T8.items():
        check(f"NETLIB {name} Pan*", pn, nval(name, "pan_row_scaled_iterations"), tol=0)
        check(f"NETLIB {name} IAM-1*", i1, nval(name, "iam1_iterations"), tol=0)
        check(f"NETLIB {name} Simplex*", sx, nval(name, "simplex_iterations"), tol=0)
        check(f"NETLIB {name} HiGHS*", hg, nval(name, "highs_simplex_iterations"), tol=0)

    def gm(vals):
        return math.exp(sum(math.log(max(v, 1.0)) for v in vals) / len(vals))
    common = [n for n in probs if all(float(last[n][c]) < LIM for c in
              ("iam1_iterations", "pan_row_scaled_iterations", "simplex_iterations"))]
    check("NETLIB common-set size", 42, len(common), tol=0)
    # Table 9 (tab:netlib_overall) as revised June 2026
    T9 = {"iam1_iterations": (1, 24.5, 12.8), "pan_row_scaled_iterations": (7, 14.6, 12.6),
          "simplex_iterations": (6, 33.2, 26.7), "highs_simplex_iterations": (0, 29.4, 14.3)}
    for col, (nf, gms, gmc) in T9.items():
        vals = [nval(n, col) for n in probs]
        check(f"Table 9 {col} failures", nf, sum(1 for v in vals if v >= LIM), tol=0)
        check(f"Table 9 {col} gm successes", gms, gm([v for v in vals if v < LIM]), tol=0.005)
        check(f"Table 9 {col} gm common", gmc, gm([nval(n, col) for n in common]), tol=0.005)
else:
    print(f"\nNETLIB batch CSV not found at {NB} - NETLIB checks skipped")

print(f"\n{checks[0]} checks run.")
if failures:
    print(f"{len(failures)} MISMATCHES:")
    print("\n".join(failures))
    sys.exit(1)
print("ALL CHECKS PASS")
