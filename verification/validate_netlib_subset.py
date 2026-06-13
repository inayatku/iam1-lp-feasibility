"""Re-solve a subset of preprocessed NETLIB problems with the paper's solvers
and compare iteration counts against the May-2025 batch log
(netlib_batch_maxiter6000_results.csv). Validates that the archived log is
reproducible in the current environment.

Run from the paper's code/ directory:
    python ../verification/validate_netlib_subset.py [problem ...]
"""
import csv
import os
import pickle
import sys
import time

import numpy as np
import scipy.sparse as sp

CODE_DIR = os.environ.get("IAM_CODE_DIR", "code")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", CODE_DIR))

from iam1_original import solve_with_iam1
from simplex_phase_1 import solve_with_simplex_phase1
from Pans_method_row_scale import check_feasibility_pan_row_scaled

PKL_DIR = ("E:/Papers_research_work/Current Research Work/Linear_Programming_Research/"
           "Research_work_augment_old/Compuations_AIM-1_random_and_netlib_Lps/"
           "Netlib_files/inequality_form_splited")
LOG = os.path.join(os.path.dirname(__file__), "..", "results", "original_2025",
                   "netlib_batch_maxiter6000_results.csv")
MAX_ITER = 6000

problems = sys.argv[1:] or ["AFIRO", "SC50A", "SCSD1", "BRANDY", "SCAGR7", "LOTFI"]

with open(LOG, newline="") as f:
    log = {}
    for r in csv.DictReader(f):
        log[r["problem"].replace("NAME", "").strip()] = r

mismatch = 0
for name in problems:
    path = os.path.join(PKL_DIR, f"NAME_{name}_reduced.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    A = data["A"]
    if sp.issparse(A):
        A = A.toarray()
    A = np.asarray(A, dtype=float)
    b = np.asarray(data["b"], dtype=float).ravel()
    exp = log[name]

    t0 = time.time()
    r_iam = solve_with_iam1(A, b, max_iter=MAX_ITER)
    r_sx = solve_with_simplex_phase1(A, b, max_iter=MAX_ITER)
    r_pan = check_feasibility_pan_row_scaled(A, b, max_iter=MAX_ITER)
    dt = time.time() - t0

    got = (r_iam["iterations"], r_pan["iterations"], r_sx["iterations"])
    # The 2025 log records Pan's count including the initial feasibility
    # check on solved runs; the solvers report pivots performed.
    pan_log = int(exp["pan_row_scaled_iterations"])
    want = (int(exp["iam1_iterations"]),
            pan_log - 1 if pan_log < MAX_ITER else pan_log,
            int(exp["simplex_iterations"]))
    ok = got == want
    if not ok:
        mismatch += 1
    print(f"{name:10s} ({A.shape[0]}x{A.shape[1]}) "
          f"iam1={got[0]} (log {want[0]}) pan={got[1]} (log {want[1]}) "
          f"simplex={got[2]} (log {want[2]}) "
          f"statuses: {r_iam['status']}/{r_pan['status']}/{r_sx['status']} "
          f"[{dt:.1f}s] {'OK' if ok else '<-- MISMATCH'}")

print(f"\n{len(problems)-mismatch}/{len(problems)} problems match the 2025 log")
sys.exit(1 if mismatch else 0)
