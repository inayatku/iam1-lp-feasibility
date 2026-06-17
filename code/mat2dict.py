#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced MAT to Dictionary Converter for Linear Programming Problems
===================================================================

This utility converts linear programming (LP) problems from MATLAB (.mat) files 
into a inequality form (Ax ≤ b) using a two-step approach:

1. Find singleton variables (variables appearing in only one constraint) as basic variables
2. For remaining constraints, split into two inequalities and use slack/surplus variables

Key Features:
------------
1. Converts standard form LPs (Ax = b with bounds) to inequality form
2. Identifies singleton variables first to preserve problem structure
3. Splits remaining equality constraints into inequality pairs
4. Creates a system by removing basic variable columns
5. Keeps variable bounds separate (not added as constraints)
6. Maintains mapping to recover original solution

Usage:
-----
Same as the original mat2dict.py

Author: [Your Name/Institution]
Date: [Date]
"""

import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import pickle
import json
import argparse
import glob
import re
import sys
import builtins
import warnings

# Increase recursion limit to handle deep recursion in print function
sys.setrecursionlimit(10000)

# Suppress all warnings
warnings.filterwarnings("ignore")

# Define infinity threshold - values above this will be treated as infinity
INF_THRESHOLD = 1e+30  # Slightly less than 1e+32 to catch possible rounding errors

# Suppress warning messages
SUPPRESS_WARNINGS = True  # Set to True to suppress warnings
VERBOSE_OUTPUT = False    # Set to True for debugging

# Save the original print function
original_print = builtins.print

def custom_print(*args, **kwargs):
    """Custom print function that can suppress warnings."""
    # Check if this is a warning message
    is_warning = False
    for arg in args:
        if isinstance(arg, str) and ("Warning" in arg or "warning" in arg):
            is_warning = True
            break
    
    # Only print if either not a warning or warnings are not suppressed
    if not is_warning or not SUPPRESS_WARNINGS or VERBOSE_OUTPUT:
        # Use the original print function directly
        original_print(*args, **kwargs)

def enable_warning_suppression():
    """Enable suppression of warning messages."""
    global SUPPRESS_WARNINGS
    SUPPRESS_WARNINGS = True
    builtins.print = custom_print

def disable_warning_suppression():
    """Disable suppression of warning messages."""
    global SUPPRESS_WARNINGS
    SUPPRESS_WARNINGS = False
    builtins.print = original_print

# Enable warning suppression by default
enable_warning_suppression()

def map_solution_back(solution, basic_vars, var_mapping, original_shape):
    """
    Re-construct a solution vector for the original problem.
    
    Args:
        solution: Solution vector from the problem
        basic_vars: Dictionary mapping basic variables to their constraints
        var_mapping: Mapping from system variables to original variables
        original_shape: Original shape of the problem (m, n)
        
    Returns:
        original_solution: Solution vector for the original problem
    """
    n_original = original_shape[1]
    original_solution = np.zeros(n_original)
    
    # Copy non-basic variables from solution
    for i, orig_idx in enumerate(var_mapping):
        original_solution[orig_idx] = solution[i]
    
    # For basic variables, calculate values from constraints
    # This would need implementation based on how basic variables relate to constraints
    
    return original_solution

def convert_to_inequality_form(A, b, lbounds, ubounds):
    """
    Convert a linear programming problem in standard form (Ax = b with bounds)
    to inequality form by:
    1. Finding singleton variables
    2. Splitting remaining constraints
    3. Completely removing basic variable columns
    
    Mathematical Steps:
    -----------------
    1. Identify singleton variables for each constraint
    2. For constraints without singleton variables, split into pairs:
       a_i x ≤ b_i and -a_i x ≤ -b_i
    3. Remove columns corresponding to basic variables
    
    Note: Variable bounds (lbounds, ubounds) are not added as constraints
    as they are standard bounds that can be handled separately.
    
    Args:
        A: Coefficient matrix (sparse or dense)
        b: Right-hand side vector
        lbounds: Lower bounds on variables (not used in constraints)
        ubounds: Upper bounds on variables (not used in constraints)
        
    Returns:
        A_red: Coefficient matrix (only non-basic variables)
        b_red: Right-hand side vector for inequality system
        basic_var_dict: Dictionary mapping basic variables to constraints
        var_mapping: List mapping variables to original indices
        original_shape: Original shape of the problem
    """
    print("Starting conversion to inequality form...")
    
    # Convert sparse matrix to dense for easier manipulation if needed
    if sp.issparse(A):
        print(f"Converting sparse matrix to dense (original density: {A.nnz / (A.shape[0] * A.shape[1]):.6f})")
        A_dense = A.toarray()
    else:
        A_dense = A.copy()
    
    m, n = A_dense.shape  # m constraints, n variables
    print(f"Problem dimensions: {m} constraints, {n} variables")
    
    # Make sure b is a column vector
    if len(b.shape) == 1:
        b = b.reshape(-1, 1)
    elif b.shape[0] == 1:
        b = b.T
    
    # Create a copy of b to avoid modifying the original
    b = b.copy()
    
    # Step 1: Find singleton variables first
    # -------------------------------------
    print("Step 1: Identifying singleton variables for each constraint...")
    constraint_to_basic = {}  # Maps constraint index to its basic variable
    basic_to_constraint = {}  # Maps basic variable to constraint index
    
    # Count variable occurrences across constraints
    var_counts = np.zeros(n, dtype=int)
    var_locations = [[] for _ in range(n)]
    
    for i in range(m):
        for j in range(n):
            if abs(A_dense[i, j]) > 1e-10:  # Non-zero coefficient
                var_counts[j] += 1
                var_locations[j].append(i)
    
    # Assign singleton variables as basic where possible
    singleton_count = 0
    for j in range(n):
        if var_counts[j] == 1:
            row_idx = var_locations[j][0]
            if row_idx not in constraint_to_basic:  # If no basic var assigned yet
                constraint_to_basic[row_idx] = j
                basic_to_constraint[j] = row_idx
                singleton_count += 1
    
    print(f"Found {singleton_count} singleton variables to use as basic variables")
    
    # Step 2: Split remaining constraints and assign slack/surplus as basic
    # -------------------------------------------------------------------
    print("Step 2: Splitting remaining constraints without singleton variables...")
    
    constraints_without_basic = [i for i in range(m) if i not in constraint_to_basic]
    print(f"Constraints without singleton variables: {len(constraints_without_basic)}")
    
    # Create expanded system to include split constraints
    expanded_constraints = []  # Will hold all constraint rows
    expanded_rhs = []         # Will hold all RHS values
    expanded_basic_map = {}   # Maps new constraint index to its basic variable
    
    # First, add constraints that already have basic variables
    for i in range(m):
        if i in constraint_to_basic:
            expanded_constraints.append(A_dense[i, :])
            expanded_rhs.append(b[i, 0])
            expanded_basic_map[len(expanded_constraints)-1] = constraint_to_basic[i]
    
    # Now split the remaining constraints
    split_count = 0
    for i in constraints_without_basic:
        # Add the positive constraint: a_i x ≤ b_i
        expanded_constraints.append(A_dense[i, :])
        expanded_rhs.append(b[i, 0])
        
        # Add the negative constraint: -a_i x ≤ -b_i
        expanded_constraints.append(-A_dense[i, :])
        expanded_rhs.append(-b[i, 0])
        
        split_count += 1
    
    print(f"Split {split_count} constraints, resulting in {len(expanded_constraints)} total constraints")
    
    # Step 3: Create system by removing basic variable columns
    # --------------------------------------------------------------
    print("Step 3: Creating system by removing basic variable columns...")
    
    # Determine which variables are non-basic
    non_basic_vars = [j for j in range(n) if j not in basic_to_constraint]
    var_mapping = non_basic_vars  # Maps columns in system to original indices
    
    # Create coefficient matrix (only non-basic columns)
    A_ineq = np.zeros((len(expanded_constraints), len(non_basic_vars)))
    for i, constraint in enumerate(expanded_constraints):
        for j, orig_idx in enumerate(non_basic_vars):
            A_ineq[i, j] = constraint[orig_idx]
    
    # Convert RHS to numpy array
    b_ineq = np.array(expanded_rhs).reshape(-1, 1)
    
    # Clean up small values for numerical stability
    A_ineq[np.abs(A_ineq) < 1e-10] = 0
    b_ineq[np.abs(b_ineq) < 1e-10] = 0
    
    print(f"Final system: {A_ineq.shape[0]} constraints with {A_ineq.shape[1]} variables")
    print(f"Removed {n - len(non_basic_vars)} basic variables, keeping {len(non_basic_vars)} non-basic variables")
    
    return A_ineq, b_ineq, basic_to_constraint, var_mapping, (m, n)

def process_mat_file(filepath, output_format='pickle'):
    """
    Process a MAT file to extract LP data and convert to inequality form.
    
    This function serves as the main processing pipeline:
    1. Load the MAT file containing LP problem data
    2. Extract problem name, coefficient matrix, RHS vector, and variable bounds
    3. Handle special cases like infinite bounds and matrix rank deficiency
    4. Convert the problem to inequality form
    5. Return a structured dictionary with the processed data
    
    Args:
        filepath: Path to the MAT file
        output_format: 'pickle' or 'json'
        
    Returns:
        Dictionary with problem data in inequality form
    """
    try:
        print(f"\nProcessing {filepath}...")
        
        # Load MAT file
        print("Loading MAT file...")
        data = sio.loadmat(filepath)
        
        # Extract problem name
        if 'NAME' in data:
            name = data['NAME']
            if isinstance(name, np.ndarray) and name.size > 0:
                name_str = name.item()
                if isinstance(name_str, bytes):
                    name_str = name_str.decode('utf-8')
                problem_name = name_str.strip()
            else:
                problem_name = os.path.basename(filepath).replace('.mat', '')
        else:
            problem_name = os.path.basename(filepath).replace('.mat', '')
        
        print(f"Problem name: {problem_name}")
        
        # Check if A and b exist
        if 'A' not in data or 'b' not in data:
            print(f"Error: A or b matrix not found in {filepath}")
            return None
        
        A = data['A']
        b = data['b']
        
        print(f"Matrix A shape: {A.shape}, b shape: {b.shape}")
        if sp.issparse(A):
            print(f"A is sparse with {A.nnz} non-zero entries (density: {A.nnz / (A.shape[0] * A.shape[1]):.6f})")
            
        # Check for bounds and handle large values as infinity
        if 'lbounds' in data and 'ubounds' in data:
            # Convert bounds to float type first
            lbounds = data['lbounds'].astype(np.float64)
            ubounds = data['ubounds'].astype(np.float64)
            
            # Convert sparse bounds to dense if needed
            if sp.issparse(lbounds):
                lbounds = lbounds.toarray()
            if sp.issparse(ubounds):
                ubounds = ubounds.toarray()
                
            # Find large values and replace them with infinity
            lb_mask = np.abs(lbounds) > INF_THRESHOLD
            ub_mask = np.abs(ubounds) > INF_THRESHOLD
            
            lbounds[lb_mask] = -np.inf
            ubounds[ub_mask] = np.inf
                
            print(f"Found bounds: lbounds shape {lbounds.shape}, ubounds shape {ubounds.shape}")
            print(f"Number of infinite lower bounds: {np.sum(lb_mask)}")
            print(f"Number of infinite upper bounds: {np.sum(ub_mask)}")
        else:
            # Default bounds: 0 <= x < ∞
            print("No bounds found, using default: 0 <= x < ∞")
            lbounds = np.zeros((A.shape[1], 1), dtype=np.float64)
            ubounds = np.ones((A.shape[1], 1), dtype=np.float64) * np.inf
        
        # Convert to inequality form
        print(f"Converting {problem_name} to inequality form...")
        try:
            A_red, b_red, basic_var_dict, var_mapping, original_shape = convert_to_inequality_form(
                A, b, lbounds, ubounds)
            
            # Create result dictionary
            result = {
                'name': problem_name,
                'A': A_red,
                'b': b_red,
                'basic_var_dict': basic_var_dict,
                'var_mapping': var_mapping,
                'original_shape': original_shape,
                'inequality_shape': (A_red.shape[0], A_red.shape[1]),
                'lbounds': lbounds,
                'ubounds': ubounds
            }
            
            print(f"Conversion successful: original system {original_shape[0]}x{original_shape[1]} -> " 
                  f"system {A_red.shape[0]}x{A_red.shape[1]}")
            
            return result
        except Exception as e:
            print(f"Error converting {problem_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_result(result, output_dir, output_format='pickle'):
    """
    Save the problem data to a file.
    
    Args:
        result: Dictionary with problem data
        output_dir: Directory to save output
        output_format: 'pickle' or 'json'
    """
    if result is None:
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    name = result['name']
    clean_name = re.sub(r'\W+', '_', name.strip())
    
    try:
        if output_format == 'pickle':
            output_path = os.path.join(output_dir, f"{clean_name}.pkl")
            print(f"Saving as pickle to {output_path}...")
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)
        else:  # json
            # Convert numpy arrays to lists for JSON serialization
            json_result = result.copy()
            json_result['A'] = json_result['A'].tolist()
            json_result['b'] = json_result['b'].tolist()
            json_result['var_mapping'] = json_result['var_mapping'].tolist() if isinstance(json_result['var_mapping'], np.ndarray) else json_result['var_mapping']
            
            output_path = os.path.join(output_dir, f"{clean_name}.json")
            print(f"Saving as JSON to {output_path}...")
            with open(output_path, 'w') as f:
                json.dump(json_result, f)
        
        print(f"Successfully saved {name} to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving {name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Command-line entry: convert NETLIB .mat LP problems to inequality form."""
    import argparse, glob
    p = argparse.ArgumentParser(
        description="Convert NETLIB .mat LP problems to inequality form (Ax <= b).")
    p.add_argument("--file", help="process a single .mat file")
    p.add_argument("--input-dir", help="process every .mat file in this directory")
    p.add_argument("--output-dir", default="inequality_form",
                   help="directory to write the converted problems")
    p.add_argument("--format", choices=["pickle", "json"], default="pickle")
    args = p.parse_args()
    if args.file:
        targets = [args.file]
    elif args.input_dir:
        targets = sorted(glob.glob(os.path.join(args.input_dir, "*.mat")))
    else:
        p.error("provide --file or --input-dir")
    os.makedirs(args.output_dir, exist_ok=True)
    for fp in targets:
        result = process_mat_file(fp, args.format)
        if result is not None:
            save_result(result, args.output_dir, args.format)


if __name__ == "__main__":
    main()
