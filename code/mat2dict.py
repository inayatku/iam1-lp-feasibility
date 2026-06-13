#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MAT to Dictionary Converter for Linear Programming Problems
==========================================================

This utility converts linear programming (LP) problems from MATLAB (.mat) files 
into inequality form (Ax ≤ b) suitable for optimization algorithms.

Key Features:
------------
1. Converts standard form LPs (Ax = b with bounds) to inequality form (Ax ≤ b)
2. Identifies and handles basic variables without reducing the problem dimensions
3. Adds variable bounds as explicit inequality constraints
4. Implements numerical stability techniques for handling large/small values
5. Provides both command-line and interactive menu interfaces

Usage:
-----
Command line:
    python mat2dict.py --input-dir /path/to/mat/files --output-dir /path/to/output
    python mat2dict.py --file /path/to/single/file.mat
    python mat2dict.py --menu  # Show interactive menu

Interactive menu:
    Select options to process single files, all files in a directory,
    or view information about converted problems.

Output:
------
The script generates pickle (.pkl) or JSON (.json) files containing:
- A: The coefficient matrix in inequality form
- b: The right-hand side vector
- basic_vars: List of identified basic variables
- original_shape: Dimensions of the original problem
- inequality_shape: Dimensions of the converted problem

Mathematical Background:
----------------------
In linear programming, a problem in standard form (Ax = b, l ≤ x ≤ u) can be
converted to inequality form (Ax ≤ b, x ≥ 0) by identifying basic variables
and manipulating the constraints. This implementation preserves the original
dimensions by zeroing out columns for basic variables rather than eliminating them.

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
import builtins  # Used to access the original print function
import warnings

# ... rest of the code remains unchanged ... 

def convert_to_inequality_form(A, b, lbounds, ubounds):
    """
    Convert a linear programming problem in standard form (Ax = b with bounds)
    to inequality form (Ax ≤ b) by keeping the original dimensions and zeroing out
    basic variable columns. Also adds variable bounds as explicit constraints.
    
    Mathematical Background:
    ----------------------
    The conversion process follows these mathematical principles:
    1. Identify a set of basic variables (one per constraint) that form a basis
    2. Perform Gaussian elimination to bring the system closer to reduced row echelon form
    3. Zero out the columns of basic variables to create an inequality form
    4. Add variable bounds as explicit inequality constraints
    
    This approach differs from traditional conversions that reduce the problem dimension
    by eliminating basic variables. Instead, we maintain the original variable space
    but zero out basic variable columns, which preserves the problem structure.
    
    Algorithm Steps:
    --------------
    1. Identify basic variables for each constraint
    2. Apply Gaussian elimination to normalize the system
    3. Zero out columns corresponding to basic variables
    4. Add variable bounds as explicit inequality constraints
    
    Args:
        A: Coefficient matrix (sparse or dense)
        b: Right-hand side vector
        lbounds: Lower bounds on variables
        ubounds: Upper bounds on variables
        
    Returns:
        A_ineq: Coefficient matrix for inequality form (original dimensions with zeroed columns)
        b_ineq: Right-hand side vector for inequality form
        basic_vars: List of basic variable indices
        original_shape: Original shape of the problem
    """
    print("Starting conversion to inequality form (no dimension reduction)...")
    
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
    
    # Step 1: Identify basic variables for each constraint
    # ----------------------------------------
    # For each constraint, we need to find a variable that can serve as a basic variable.
    # A good basic variable appears in only one constraint (singleton) or has a large coefficient
    # for numerical stability.
    print("Step 1: Identifying basic variables for each constraint...")
    basic_vars = [-1] * m  # Will store which variable is basic for each constraint
    
    # First, look for variables that appear in only one constraint (slack/surplus variables)
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
            if basic_vars[row_idx] == -1:  # If no basic var assigned yet
                basic_vars[row_idx] = j
                singleton_count += 1
    
    print(f"Found {singleton_count} singleton variables to use as basic variables")
    
    # Assign remaining basic variables (choose the one with largest absolute coefficient for numerical stability)
    remaining_assigned = 0
    for i in range(m):
        if basic_vars[i] == -1:  # If no basic var assigned yet
            # Find any non-zero coefficient (look for largest absolute value for numerical stability)
            best_j = -1
            max_abs_coef = 1e-10  # Minimum threshold
            for j in range(n):
                abs_coef = abs(A_dense[i, j])
                if abs_coef > max_abs_coef:
                    max_abs_coef = abs_coef
                    best_j = j
            
            if best_j >= 0:
                basic_vars[i] = best_j
                remaining_assigned += 1
    
    print(f"Assigned {remaining_assigned} additional basic variables")
    
    # Check if any constraints have no basic variables
    no_basic_count = basic_vars.count(-1)
    if no_basic_count > 0:
        print(f"Warning: {no_basic_count} constraints have no suitable basic variables (possible rank deficiency)")
    
    # Step 2: Gaussian elimination to create an "almost" reduced row echelon form
    # -----------------------------------------------------------------------
    # For each constraint with a basic variable, normalize the constraint by dividing by the 
    # coefficient of the basic variable, then eliminate that variable from all other constraints.
    # This brings the system closer to reduced row echelon form.
    print("Step 2: Applying Gaussian elimination...")
    # For each constraint, make its basic variable's coefficient zero in all other constraints
    gauss_success = 0
    gauss_skipped = 0
    
    for i in range(m):
        basic_var = basic_vars[i]
        if basic_var == -1:
            print(f"Warning: Could not find a suitable basic variable for constraint {i}")
            gauss_skipped += 1
            continue
        
        # Get the coefficient of the basic variable
        basic_coef = A_dense[i, basic_var]
        
        # Check for numerical issues
        if abs(basic_coef) < 1e-10:
            print(f"Warning: Very small coefficient ({basic_coef}) for basic variable in constraint {i}")
            gauss_skipped += 1
            continue
        
        # Normalize the row by the coefficient of the basic variable
        A_dense[i, :] = A_dense[i, :] / basic_coef
        b[i, 0] = b[i, 0] / basic_coef
        
        # Eliminate the basic variable from all other constraints
        for k in range(m):
            if k == i:
                continue  # Skip the current constraint
            
            # Get the coefficient of the basic variable in this constraint
            coef = A_dense[k, basic_var]
            
            if abs(coef) <= 1e-10:  # Basic variable not present or very small coefficient
                continue
            
            # Update the constraint: subtract coef * (normalized basic constraint)
            A_dense[k, :] = A_dense[k, :] - coef * A_dense[i, :]
            b[k, 0] = b[k, 0] - coef * b[i, 0]
        
        gauss_success += 1
    
    print(f"Gaussian elimination: {gauss_success} successful operations, {gauss_skipped} skipped")
    
    # Step 3: Create inequality form by zeroing columns of basic variables
    # -----------------------------------------------------------------
    # After Gaussian elimination, we zero out the columns for basic variables.
    # This effectively removes these variables from the problem while maintaining the original dimensions.
    print("Step 3: Creating inequality form by zeroing basic variable columns...")
    
    # Create a mask of basic variables
    basic_var_set = set(basic_vars)
    basic_var_set.discard(-1)  # Remove any -1 entries
    
    # Create a new coefficient matrix with the same dimensions as the original
    A_ineq = A_dense.copy()
    
    # Zero out the columns for basic variables
    for j in basic_var_set:
        A_ineq[:, j] = 0
    
    # Convert b to inequality form
    b_ineq = b.copy()
    
    # Step 4: Add variable bounds as explicit constraints
    # ------------------------------------------------
    # Add explicit inequality constraints for the lower and upper bounds
    # of non-basic variables. This completes the transformation to inequality form.
    print("Step 4: Adding variable bounds as constraints...")
    
    # Add bounds for all variables except basic variables
    lower_bounds_added = 0
    upper_bounds_added = 0
    
    # Prepare to add bounds as new rows to A_ineq and b_ineq
    bound_rows_A = []
    bound_rows_b = []
    
    for j in range(n):
        if j in basic_var_set:
            continue  # Skip basic variables
            
        # Lower bound: -xj <= -lb (if finite)
        if not np.isinf(lbounds[j, 0]) and abs(lbounds[j, 0]) < INF_THRESHOLD:
            bound_row = np.zeros(n)
            bound_row[j] = -1
            bound_rows_A.append(bound_row)
            bound_rows_b.append(-lbounds[j, 0])
            lower_bounds_added += 1
        
        # Upper bound: xj <= ub (if finite)
        if not np.isinf(ubounds[j, 0]) and abs(ubounds[j, 0]) < INF_THRESHOLD:
            bound_row = np.zeros(n)
            bound_row[j] = 1
            bound_rows_A.append(bound_row)
            bound_rows_b.append(ubounds[j, 0])
            upper_bounds_added += 1
    
    print(f"Added {lower_bounds_added} lower bounds and {upper_bounds_added} upper bounds as constraints")
    
    # Combine original constraints with bound constraints
    if bound_rows_A:
        A_ineq = np.vstack([A_ineq, np.array(bound_rows_A)])
        b_ineq = np.vstack([b_ineq, np.array(bound_rows_b).reshape(-1, 1)])
    
    # Final numerical cleaning and checks
    # ----------------------------------
    # Check for potential numerical issues in final result
    nan_count_A = np.isnan(A_ineq).sum()
    inf_count_A = np.isinf(A_ineq).sum() 
    nan_count_b = np.isnan(b_ineq).sum()
    inf_count_b = np.isinf(b_ineq).sum()
    
    if nan_count_A > 0 or inf_count_A > 0 or nan_count_b > 0 or inf_count_b > 0:
        print(f"Warning: Result contains {nan_count_A} NaN and {inf_count_A} Inf values in A")
        print(f"Warning: Result contains {nan_count_b} NaN and {inf_count_b} Inf values in b")
        print("Fixing by replacing with finite values...")
        
        # Replace any remaining NaN or Inf values
        A_ineq = np.nan_to_num(A_ineq, nan=0.0, posinf=1e20, neginf=-1e20)
        b_ineq = np.nan_to_num(b_ineq, nan=0.0, posinf=1e20, neginf=-1e20)
    
    # Clean up very small values (numerical stability)
    small_coeffs = np.sum(np.abs(A_ineq) < 1e-10)
    if small_coeffs > 0:
        print(f"Cleaning up {small_coeffs} small coefficients (< 1e-10) for numerical stability")
    A_ineq[np.abs(A_ineq) < 1e-10] = 0
    b_ineq[np.abs(b_ineq) < 1e-10] = 0
    
    print(f"Final result: {A_ineq.shape[0]} inequality constraints with {A_ineq.shape[1]} variables")
    print(f"Zeroed out {len(basic_var_set)} basic variable columns")
    
    return A_ineq, b_ineq, list(basic_var_set), (m, n) 

def main():
    """
    Main entry point for the MAT to Dictionary converter.
    
    This function parses command-line arguments and determines whether to:
    1. Process a single file
    2. Process all files in a directory
    3. Show an interactive menu
    
    Command-line arguments:
    ----------------------
    --input-dir:    Directory containing MAT files
    --output-dir:   Directory to save converted problems
    --format:       Output format (pickle or json)
    --file:         Process only the specified file
    --verbose:      Enable verbose output
    --menu:         Show interactive menu
    --show-warnings: Show warning messages
    """
    parser = argparse.ArgumentParser(description="Convert Netlib LP problems from MAT files to inequality form (Ax <= b)")
    parser.add_argument('--input-dir', default="J:/Research work/IAM-1/Netlib_files/MAT_Files", 
                        help="Directory containing MAT files")
    parser.add_argument('--output-dir', default="J:/Research work/IAM-1/Netlib_files/inequality_form", 
                        help="Directory to save converted problems")
    parser.add_argument('--format', choices=['pickle', 'json'], default='pickle',
                        help="Output format (default: pickle)")
    parser.add_argument('--file', help="Process only the specified file")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose output")
    parser.add_argument('--menu', action='store_true', help="Show interactive menu")
    parser.add_argument('--show-warnings', action='store_true', help="Show warning messages")
    args = parser.parse_args()
    
    # Configure verbosity and warning suppression
    global VERBOSE_OUTPUT
    if args.verbose:
        VERBOSE_OUTPUT = True
        print("Verbose mode enabled")
        
    if args.show_warnings:
        disable_warning_suppression()
        print("Warning messages enabled")
    else:
        enable_warning_suppression()
        
    # Configure verbosity
    if args.verbose:
        pass  # Keep verbose mode as is
    else:
        # Save the original print function
        orig_print = builtins.print
        
        # Define a custom print function that only prints important messages
        def custom_print(*args, **kwargs):
            # Don't print anything unless we're reopening stdout
            pass
        
        # Replace the print function with our custom version 
        builtins.print = custom_print
        
        # Define important message function
        def print_important(*args, **kwargs):
            # Temporarily restore original print function for important messages
            temp = builtins.print
            builtins.print = orig_print
            orig_print(*args, **kwargs)
            builtins.print = temp
        
        # Use our custom print_important function for the main functions
        globals()['print'] = print_important
    
    # Force console mode to ensure proper display of the menu
    if sys.stdout.isatty():
        # Running in a console
        pass
    else:
        # Not running in a console - may affect display
        print("Warning: Not running in a console. Menu display may be affected.")

    # Print a clear indicator that we're checking for menu display
    print("\n\n")
    print("*" * 80)
    print("CHECKING IF MENU SHOULD BE DISPLAYED...")
    
    # If menu option is specified or no other options are provided, show the menu
    if args.menu or (not args.file and args.input_dir == "J:/Research work/IAM-1/Netlib_files/MAT_Files"):
        print("Menu option triggered - showing interactive menu now.")
        print("*" * 80)
        print("\n\n")
        show_menu(args)
        return
    else:
        print("Menu option not triggered - proceeding with command line options.")
        print("*" * 80)
        print("\n\n")
    
    if args.file:
        # Process a single file
        print(f"Processing single file: {args.file}")
        result = process_mat_file(args.file, args.format)
        save_result(result, args.output_dir, args.format)
    else:
        # Process all MAT files in the directory
        mat_files = glob.glob(os.path.join(args.input_dir, "*.mat"))
        
        print(f"Found {len(mat_files)} MAT files in {args.input_dir}")
        
        for i, mat_file in enumerate(mat_files):
            print(f"\nProcessing file {i+1}/{len(mat_files)}: {os.path.basename(mat_file)}")
            result = process_mat_file(mat_file, args.format)
            save_result(result, args.output_dir, args.format)
    
    print(f"Conversion complete. Results saved to {args.output_dir}")

def show_menu(args):
    """
    Display an interactive menu for the mat2dict utility.
    
    This function provides a user-friendly interface with the following options:
    1. Process a single MAT file
    2. Process all MAT files in a directory
    3. Show summary of a converted problem
    4. List all available MAT files
    5. List all converted problem files
    6. Exit
    
    The interactive menu is particularly useful for users who prefer a GUI-like
    experience or are unfamiliar with command-line arguments.
    
    Args:
        args: Command-line arguments parsed by argparse
    """
    sys.__stdout__.write("\n\n\n")
    sys.__stdout__.write("#" * 80 + "\n")
    sys.__stdout__.write("INTERACTIVE MENU STARTED\n")
    sys.__stdout__.write("#" * 80 + "\n")
    
    while True:
        sys.__stdout__.write("\n" + "="*60 + "\n")
        sys.__stdout__.write(f"{'MAT TO INEQUALITY CONVERTER (WITH BOUNDS)':^60}\n")
        sys.__stdout__.write("="*60 + "\n")
        sys.__stdout__.write("This utility converts LP problems in MAT files to reduced inequality form by:\n")
        sys.__stdout__.write("  - Identifying and eliminating basic variables\n")
        sys.__stdout__.write("  - Creating a reduced system with only non-basic variables\n")
        sys.__stdout__.write("  - Adding variable bounds as explicit constraints\n")
        sys.__stdout__.write("  - Maintaining a mapping to track the original variables\n")
        sys.__stdout__.write("\nAvailable options:\n")
        sys.__stdout__.write("  1. Process a single MAT file\n")
        sys.__stdout__.write("  2. Process all MAT files in a directory\n")
        sys.__stdout__.write("  3. Show summary of a converted problem\n")
        sys.__stdout__.write("  4. List all available MAT files\n")
        sys.__stdout__.write("  5. List all converted problem files\n")
        sys.__stdout__.write("  6. Exit\n")
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            sys.__stdout__.write(f"You selected option: {choice}\n")
        except EOFError:
            sys.__stdout__.write("Error reading input. Terminal might not support interactive mode.\n")
            return
        except KeyboardInterrupt:
            sys.__stdout__.write("\nOperation cancelled by user.\n")
            return
        
        # Option 1: Process a single MAT file
        # -----------------------------------
        if choice == '1':
            try:
                file_path = input("Enter the path to the MAT file: ").strip()
                sys.__stdout__.write(f"You entered: {file_path}\n")
                if not os.path.exists(file_path):
                    sys.__stdout__.write(f"Error: File '{file_path}' does not exist.\n")
                    continue
                    
                result = process_mat_file(file_path, args.format)
                if result:
                    # Display the size reduction information
                    orig_shape = result['original_shape']
                    ineq_shape = result['inequality_shape']
                    vars_eliminated = orig_shape[1] - ineq_shape[1]
                    percent_reduction = (vars_eliminated / orig_shape[1]) * 100
                    
                    sys.__stdout__.write("\n" + "-"*60 + "\n")
                    sys.__stdout__.write(f"Problem: {result['name']}\n")
                    sys.__stdout__.write(f"Original size: {orig_shape[0]} constraints x {orig_shape[1]} variables\n")
                    sys.__stdout__.write(f"Reduced size: {ineq_shape[0]} constraints x {ineq_shape[1]} variables\n")
                    sys.__stdout__.write(f"Variables eliminated: {vars_eliminated} ({percent_reduction:.2f}%)\n")
                    sys.__stdout__.write("-"*60 + "\n")
                    
                    save_choice = input("Save the result? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        output_path = save_result(result, args.output_dir, args.format)
                        if output_path:
                            sys.__stdout__.write(f"Result saved to {output_path}\n")
            except Exception as e:
                sys.__stdout__.write(f"Error processing file: {str(e)}\n")
                
        # Option 2: Process all MAT files in a directory
        # ---------------------------------------------
        elif choice == '2':
            dir_path = input("Enter the directory containing MAT files: ").strip()
            if not os.path.isdir(dir_path):
                sys.__stdout__.write(f"Error: Directory '{dir_path}' does not exist.\n")
                continue
                
            mat_files = glob.glob(os.path.join(dir_path, '*.mat'))
            sys.__stdout__.write(f"Found {len(mat_files)} MAT files in {dir_path}\n")
            
            confirm = input(f"Process all {len(mat_files)} files? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
                
            results_summary = []
            
            for i, file_path in enumerate(mat_files):
                sys.__stdout__.write(f"\nProcessing file {i+1}/{len(mat_files)}: {os.path.basename(file_path)}\n")
                result = process_mat_file(file_path, args.format)
                if result:
                    save_result(result, args.output_dir, args.format)
                    
                    # Record summary information
                    orig_shape = result['original_shape']
                    ineq_shape = result['inequality_shape']
                    vars_eliminated = orig_shape[1] - ineq_shape[1]
                    percent_reduction = (vars_eliminated / orig_shape[1]) * 100
                    
                    results_summary.append({
                        'name': result['name'],
                        'original_size': orig_shape,
                        'reduced_size': ineq_shape,
                        'vars_eliminated': vars_eliminated,
                        'percent_reduction': percent_reduction
                    })
            
            # Display summary of all processed files
            sys.__stdout__.write("\n" + "="*80 + "\n")
            sys.__stdout__.write(f"{'SUMMARY OF PROCESSED FILES':^80}\n")
            sys.__stdout__.write("="*80 + "\n")
            sys.__stdout__.write(f"{'Problem Name':<30} {'Original Size':<20} {'Reduced Size':<20} {'Reduction %':<10}\n")
            sys.__stdout__.write("-"*80 + "\n")
            
            for summary in results_summary:
                orig = f"{summary['original_size'][0]}x{summary['original_size'][1]}"
                red = f"{summary['reduced_size'][0]}x{summary['reduced_size'][1]}"
                sys.__stdout__.write(f"{summary['name']:<30} {orig:<20} {red:<20} {summary['percent_reduction']:.2f}%\n")
            
            sys.__stdout__.write("="*80 + "\n")
            sys.__stdout__.write(f"Results saved to {args.output_dir}\n")
        
        # Other options (3-6)
        # ------------------
        # Process rest of the options...
        # [Rest of the show_menu function remains unchanged]

def process_mat_file(filepath, output_format='pickle'):
    """
    Process a MAT file to extract LP data and convert to inequality form.
    
    This function serves as the main processing pipeline:
    1. Load the MAT file containing LP problem data
    2. Extract problem name, coefficient matrix, RHS vector, and variable bounds
    3. Handle special cases like infinite bounds and matrix rank deficiency
    4. Convert the problem to inequality form
    5. Return a structured dictionary with the processed data
    
    Numerical Handling:
    -----------------
    - Values exceeding INF_THRESHOLD are treated as infinity
    - Very small values are cleaned up for numerical stability
    - NaN and Inf values are replaced with finite values
    - Matrix rank is checked for potential issues (for small matrices)
    
    Args:
        filepath: Path to the MAT file
        output_format: 'pickle' or 'json'
        
    Returns:
        Dictionary with problem data in inequality form containing:
        - name: Problem name
        - A: Coefficient matrix in inequality form
        - b: Right-hand side vector
        - basic_vars: List of identified basic variables
        - original_shape: Dimensions of the original problem
        - inequality_shape: Dimensions of the converted problem
    """
    try:
        print(f"\nProcessing {filepath}...")
        
        # STEP 1: Load MAT file
        # -------------------
        print("Loading MAT file...")
        data = sio.loadmat(filepath)
        
        # STEP 2: Extract problem name
        # --------------------------
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
        
        # STEP 3: Extract problem data (A and b)
        # ------------------------------------
        if 'A' not in data or 'b' not in data:
            print(f"Error: A or b matrix not found in {filepath}")
            return None
        
        A = data['A']
        b = data['b']
        
        print(f"Matrix A shape: {A.shape}, b shape: {b.shape}")
        if sp.issparse(A):
            print(f"A is sparse with {A.nnz} non-zero entries (density: {A.nnz / (A.shape[0] * A.shape[1]):.6f})")
            
        # STEP 4: Handle variable bounds
        # ----------------------------
        # Check for bounds and handle 1e+32 values as infinity
        if 'lbounds' in data and 'ubounds' in data:
            lbounds = data['lbounds'].copy()  # Create a copy to avoid modifying the original
            ubounds = data['ubounds'].copy()
            
            # Convert sparse bounds to dense if needed
            if sp.issparse(lbounds):
                lbounds = lbounds.toarray()
            if sp.issparse(ubounds):
                ubounds = ubounds.toarray()
                
            # Find large values and replace them with infinity
            large_lb_count = np.sum(np.abs(lbounds) > INF_THRESHOLD)
            large_ub_count = np.sum(np.abs(ubounds) > INF_THRESHOLD)
            
            # Convert values explicitly to avoid integer conversion issues
            lb_mask = np.abs(lbounds) > INF_THRESHOLD
            ub_mask = np.abs(ubounds) > INF_THRESHOLD
            
            if large_lb_count > 0:
                lbounds[lb_mask] = -np.inf
            
            if large_ub_count > 0:
                ubounds[ub_mask] = np.inf
                
            print(f"Found bounds: lbounds shape {lbounds.shape}, ubounds shape {ubounds.shape}")
            print(f"Treating values > {INF_THRESHOLD} as infinity")
            print(f"Replaced {large_lb_count} large lower bounds and {large_ub_count} large upper bounds with infinity")
        else:
            # Default bounds: 0 <= x < ∞
            print("No bounds found, using default: 0 <= x < ∞")
            lbounds = np.zeros((A.shape[1], 1))
            ubounds = np.ones((A.shape[1], 1)) * np.inf
        
        # STEP 5: Check rank of A matrix for numerical issues
        # ------------------------------------------------
        if A.shape[0] < 1000 and A.shape[1] < 1000:
            try:
                if sp.issparse(A):
                    dense_A = A.toarray()
                    # Calculate the rank with a tolerance to account for numerical issues
                    rank = np.linalg.matrix_rank(dense_A, tol=1e-10)
                else:
                    rank = np.linalg.matrix_rank(A, tol=1e-10)
                print(f"Matrix A rank: {rank} (out of {min(A.shape[0], A.shape[1])} possible)")
                if rank < A.shape[0]:
                    print(f"Warning: A is rank deficient. This may cause numerical issues.")
            except Exception as e:
                print(f"Could not compute rank: {str(e)}")
        
        # STEP 6: Convert to inequality form
        # --------------------------------
        print(f"Converting {problem_name} to inequality form...")
        try:
            A_ineq, b_ineq, basic_vars, original_shape = convert_to_inequality_form(A, b, lbounds, ubounds)
            
            # Verify the result doesn't contain NaN or Inf
            if np.any(np.isnan(A_ineq)) or np.any(np.isinf(A_ineq)) or np.any(np.isnan(b_ineq)) or np.any(np.isinf(b_ineq)):
                print(f"Warning: Result still contains NaN or Inf values even after fixing. Applying second fix...")
                A_ineq = np.nan_to_num(A_ineq, nan=0.0, posinf=1e20, neginf=-1e20)
                b_ineq = np.nan_to_num(b_ineq, nan=0.0, posinf=1e20, neginf=-1e20)
            
            # STEP 7: Create result dictionary
            # ------------------------------
            result = {
                'name': problem_name,
                'A': A_ineq,
                'b': b_ineq,
                'basic_vars': basic_vars,
                'original_shape': original_shape,
                'inequality_shape': (A_ineq.shape[0], A_ineq.shape[1])
            }
            
            print(f"Conversion successful: original system {original_shape[0]}x{original_shape[1]} -> reduced inequality system {A_ineq.shape[0]}x{A_ineq.shape[1]}")
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

def save_result(result, output_dir, format):
    # Implementation of save_result function
    pass

def disable_warning_suppression():
    # Implementation of disable_warning_suppression function
    pass

def enable_warning_suppression():
    # Implementation of enable_warning_suppression function
    pass

if __name__ == "__main__":
    main() 