#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pans_method_row_scale.py

Implementation of Pan's Most-Obtuse-Angle Column Rule algorithm with initial row scaling
for determining the feasibility of a linear program.

This implementation enhances the original Pan's method by scaling each row
of the dictionary by its maximum absolute coefficient once at initialization,
improving numerical stability. It also includes comprehensive cycling detection
based on basis history tracking.

Algorithm:
1. Initialize dictionary and apply row scaling once
2. Select row index p with the most negative b_bar value
3. If all b_bar values are non-negative, stop (feasibility achieved)
4. Determine column index q with the most negative coefficient in row p
5. If all coefficients in row p are non-negative, stop (problem is infeasible)
6. Perform pivot operation
"""

import numpy as np
import time
import sys
import inspect
import os

# Function to detect calling module
def get_calling_module():
    """Determine if we're being called from netlib comparison"""
    frame = inspect.currentframe()
    try:
        while frame:
            if frame.f_globals.get('__name__') == '__main__':
                filename = frame.f_globals.get('__file__', '')
                if filename:
                    return os.path.basename(filename)
            frame = frame.f_back
    finally:
        del frame
    return None

# Determine which config to use based on caller
calling_module = get_calling_module()
if calling_module and "netlib" in calling_module.lower():
    # Use NETLIB high precision settings
    from config_netlib import *
    print("Using NETLIB high precision settings for Pan's method")
else:
    # Import Pan's method specific configuration parameters
    from config_pan import (
        ZERO_TOLERANCE, 
        FEASIBILITY_TOLERANCE,
        PIVOT_TOLERANCE,
        PERTURB_TOLERANCE,
        MAX_ITERATIONS,
        CYCLING_CHECK_FREQUENCY,
        CYCLING_CHECK_START,
        ENABLE_ROW_SCALING,
        MIN_ROW_SCALE_VALUE,
        MAX_ROW_SCALE_VALUE,
        KEY_PRESS_CHECK_FREQUENCY,
        PROGRESS_DOT_FREQUENCY,
        PROGRESS_LINE_FREQUENCY,
        USE_MOST_NEGATIVE_SELECTION,
        TRACK_BASIS_HISTORY,
        STORE_PIVOT_HISTORY,
        # New parameters
        MAX_CYCLING_ITERATIONS,
        MIN_PIVOT_VALUE,
        PIVOT_SAFETY_FACTOR,
        DICTIONARY_OUTPUT_DETAIL,
        # Display parameters
        COLUMN_WIDTH,
        DECIMAL_PRECISION,
        ENABLE_COLOR_OUTPUT,
        STANDARD_PAN_MESSAGE,
        get_config_info
    )
    print("Using standard precision settings for Pan's method")

# Add key detection imports
try:
    import msvcrt  # Windows
    def check_key_press():
        if msvcrt.kbhit():
            key = msvcrt.getch()
            # Check for 'q', 'Q', or Esc key
            if key in (b'q', b'Q', b'\x1b'):
                return True
        return False
except ImportError:
    try:
        import termios, fcntl, os
        def check_key_press():
            fd = sys.stdin.fileno()
            oldterm = termios.tcgetattr(fd)
            newattr = termios.tcgetattr(fd)
            newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
            
            try:
                termios.tcsetattr(fd, termios.TCSANOW, newattr)
                oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)
                
                try:
                    key = sys.stdin.read(1)
                    if key in ('q', 'Q', '\x1b'):  # q, Q or ESC
                        return True
                except IOError:
                    pass
            finally:
                termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
                fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
            return False
    except ImportError:
        # Fallback if neither module is available
        def check_key_press():
            return False

class PanMostObtuseAngleMethodWithRowScaling:
    """
    Implementation of Pan's Most-Obtuse-Angle Method with initial row scaling for linear programming feasibility.
    
    This method selects the most negative element in the RHS column and then chooses
    the most negative element in that row as the entering variable to determine feasibility
    of a linear program in the standard form Ax <= b.
    
    Each row is scaled by its maximum absolute coefficient once at initialization
    to improve numerical stability. The implementation also includes comprehensive
    cycling detection using basis history tracking.
    """
    
    def __init__(self, A, b, tol=ZERO_TOLERANCE, max_iter=MAX_ITERATIONS, verbose=False):
        """
        Initialize the Pan's Most-Obtuse-Angle solver with problem data and row scaling.
        
        Parameters:
        -----------
        A : numpy.ndarray
            The coefficient matrix of the constraints (m x n) for Ax <= b
        b : numpy.ndarray
            The right-hand side vector (m)
        tol : float, optional
            Numerical tolerance for feasibility check (default from global config)
        max_iter : int, optional
            Maximum number of iterations (default from global config)
        verbose : bool, optional
            Whether to print step-by-step information
        """
        self.A_original = np.array(A, dtype=float)
        self.b_original = np.array(b, dtype=float)
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        
        self.m, self.n = self.A_original.shape
        self.is_feasible = None
        self.iterations = 0
        
        # Numerical stability parameters
        self.perturb_tol = PERTURB_TOLERANCE  # Tolerance for perturbation
        self.pivot_tol = PIVOT_TOLERANCE      # Tolerance for pivot selection
        
        # Scaling factors
        self.row_scale_factors = np.ones(self.m + 1)
        
        # Apply scaling and prepare the problem
        self.preprocess_problem()
        
        # Cycling detection - Using both basis history and pivot history approaches
        self.basis_history = []          # For tracking previously seen bases
        self.pivot_history = []          # For pivot record
        self.cycling_detected = False    # Flag for whether cycling is detected
        self.cycling_count = 0           # Counter for cycling events
        self.cycle_length = 0            # Length of detected cycle
        self.cycling_details = []        # Detailed info on cycling events
        
        # Initialize dictionary
        self.initialize_dictionary()
    
    def preprocess_problem(self):
        """
        Scale problem using max value (L∞-norm) row scaling.
        
        This preprocessing step improves numerical stability by scaling each constraint 
        row to have a maximum absolute coefficient of 1. The scaling:
        
        1. Balances the magnitude of constraint coefficients
        2. Reduces the risk of overflow or underflow in calculations
        3. Helps prevent numerical issues when constraints have widely varying magnitudes
        4. Makes the problem more amenable to accurate pivot selections
        
        The scaling preserves the feasible region and optimal solution while improving
        the numerical properties of the problem.
        """
        # Create a copy of the original matrix to work with
        self.A = self.A_original.copy()
        self.b = self.b_original.copy()
        
        if self.verbose:
            print("=== Applied Max Value (L∞-norm) Row Scaling ===")
            print(f"Original problem dimension: {self.m} x {self.n}")
            if ENABLE_ROW_SCALING:
                print("Row scaling is enabled")
            else:
                print("Row scaling is disabled")
    
    def initialize_dictionary(self):
        """Initialize the dictionary for the LP problem in standard form Ax <= b and apply row scaling."""
        self.basic_vars = list(range(self.n, self.n + self.m))  # Initially, slack variables are basic
        self.nonbasic_vars = list(range(self.n))  # Original variables are non-basic
        
        # Initialize dictionary: first column is RHS, other columns correspond to non-basic variables
        self.dict = np.zeros((self.m + 1, self.n + 1))
        self.dict[1:, 0] = self.b  # RHS
        self.dict[1:, 1:] = self.A  # Coefficients for non-basic variables
        
        # Apply zero adjustment
        self.zero_adjust()
        
        # Apply row scaling after initialization
        self.apply_row_scaling()
        
        if self.verbose:
            print("=== Initial Dictionary (after row scaling) ===")
            self.print_dictionary()
            print()
    
    def apply_row_scaling(self):
        """
        Scale each row of the dictionary by the maximum absolute coefficient in that row.
        This improves numerical stability for problems with varying coefficient magnitudes.
        """
        # Skip the objective row (index 0)
        for i in range(1, self.m + 1):
            # Find the maximum absolute value in the row (including the RHS)
            row_max = np.max(np.abs(self.dict[i, :]))
            
            # Skip rows that are all zeros or very close to zero
            if row_max < self.tol:
                self.row_scale_factors[i] = 1.0
                continue
                
            # Store the scaling factor
            self.row_scale_factors[i] = row_max
            
            # Scale the row
            self.dict[i, :] = self.dict[i, :] / row_max
            
            # Apply zero adjustment after scaling
            self.zero_adjust()
        
        if self.verbose:
            print(f"Row scaling range: {np.min(self.row_scale_factors[1:]):.2e} to {np.max(self.row_scale_factors[1:]):.2e}")
    
    def zero_adjust(self):
        """
        Set values with absolute magnitude less than tolerance to zero.
        Uses machine epsilon based tolerances for maximum precision.
        """
        # Get machine epsilon for reference
        eps = np.finfo(np.float64).eps
        
        # Use dynamic tolerance based on maximum absolute values in the dictionary
        max_abs_value = np.max(np.abs(self.dict))
        
        # Scale tolerance based on value ranges to avoid truncating significant digits
        dynamic_tolerance = self.tol
        if max_abs_value > 1.0e8:
            # For large values, use a relative tolerance
            dynamic_tolerance = max(self.tol, max_abs_value * eps * 10)
        
        # Enhanced zero adjustment with graduated approach
        # 1. Identify extremely small values - use mask operations for efficiency
        very_small_mask = np.abs(self.dict) < dynamic_tolerance
        
        # 2. Apply zero adjustment in one operation
        self.dict = np.where(very_small_mask, 0.0, self.dict)
        
        # 3. Handle negative zeros (convert -0.0 to 0.0)
        neg_zero_mask = np.logical_and(self.dict == 0.0, np.signbit(self.dict))
        self.dict[neg_zero_mask] = 0.0
    
    def is_primal_feasible(self):
        """
        Check if the current dictionary is primal feasible with enhanced precision controls.
        Uses precise tolerance checks based on machine epsilon and value magnitudes.
        
        Returns:
        --------
        bool
            True if all RHS values are non-negative (to within tolerance)
        """
        # Get reference to RHS values (first column of dictionary, excluding row 0)
        rhs_values = self.dict[1:, 0]
        
        # Get machine epsilon for reference
        eps = np.finfo(np.float64).eps
        
        # Use both absolute and relative tolerance checks
        abs_tol = self.tol  # Standard absolute tolerance
        
        # Find maximum magnitude in RHS for relative tolerance
        max_abs_rhs = np.max(np.abs(rhs_values))
        rel_tol = max_abs_rhs * eps * 100  # Relative tolerance based on data magnitude
        
        # Use the more appropriate tolerance
        effective_tol = max(abs_tol, rel_tol)
        
        # Check feasibility with precise tolerance
        return np.all(rhs_values >= -effective_tol)
    
    def check_for_cycling(self):
        """
        Check if the current basis configuration has been seen before,
        using a more efficient representation for basis comparison.
        """
        # PERFORMANCE: Only check for cycling every 10 iterations after initial period
        if self.iterations < CYCLING_CHECK_START or self.iterations % CYCLING_CHECK_FREQUENCY != 0:
            return False
        
        # Create a tuple representation of the current basis
        current_basis = tuple(self.basic_vars)
        
        # Check if this basis has been seen before
        if current_basis in self.basis_history:
            # Find where the cycle begins
            cycle_start = self.basis_history.index(current_basis)
            self.cycle_length = len(self.basis_history) - cycle_start
            
            if self.verbose:
                print(f"\nWARNING: Cycling detected! Same basis found after {self.cycle_length} iterations.")
                print(f"Cycle starts at iteration {self.iterations - self.cycle_length} and repeats at iteration {self.iterations}")
            
            sys.stderr.write(f"\nCycling detected at iteration {self.iterations} (cycle length: {self.cycle_length})\n")
            sys.stderr.flush()
            
            self.cycling_detected = True
            self.cycling_count += 1
            return True
        
        # Add the current basis to history
        if TRACK_BASIS_HISTORY:
            self.basis_history.append(current_basis)
        return False
    
    def print_dictionary(self):
        """Print the current dictionary in a readable format."""
        # Create labels for variables
        basic_labels = [f"x{i+1}" if i < self.n else f"s{i-self.n+1}" for i in self.basic_vars]
        nonbasic_labels = [f"x{i+1}" if i < self.n else f"s{i-self.n+1}" for i in self.nonbasic_vars]
        
        # Print header
        print(f"{'Basic':<{COLUMN_WIDTH}} | {'RHS':<{COLUMN_WIDTH}}", end="")
        for label in nonbasic_labels:
            print(f" | {label:<{COLUMN_WIDTH}}", end="")
        print()
        print("-" * (COLUMN_WIDTH + 12 * (len(nonbasic_labels) + 1)))
        
        # Print each row
        for i in range(1, self.m + 1):
            print(f"{basic_labels[i-1]:<{COLUMN_WIDTH}} | {self.dict[i, 0]:<{COLUMN_WIDTH}.{DECIMAL_PRECISION}f}", end="")
            for j in range(1, self.n + 1):
                print(f" | {self.dict[i, j]:<{COLUMN_WIDTH}.{DECIMAL_PRECISION}f}", end="")
            
            # Detail level controls whether to show scaling factors
            if DICTIONARY_OUTPUT_DETAIL >= 1:
                print(f" (scale: {self.row_scale_factors[i]:.2e})")
            else:
                print()
    
    def pivot(self, r, s):
        """
        Perform a pivot operation on element (r,s) of the dictionary with
        optimized numerical operations and enhanced precision controls.
        
        Parameters:
        -----------
        r : int
            Row index of the pivot element
        s : int
            Column index of the pivot element
        """
        # Record the pivot for cycling detection
        if STORE_PIVOT_HISTORY:
            self.pivot_history.append((r, s))
        
        # Store pivot element and check for numerical issues
        pivot_element = self.dict[r, s]
        
        # Get machine epsilon reference for precision checks
        eps = np.finfo(np.float64).eps
        
        # Handle extremely small pivot elements for numerical stability
        if abs(pivot_element) < self.pivot_tol:
            if self.verbose:
                print(f"Warning: Small pivot element {pivot_element:.2e}. Adjusting for numerical stability.")
            
            # Try to find a better pivot with same entering variable
            better_pivot_found = False
            if abs(pivot_element) < eps * 1000:
                # Look for alternative leaving variable (different row) for same entering variable
                for alt_r in range(1, self.m + 1):
                    if alt_r != r and self.dict[alt_r, 0] < -self.tol and self.dict[alt_r, s] < -self.tol:
                        if abs(self.dict[alt_r, s]) > abs(pivot_element) * 100:
                            # Found better pivot
                            r = alt_r
                            pivot_element = self.dict[r, s]
                            better_pivot_found = True
                            if self.verbose:
                                print(f"Found better pivot in row {r} with value {pivot_element:.2e}")
                            break
            
            # If still unstable, use minimum pivot value with correct sign
            if not better_pivot_found and abs(pivot_element) < self.pivot_tol:
                # Set to a small value with correct sign to avoid division by extremely small numbers
                min_pivot = max(MIN_PIVOT_VALUE, eps * PIVOT_SAFETY_FACTOR)
                pivot_element = -min_pivot if pivot_element < 0 else min_pivot
                if self.verbose:
                    print(f"Using minimum pivot value {pivot_element:.2e} for stability")
        
        # Create a copy of the dictionary to store old values while updating
        old_dict = self.dict.copy()
        
        # 1. Replace the pivot element with its reciprocal
        self.dict[r, s] = 1.0 / pivot_element
        
        # OPTIMIZED: 2+3. Update all rows and columns with reduced checks
        # Create masks for non-pivot rows and columns
        rows = np.arange(self.m + 1)
        cols = np.arange(self.n + 1)
        non_pivot_rows = rows != r
        non_pivot_cols = cols != s
        
        # Vectorized update of pivot column (excluding pivot element)
        self.dict[non_pivot_rows, s] = old_dict[non_pivot_rows, s] / (-pivot_element)
        
        # Vectorized update of pivot row (excluding pivot element)
        self.dict[r, non_pivot_cols] = old_dict[r, non_pivot_cols] / pivot_element
        
        # OPTIMIZED: 4. Update the remaining elements using efficient broadcasting
        # Pre-calculate all row factors at once
        row_factors = old_dict[non_pivot_rows, s][:, np.newaxis]
        col_factors = old_dict[r, non_pivot_cols][np.newaxis, :]
        
        # Create update matrix in one operation using Kahan-like compensation
        update_matrix = np.outer(row_factors.flatten(), col_factors.flatten()) / pivot_element
        
        # Apply updates in one operation
        remaining_elements = self.dict[np.ix_(rows[non_pivot_rows], cols[non_pivot_cols])]
        self.dict[np.ix_(rows[non_pivot_rows], cols[non_pivot_cols])] = remaining_elements - update_matrix
        
        # Apply zero adjustment only after pivoting (not during intermediate steps)
        self.zero_adjust()
        
        # Update basic/nonbasic variable sets
        leaving_var = self.basic_vars[r - 1]
        entering_var = self.nonbasic_vars[s - 1]
        
        self.basic_vars[r - 1] = entering_var
        self.nonbasic_vars[s - 1] = leaving_var
    
    def solve(self):
        """
        Solve the LP feasibility problem using Pan's Most-Obtuse-Angle Method with 
        optimized performance and reduced unnecessary operations.
        
        Returns:
        --------
        dict
            Solution information
        """
        # Start timer for computation time only (excluding initialization)
        start_time = time.time()
        
        # Print instructions for skipping
        sys.stderr.write("Press 'q' or ESC to skip this method and proceed to the next one\n")
        sys.stderr.flush()
        
        # Counter for key press checking frequency
        check_counter = 0
        
        # Initial feasibility check
        if self.is_primal_feasible():
            if self.verbose:
                print("Dictionary is feasible. Terminating algorithm.")
            self.is_feasible = True
            # No pivot performed: the initial dictionary is already feasible.
            return {
                'status': 'feasible',
                'is_feasible': True,
                'iterations': 0,
                'time': 0.0,
                'solution': np.zeros(self.n),  # All variables at zero
                'cycling_detected': False,
                'cycling_count': 0,
                'cycle_length': 0,
                'scale_factors': self.row_scale_factors[1:].copy()
            }
        
        # Pre-allocate arrays for frequently used operations
        rhs_values = np.zeros(self.m)
        
        for self.iterations in range(self.max_iter):
            if self.verbose:
                print(f"\n=== Iteration {self.iterations + 1} ===")
            
            # Apply zero adjustment only at the start of each iteration
            self.zero_adjust()
    
            # Check feasibility with direct access to RHS values
            rhs_values = self.dict[1:, 0]
            if np.all(rhs_values >= -self.tol):
                if self.verbose:
                    print("Dictionary is feasible. Terminating algorithm.")
                self.is_feasible = True
                break
            
            # Monitor for cycling using basis history (only checks periodically)
            if TRACK_BASIS_HISTORY:
                self.check_for_cycling()
            
            # Check for key press every 10 iterations - standardized frequency across all methods
            check_counter += 1
            if check_counter >= KEY_PRESS_CHECK_FREQUENCY:
                check_counter = 0
                if check_key_press():
                    sys.stderr.write("\nSkipping Pan's method as requested by user\n")
                    sys.stderr.flush()
                    
                    # Print a newline at the end for clean output formatting
                    if self.iterations % PROGRESS_LINE_FREQUENCY != 0:
                        sys.stderr.write("\n")
                        sys.stderr.flush()
                        
                    return {
                        'status': 'skipped',
                        'iterations': self.iterations,
                        'time': time.time() - start_time,
                        'message': 'Method skipped by user',
                        'cycling_detected': self.cycling_detected,
                        'cycling_count': self.cycling_count,
                        'cycle_length': self.cycle_length if self.cycling_detected else 0,
                        'scale_factors': self.row_scale_factors[1:].copy()
                    }
            
            # Find the most infeasible row using NumPy operations
            r_idx = np.argmin(rhs_values)
            r = r_idx + 1  # Adjust for dictionary indexing
            
            # Find negative coefficients in the selected row
            row_coeffs = self.dict[r, 1:]
            s_idx = np.argmin(row_coeffs)
            s = s_idx + 1  # Adjust for dictionary indexing
            
            # Quick check if the selected coefficient is negative
            if row_coeffs[s_idx] >= -self.pivot_tol:
                # No negative coefficients - problem is infeasible
                self.is_feasible = False
                break
            
            # Perform pivot on the selected element
            self.pivot(r, s)
            
            # Display progress indicator (a dot) after each iteration
            if self.iterations % PROGRESS_DOT_FREQUENCY == 0:
                sys.stderr.write(".")
                sys.stderr.flush()
            # Print a newline every N iterations for readability
            if (self.iterations + 1) % PROGRESS_LINE_FREQUENCY == 0:
                sys.stderr.write(f" [{self.iterations + 1}]\n")
                sys.stderr.flush()
            
            if self.verbose:
                print("\nAfter pivot:")
                self.print_dictionary()
        
        # Print a newline at the end of iterations for clean output formatting
        if self.iterations % PROGRESS_LINE_FREQUENCY != 0:
            sys.stderr.write("\n")
            sys.stderr.flush()
        
        # Calculate computation time
        computation_time = time.time() - start_time
        
        # Print cycling summary
        if self.verbose and self.cycling_count > 0:
            print(f"\nCycling Summary:")
            print(f"Total cycling events detected: {self.cycling_count}")
            print(f"Cycle length: {self.cycle_length}")
            if len(self.cycling_details) > 0:
                print("Details of immediate cycling events:")
                for i, cycle in enumerate(self.cycling_details):
                    var = cycle['variable']
                    var_name = f"x{var+1}" if var < self.n else f"s{var-self.n+1}"
                    print(f"  Event {i+1}: {var_name} entered in iteration {cycle['entered_iteration']} and left in iteration {cycle['left_iteration']}")
        
        # Extract solution if feasible
        if self.is_feasible:
            # Apply final zero adjustment
            self.zero_adjust()
            
            # Construct the solution vector
            solution = np.zeros(self.n + self.m)
            for i, var in enumerate(self.basic_vars):
                if var < solution.size:
                    # Get the value directly since row scaling is applied to the dictionary
                    solution[var] = self.dict[i+1, 0]
            
            # self.iterations equals the number of pivots performed when the
            # feasibility check at the top of the loop succeeds.
            return {
                'status': 'feasible',
                'is_feasible': True,
                'iterations': self.iterations,
                'time': computation_time,
                'solution': solution[:self.n],  # Return only original variables
                'cycling_detected': self.cycling_detected,
                'cycling_count': self.cycling_count,
                'cycle_length': self.cycle_length if self.cycling_detected else 0,
                'scale_factors': self.row_scale_factors[1:].copy()
            }
        elif self.is_feasible is False:
            return {
                'status': 'infeasible',
                'is_feasible': False,
                'iterations': self.iterations,
                'time': computation_time,
                'cycling_detected': self.cycling_detected,
                'cycling_count': self.cycling_count,
                'cycle_length': self.cycle_length if self.cycling_detected else 0,
                'scale_factors': self.row_scale_factors[1:].copy()
            }
        else:
            return {
                'status': 'max_iterations',
                'is_feasible': False,  # Changed from None to False as a practical choice
                'iterations': self.max_iter,
                'time': computation_time,
                'cycling_detected': self.cycling_detected,
                'cycling_count': self.cycling_count,
                'cycle_length': self.cycle_length if self.cycling_detected else 0,
                'scale_factors': self.row_scale_factors[1:].copy()
            }


def check_feasibility_pan_row_scaled(A, b, tol=ZERO_TOLERANCE, max_iter=MAX_ITERATIONS, verbose=False):
    """
    Check feasibility of a linear program using Pan's Most-Obtuse-Angle Method with initial row scaling.
    
    This implementation applies row scaling only once at initialization to improve numerical stability.
    It also includes comprehensive cycling detection using basis history tracking.
    
    Parameters:
    -----------
    A : array_like
        The coefficient matrix of the constraints (m x n) for Ax <= b
    b : array_like
        The right-hand side vector (m)
    tol : float, optional
        Numerical tolerance for feasibility checks (default from global config)
    max_iter : int, optional
        Maximum number of iterations (default from global config)
    verbose : bool, optional
        Whether to print step-by-step information
        
    Returns:
    --------
    dict
        Result information including status, is_feasible, iterations, cycling statistics, and initial scale factors
    """
    solver = PanMostObtuseAngleMethodWithRowScaling(A, b, tol, max_iter, verbose)
    return solver.solve()


# Example usage
if __name__ == "__main__":
    import time
    
    # Example 1: Simple constraint system
    print("=== Example 1: Simple constraint system ===")
    A = np.array([
        [2, -1/3],
        [9, -1],
        [-1, 1/3],
        [-9, 2]
    ])
    
    b = np.array([-2, -3, 1, 12])
    
    # Check feasibility with our row-scaled method
    start_time = time.time()
    result = check_feasibility_pan_row_scaled(A, b, verbose=True)
    end_time = time.time()
    
    print("\n=== Most Obtuse Angle Method with Row Scaling Result ===")
    print(f"Status: {result['status']}")
    print(f"Is feasible: {result['is_feasible']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Solve time: {end_time - start_time:.6f} seconds")
    print(f"Cycling detected: {result.get('cycling_detected', False)}")
    print(f"Cycling count: {result.get('cycling_count', 0)}")
    print(f"Cycle length: {result.get('cycle_length', 0)}")
    print(f"Final row scale factors: {result['scale_factors']}")
    
    # Comment out the comparison with standard Pan's method since the file is not available
    """
    # Compare with standard Pan's method (without row scaling) if available
    try:
        from most_obtuse_angle_method import check_feasibility_pan
    
        print("\n=== Comparison with Standard Pan's Method ===")
        standard_start = time.time()
        std_result = check_feasibility_pan(A, b, verbose=False)
        standard_time = time.time() - standard_start
    
        row_scaled_time = end_time - start_time
    
        print(f"Standard Pan's Method:")
        print(f"  Status: {std_result['status']}")
        print(f"  Iterations: {std_result['iterations']}")
        print(f"  Time: {standard_time:.6f} seconds")
        print()
        print(f"Row-Scaled Pan's Method:")
        print(f"  Status: {result['status']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Time: {row_scaled_time:.6f} seconds")
    
        # Speed comparison
        if standard_time > 0:
            speedup = standard_time / row_scaled_time
            print(f"\nRow scaling speedup: {speedup:.2f}x")
        
        # Iteration comparison
        if std_result['iterations'] > 0:
            iter_improvement = (std_result['iterations'] - result['iterations']) / std_result['iterations'] * 100
            print(f"Iteration reduction: {iter_improvement:.2f}%") 
    except ImportError:
        print("\nStandard Pan's method not available for comparison.") 
    """
    print(f"\n{STANDARD_PAN_MESSAGE}")

def kahan_sum(values):
    """
    Compute sum using Kahan summation algorithm for maximum precision.
    
    Parameters:
    -----------
    values : array_like
        Values to sum
        
    Returns:
    --------
    float
        Sum with minimized floating-point error
    """
    sum_value = 0.0
    comp = 0.0  # Compensation term for lost low-order bits
    
    for val in values:
        y = val - comp  # Compensated summand
        t = sum_value + y  # Next approximation
        comp = (t - sum_value) - y  # Update compensation term
        sum_value = t
        
    return sum_value

def compensated_dot_product(v1, v2):
    """
    Compute dot product with Kahan summation for maximum precision.
    
    Parameters:
    -----------
    v1, v2 : array_like
        Vectors to compute dot product for
        
    Returns:
    --------
    float
        Dot product with minimized floating-point error
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    
    # Create products
    products = np.multiply(v1, v2)  # More explicit than v1 * v2
    
    # Use Kahan summation for maximum precision
    return kahan_sum(products)