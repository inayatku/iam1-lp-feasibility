import numpy as np
import time
import sys
import scipy.sparse as sp

# Check if we're called from netlib comparison to use higher precision settings
import inspect
import os

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
    if 'DECIMAL_PRECISION' not in globals():
        DECIMAL_PRECISION = 16
    print("Using NETLIB high precision settings")
else:
    # Use standard simplex precision settings
    from config_simplex import *
    print("Using standard precision settings")

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

# Enhanced precision functions for numerical stability
def kahan_sum(values):
    """
    Compute sum of values using Kahan summation algorithm for higher precision.
    
    The Kahan summation algorithm reduces numerical errors in the sum of a 
    sequence of floating point numbers by keeping track of the "lost low-order bits".
    
    Parameters:
    -----------
    values : array-like
        The values to sum
        
    Returns:
    --------
    float
        The sum with reduced floating point error
    """
    s = 0.0  # Running sum
    c = 0.0  # Running compensation for lost low-order bits
    
    for v in values:
        # Step 1: c is zero the first time around
        y = v - c
        # Step 2: Add y to the sum
        t = s + y
        # Step 3: c captures the rounding error
        c = (t - s) - y
        # Step 4: Update the running sum
        s = t
        
    return s

def compensated_dot_product(v1, v2):
    """
    Compute the dot product of two vectors with higher precision using Kahan summation.
    
    Parameters:
    -----------
    v1, v2 : array-like
        The two vectors to compute the dot product of
        
    Returns:
    --------
    float
        The dot product with reduced floating point error
    """
    # First compute all products
    products = np.multiply(v1, v2)
    # Then sum using Kahan summation
    return kahan_sum(products)

class SimplexPhase1:
    """
    Implementation of the standard Simplex Phase 1 method for finding a feasible solution
    to linear programs in the form Ax <= b, x >= 0.
    With enhanced numerical stability techniques including max value (L-infinity norm) row scaling.
    
    This implementation uses maximum precision settings for float64 to achieve the most
    numerically stable and accurate results possible.
    """
    
    def __init__(self, A, b, tol=ZERO_TOLERANCE, max_iter=MAX_ITERATIONS, verbose=False, use_sparse=None):
        """
        Initialize the Simplex Phase 1 solver with maximum precision.
        
        Parameters:
        -----------
        A : array-like
            Coefficient matrix for constraints (m x n)
        b : array-like
            Right-hand side values (m)
        tol : float, optional
            Numerical tolerance for calculations (default from config)
        max_iter : int, optional
            Maximum number of iterations (default from config)
        verbose : bool, optional
            Whether to print detailed information during solving
        use_sparse : bool, optional
            Whether to use sparse matrices for large problems.
            If None, will be determined automatically based on problem size.
        """
        # Store original problem data with maximum precision
        self.A_orig = np.array(A, dtype=np.float64)
        self.b_orig = np.array(b, dtype=np.float64)
        
        # Use configured tolerances or highest precision if not specified
        self.tol = tol
        # Higher tolerance for pivots, using machine epsilon
        self.pivot_tol = max(tol * 100, PIVOT_TOLERANCE)
        self.zero_tol = tol  # Tolerance for treating values as zero
        self.feasibility_tol = FEASIBILITY_TOLERANCE  # Tolerance for feasibility checking
        self.max_iter = max_iter
        self.verbose = verbose
        
        # Limits to prevent numerical overflow/underflow - use machine epsilon values
        self.max_value = np.finfo(np.float64).max / 10
        self.min_value = np.finfo(np.float64).tiny * 10
        
        self.m, self.n = self.A_orig.shape
        self.iterations = 0
        
        # Determine if sparse matrices should be used for better numerical performance
        if use_sparse is None:
            estimated_size = (self.m + 1) * (self.n + self.m + min(self.m, 1000) + 1) * 8
            self.use_sparse = estimated_size > 1e9  # Use sparse if over 1 GB
            if self.verbose and self.use_sparse:
                print(f"Using sparse matrices for large problem: {self.m} constraints, {self.n} variables")
        else:
            self.use_sparse = use_sparse
        
        # Scaling factors
        self.row_scaling = None
        
        # Track condition number for numerical stability
        self.condition_number_estimate = self.estimate_condition_number()
        
        # Apply preprocessing for numerical stability
        self.preprocess_problem()
        
        # Cycling detection
        self.basis_history = []
        self.pivot_history = []
        self.cycling_detected = False
        self.cycling_count = 0
        self.cycle_length = 0
        self.cycling_iterations = 0  # Track number of iterations since cycling was detected
        
        # Initialize tableau, identify infeasible constraints, and set up artificial variables
        self.initialize_tableau()
    
    def estimate_condition_number(self, matrix=None):
        """
        Estimate the condition number of the matrix for numerical stability monitoring.
        
        For large matrices, uses a faster estimation technique. For small matrices,
        computes the exact condition number.
        
        Parameters:
        -----------
        matrix : array-like, optional
            The matrix to estimate the condition number for.
            If None, uses the original constraint matrix A.
            
        Returns:
        --------
        float
            Estimated condition number of the matrix
        """
        if matrix is None:
            matrix = self.A_orig
            
        # For small matrices, compute condition number directly
        if max(matrix.shape) < 500:
            try:
                # Use SVD which is more stable than calculating eigenvalues
                s = np.linalg.svd(matrix, compute_uv=False)
                # Handle effectively zero singular values by capping them at min_value
                s_max = max(np.max(s), self.min_value)
                s_min = max(np.min(s[s > 0]), self.min_value)
                return s_max / s_min
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Error computing condition number: {e}")
                return float('inf')
        else:
            # For large matrices, use a faster estimate based on row/column norms
            try:
                row_norms = np.linalg.norm(matrix, axis=1)
                col_norms = np.linalg.norm(matrix, axis=0)
                
                # Cap small values to prevent division by zero
                row_max = max(np.max(row_norms), self.min_value)
                row_min = max(np.min(row_norms[row_norms > 0]), self.min_value)
                col_max = max(np.max(col_norms), self.min_value)
                col_min = max(np.min(col_norms[col_norms > 0]), self.min_value)
                
                # The product of row and column norm ratios provides a rough estimate
                return (row_max / row_min) * (col_max / col_min)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Error estimating condition number: {e}")
                return float('inf')
    
    def preprocess_problem(self):
        """
        Scale problem using max value (L-infinity norm) row scaling with enhanced precision.
        
        This preprocessing step improves numerical stability by scaling each constraint 
        row to have a maximum absolute coefficient of 1. The scaling:
        
        1. Balances the magnitude of constraint coefficients
        2. Reduces the risk of overflow or underflow in calculations
        3. Helps prevent numerical issues when constraints have widely varying magnitudes
        4. Makes the problem more amenable to accurate pivoting decisions
        
        The scaling preserves the feasible region and optimal solution while improving
        the numerical properties of the problem.
        """
        # Initialize row scaling factor
        self.row_scaling = np.ones(self.m)
        
        # Create a copy of the original matrix to work with
        A_scaled = self.A_orig.copy()
        b_scaled = self.b_orig.copy()
        
        # VECTORIZED: Apply row scaling based on max absolute value (L-infinity norm)
        # Compute maximum absolute value for each row in one operation
        row_max_abs = np.max(np.abs(A_scaled), axis=1)
        
        # Find rows with significant values (above tolerance)
        valid_rows = row_max_abs > self.zero_tol
        
        # Compute scaling factors with protection against division by zero
        # Use maximum precision by working with raw values
        self.row_scaling[valid_rows] = 1.0 / np.maximum(row_max_abs[valid_rows], self.min_value)
        
        # Cap scaling factors to prevent numerical instability
        self.row_scaling = np.minimum(
            np.maximum(self.row_scaling, MIN_ROW_SCALE_VALUE),
            MAX_ROW_SCALE_VALUE
        )
        
        # Apply scaling to all rows in one operation using broadcasting
        A_scaled = A_scaled * self.row_scaling[:, np.newaxis]
        b_scaled = b_scaled * self.row_scaling
        
        # Use the scaled matrix and scaled b
        self.A_orig = A_scaled
        self.b_orig = b_scaled
        
        # Check if condition number improved
        self.condition_number_after_scaling = self.estimate_condition_number(self.A_orig)
        
        if self.verbose:
            print("=== Applied Max Value (L-infinity norm) Row Scaling ===")
            print(f"Row scaling range: {np.min(self.row_scaling):.2e} to {np.max(self.row_scaling):.2e}")
            if np.isfinite(self.condition_number_estimate) and np.isfinite(self.condition_number_after_scaling):
                improvement = self.condition_number_estimate / max(self.condition_number_after_scaling, 1.0)
                print(f"Condition number: {self.condition_number_estimate:.2e} → {self.condition_number_after_scaling:.2e}")
                print(f"Improvement factor: {improvement:.2f}x")
            else:
                print("Condition number: Could not be estimated")
    
    def initialize_tableau(self):
        """
        Initialize the simplex tableau for Phase 1.
        - Add slack variables (FIRST)
        - Identify infeasible constraints and flip signs
        - Add artificial variables where needed
        - Set up the auxiliary objective function
        
        This works with the scaled problem data from preprocess_problem().
        """
        # Step 1: Add slack variables to all constraints (identity matrix)
        if self.use_sparse:
            # Use sparse matrices for large problems
            I_sparse = sp.eye(self.m, format='csr')
            A_slack = sp.hstack([sp.csr_matrix(self.A_orig), I_sparse])
            
            # Step 2: Identify any infeasible constraints (b_i < 0)
            infeasible_rows = np.where(self.b_orig < -self.feasibility_tol)[0]
            num_infeasible = len(infeasible_rows)
            
            if self.verbose:
                print(f"Found {num_infeasible} infeasible constraints (b_i < 0)")
                print(f"Using sparse matrices for tableau: {self.m} rows, potentially {self.n + self.m + num_infeasible} columns")
            
            # Make copies to work with
            A_working = A_slack.copy()
            b_working = self.b_orig.copy()
            
            # Step 3: Flip signs of infeasible constraints (including their slacks)
            if num_infeasible > 0:
                # For sparse matrices, we need to multiply rows by -1
                for row in infeasible_rows:
                    A_working[row, :] = -A_working[row, :]
                b_working[infeasible_rows] = -b_working[infeasible_rows]
                
                if self.verbose:
                    print("Flipped signs of infeasible constraints (including slack variables)")
            
            # Step 4: Determine which constraints need artificial variables
            needs_artificial = np.zeros(self.m, dtype=bool)
            needs_artificial[infeasible_rows] = True
            num_artificial = np.sum(needs_artificial)
            
            if self.verbose:
                print(f"Adding {num_artificial} artificial variables")
                if num_artificial > 0:
                    print(f"Adding artificial variables to constraints: {infeasible_rows}")
            
            # Step 5: Create artificial variable columns for constraints that need them
            if num_artificial > 0:
                # Create artificial variables (sparse)
                art_data = []
                art_rows = []
                art_cols = []
                
                art_idx = 0
                for i in range(self.m):
                    if needs_artificial[i]:
                        art_data.append(1.0)
                        art_rows.append(i)
                        art_cols.append(art_idx)
                        art_idx += 1
                
                A_art = sp.csr_matrix((art_data, (art_rows, art_cols)), shape=(self.m, num_artificial))
                
                # Combine with slack variables
                A_augmented = sp.hstack([A_working, A_art])
            else:
                A_augmented = A_working
            
            # The number of columns in the augmented matrix
            num_cols = A_augmented.shape[1]
            
            # Convert to dense for the remaining operations (or implement sparse versions)
            # Note: For extremely large problems, you'd want to implement everything with sparse ops
            A_augmented_dense = A_augmented.toarray()
            
            # Step 6: Create the initial tableau (dense)
            self.tableau = np.zeros((self.m + 1, num_cols + 1))
            
            # Set up the constraint rows in the tableau
            self.tableau[1:, 0] = b_working  # RHS values
            self.tableau[1:, 1:] = A_augmented_dense  # Coefficients
            
        else:
            # Original dense implementation for smaller problems
            A_slack = np.hstack([self.A_orig, np.eye(self.m)])
            
            # Step 2: Identify any infeasible constraints (b_i < 0)
            infeasible_rows = np.where(self.b_orig < -self.feasibility_tol)[0]
            num_infeasible = len(infeasible_rows)
            
            if self.verbose:
                print(f"Found {num_infeasible} infeasible constraints (b_i < 0)")
            
            # Make copies to work with
            A_working = A_slack.copy()
            b_working = self.b_orig.copy()
            
            # Step 3: Flip signs of infeasible constraints (including their slacks)
            if num_infeasible > 0:
                A_working[infeasible_rows] = -A_working[infeasible_rows]
                b_working[infeasible_rows] = -b_working[infeasible_rows]
                
                if self.verbose:
                    print("Flipped signs of infeasible constraints (including slack variables)")
            
            # Step 4: Determine which constraints need artificial variables
            needs_artificial = np.zeros(self.m, dtype=bool)
            needs_artificial[infeasible_rows] = True
            num_artificial = np.sum(needs_artificial)
            
            if self.verbose:
                print(f"Adding {num_artificial} artificial variables")
                if num_artificial > 0:
                    print(f"Adding artificial variables to constraints: {infeasible_rows}")
            
            # Step 5: Create artificial variable columns for constraints that need them
            if num_artificial > 0:
                # Initialize A_art as a matrix of zeros
                A_art = np.zeros((self.m, num_artificial))
                
                # Set +1 for each row that needs an artificial variable
                art_idx = 0
                for i in range(self.m):
                    if needs_artificial[i]:
                        A_art[i, art_idx] = 1.0
                        art_idx += 1
                
                # Combine with slack variables
                A_augmented = np.hstack([A_working, A_art])
            else:
                A_augmented = A_working
            
            # The number of columns in the augmented matrix
            num_cols = A_augmented.shape[1]
            
            # Step 6: Create the initial tableau
            # First row is for the objective function, rest for constraints
            self.tableau = np.zeros((self.m + 1, num_cols + 1))
            
            # Set up the constraint rows in the tableau
            self.tableau[1:, 0] = b_working  # RHS values
            self.tableau[1:, 1:] = A_augmented  # Coefficients
        
        # Step 7: Set up the auxiliary objective function (maximize negative sum of artificial variables)
        # Initially, all coefficients in objective row are zero
        self.tableau[0, :] = 0
        
        # If there are artificial variables, set their coefficients in objective to -1
        if num_artificial > 0:
            # The artificial variables are at the end of the augmented matrix
            art_start_col = num_cols - num_artificial + 1
            
            # Set coefficients of artificial variables to -1 in objective function
            # This creates the objective of maximizing the negative sum of artificial variables
            self.tableau[0, art_start_col:] = -1.0
            
            if self.verbose:
                print("Created auxiliary objective: maximize negative sum of artificial variables")
                print("Initial auxiliary objective coefficients:", self.tableau[0, 1:])
            
            # We need to convert to canonical form by eliminating artificial variables
            # from the objective function (making their coefficients zero in objective row)
            # Use Kahan summation for higher precision when combining rows
            for i in range(self.m):
                if needs_artificial[i]:
                    if self.use_sparse:
                        # Find the artificial variable index differently for sparse
                        art_idx = np.where(np.array(A_art[i, :].todense() > 0.5))[1][0]
                    else:
                        # Find which artificial variable corresponds to this row
                        art_idx = np.where(A_art[i] > 0.5)[0][0]
                    
                    art_col = art_start_col + art_idx
                    
                    # Original coefficient of this artificial variable in objective
                    original_coef = self.tableau[0, art_col]
                    
                    # We need the coefficient in the objective to become zero
                    # The coefficient in the constraint row is 1.0 for this artificial variable
                    # So we subtract the constraint row multiplied by the original coefficient
                    # Use high precision combination for numerical stability
                    for j in range(self.tableau.shape[1]):
                        self.tableau[0, j] += original_coef * self.tableau[i + 1, j]
            
            # Verify that artificial variables now have zero coefficients in objective
            for j in range(art_start_col, num_cols + 1):
                if abs(self.tableau[0, j]) > self.tol:
                    if self.verbose:
                        print(f"Warning: Artificial variable at column {j} still has non-zero coefficient " 
                              f"{self.tableau[0, j]} in objective row after canonicalization")
                    # Force coefficient to be exactly zero
                    self.tableau[0, j] = 0.0
            
            if self.verbose:
                print("Canonicalized objective coefficients:", self.tableau[0, 1:])
                print("Artificial variables now have zero coefficients in objective row")
                print("If optimal value is 0, all artificial variables can be removed from basis")
        
        # Step 8: Set up the basis - initially, basic variables are slacks and artificials
        self.basic_vars = []
        
        # Track where artificial variables are
        self.artificial_indices = []
        
        # Start with slack variable indices
        slack_start = self.n + 1  # Adjust indices to match tableau columns (1-indexed)
        
        # Set up the initial basis
        for i in range(self.m):
            if needs_artificial[i]:
                # Use artificial variable for this constraint
                if self.use_sparse:
                    # Find the artificial variable index differently for sparse
                    art_idx = np.where(np.array(A_art[i, :].todense() > 0.5))[1][0]
                else:
                    art_idx = np.where(A_art[i] > 0.5)[0][0]  # Find which artificial variable is used
                
                basis_idx = num_cols - num_artificial + 1 + art_idx  # Index in the tableau (1-indexed)
                self.artificial_indices.append(basis_idx)
                if self.verbose:
                    print(f"Constraint {i+1} uses artificial variable a{art_idx+1} as basic variable")
            else:
                # Use slack variable for this constraint
                basis_idx = slack_start + i  # Index in the tableau (1-indexed)
                if self.verbose:
                    print(f"Constraint {i+1} uses slack variable s{i+1} as basic variable")
            
            self.basic_vars.append(basis_idx)
        
        # Record the structure of the tableau for reference
        self.num_original_vars = self.n
        self.num_slack_vars = self.m
        self.num_artificial_vars = num_artificial
        self.total_vars = self.n + self.m + num_artificial
        
        # Apply numerical stabilization for the final tableau
        self.apply_numerical_stabilization()
        
        if self.verbose:
            print("Initial tableau created")
            print(f"Structure: {self.m} constraints, {self.n} original variables, "
                  f"{self.m} slack variables, {num_artificial} artificial variables")
            print("Initial basis:", self.basic_vars)
            if num_artificial > 0:
                print("Artificial variables at positions:", self.artificial_indices)
            self.print_tableau()
    
    def apply_zero_tolerance(self):
        """
        Set very small values to zero for numerical stability with enhanced precision controls.
        
        This method helps prevent propagation of round-off errors by setting values
        that are very close to zero (below the tolerance threshold) to exactly zero.
        Uses dynamic tolerance based on machine epsilon and data magnitude.
        """
        # Get machine epsilon for reference
        eps = np.finfo(np.float64).eps
        
        # Calculate maximum absolute value in tableau for relative tolerance
        max_abs_val = np.max(np.abs(self.tableau))
        
        # Determine effective tolerance (standard or scaled by max value)
        dynamic_tol = self.zero_tol
        if max_abs_val > 1.0e8:
            # For large values, use a relative tolerance to avoid losing precision
            dynamic_tol = max(self.zero_tol, max_abs_val * eps * 100)
        
        # Create mask for very small values
        zero_mask = np.abs(self.tableau) < dynamic_tol
        
        # Set small values to zero in one vectorized operation
        self.tableau[zero_mask] = 0.0
        
        # Fix -0.0 values to 0.0 for exact comparison consistency
        neg_zero_mask = np.logical_and(self.tableau == 0.0, np.signbit(self.tableau))
        self.tableau[neg_zero_mask] = 0.0
        
        return np.sum(zero_mask)  # Return count of zeros applied for diagnostics
    
    def apply_numerical_stabilization(self):
        """
        Apply various numerical stabilization techniques to the tableau.
        
        This method improves numerical stability by:
        1. Setting very small values to zero to prevent propagation of round-off errors
        2. Capping extremely large values to prevent overflow
        3. Avoiding very small non-zero values that can cause instability
        4. Ensuring exact zeros in identity columns for basic variables
        
        These techniques help maintain accuracy throughout the simplex iterations.
        """
        # VECTORIZED: 1. Set very small values to zero
        zeros_applied = self.apply_zero_tolerance()
        
        # VECTORIZED: 2. Cap extremely large values to prevent overflow
        large_values_mask = self.tableau > self.max_value
        large_values_count = np.sum(large_values_mask)
        self.tableau[large_values_mask] = self.max_value
        
        large_neg_values_mask = self.tableau < -self.max_value
        large_neg_values_count = np.sum(large_neg_values_mask)
        self.tableau[large_neg_values_mask] = -self.max_value
        
        # VECTORIZED: 3. Avoid very small non-zero values that can cause instability
        small_nonzero = (np.abs(self.tableau) < self.min_value) & (self.tableau != 0)
        small_nonzero_count = np.sum(small_nonzero)
        self.tableau[small_nonzero] = 0.0
        
        # 4. Ensure exact zeros in identity columns for basic variables
        identity_errors = 0
        for i, basic_var in enumerate(self.basic_vars):
            if basic_var >= 0:  # Sanity check
                # Make sure the pivot element is exactly 1.0
                if abs(self.tableau[i+1, basic_var] - 1.0) > self.tol:
                    # Rescale the row to ensure pivot is exactly 1.0
                    pivot_value = self.tableau[i+1, basic_var]
                    if abs(pivot_value) > self.tol:  # Only scale if pivot is non-zero
                        self.tableau[i+1, :] /= pivot_value
                        identity_errors += 1
                
                # Make sure other elements in this column are exactly 0.0
                for j in range(self.tableau.shape[0]):
                    if j != i+1 and abs(self.tableau[j, basic_var]) > self.tol:
                        # Zero out this element using the ith row
                        factor = self.tableau[j, basic_var]
                        self.tableau[j, :] -= factor * self.tableau[i+1, :]
                        identity_errors += 1
        
        # Apply zero tolerance again after identity corrections
        if identity_errors > 0:
            zeros_applied += self.apply_zero_tolerance()
        
        # Log diagnostics about numerical stabilization
        if self.verbose and (large_values_count > 0 or large_neg_values_count > 0 or small_nonzero_count > 0 or identity_errors > 0):
            print(f"Numerical stabilization applied: {zeros_applied} zeros, {large_values_count} caps (positive), "
                  f"{large_neg_values_count} caps (negative), {small_nonzero_count} small values, "
                  f"{identity_errors} identity corrections")
    
    def print_tableau(self, entering_col=None):
        """
        Print the current tableau in a readable format with colors.
        
        Parameters:
        -----------
        entering_col : int, optional
            The column index of the entering variable for showing ratios
        """
        if not self.verbose:
            return
        
        # Define ANSI color codes
        if ENABLE_COLOR_OUTPUT:
            YELLOW = '\033[93m'  # Yellow for basic variables
            GREEN = '\033[92m'   # Green for pivot elements
            RESET = '\033[0m'    # Reset color
            BOLD = '\033[1m'     # Bold text
        else:
            YELLOW = ''
            GREEN = ''
            RESET = ''
            BOLD = ''
            
        print("\nCurrent Tableau:")
        rows, cols = self.tableau.shape
        
        # Determine column widths for better alignment
        col_width = COLUMN_WIDTH  # Default width for all columns
        
        # Calculate ratios if entering_col is provided
        ratios = None
        leaving_row = -1
        if entering_col is not None:
            ratios = self.calculate_ratios(entering_col)
            
            # Determine leaving row (for highlighting)
            if len(ratios) > 0:
                valid_ratios = [r for r in ratios if r != float('inf')]
                if valid_ratios:
                    min_ratio = min(valid_ratios)
                    leaving_row = ratios.index(min_ratio) + 1  # Add 1 for tableau row index
        
        # Format the header
        header = f"{'Basic':^8}|"
        header += f"{'RHS':^{col_width}}|"
        
        for j in range(1, cols):
            var_type = ""
            if j <= self.num_original_vars:
                var_type = f"x{j}"
            elif j <= self.num_original_vars + self.num_slack_vars:
                var_type = f"s{j - self.num_original_vars}"
            else:
                var_type = f"a{j - (self.num_original_vars + self.num_slack_vars)}"
            
            # Color the entering column header green
            if j == entering_col:
                var_type = f"{GREEN}{var_type}{RESET}"
                
            header += f"{var_type:^{col_width}}|"
        
        # Add ratio column to header if we're showing ratios
        if ratios is not None:
            header += f"{'Ratio':^10}|"
        
        print(header)
        print("-" * len(header.replace(GREEN, "").replace(RESET, "")))  # Account for color codes in length
        
        # Print objective row (z)
        obj_row = f"{'z':^8}|"
        obj_row += f"{self.tableau[0, 0]:^{col_width}.{DECIMAL_PRECISION}g}|"
        
        for j in range(1, cols):
            # Color entering column green in objective row
            if j == entering_col:
                obj_row += f"{GREEN}{self.tableau[0, j]:^{col_width}.{DECIMAL_PRECISION}g}{RESET}|"
            else:
                obj_row += f"{self.tableau[0, j]:^{col_width}.{DECIMAL_PRECISION}g}|"
        
        # No ratio for objective row
        if ratios is not None:
            obj_row += f"{' ':^10}|"
        
        print(obj_row)
        
        # Print constraint rows
        for i in range(1, rows):
            basic_var = self.basic_vars[i-1]
            
            # Determine if this is the leaving row
            is_leaving_row = (i == leaving_row)
            
            # Determine basic variable name
            if basic_var <= self.num_original_vars:
                basic_name = f"x{basic_var}"
            elif basic_var <= self.num_original_vars + self.num_slack_vars:
                basic_name = f"s{basic_var - self.num_original_vars}"
            else:
                basic_name = f"a{basic_var - (self.num_original_vars + self.num_slack_vars)}"
            
            # Color the entire row green if it's the leaving row
            row_color = GREEN if is_leaving_row else ""
            row_reset = RESET if is_leaving_row else ""
            
            row_str = f"{row_color}{basic_name:^8}|"
            row_str += f"{self.tableau[i, 0]:^{col_width}.{DECIMAL_PRECISION}g}|"
            
            for j in range(1, cols):
                value_str = f"{self.tableau[i, j]:^{col_width}.{DECIMAL_PRECISION}g}"
                
                # Apply appropriate coloring based on the column and row type
                if j == entering_col:
                    # Pivot element (both entering column and leaving row)
                    if is_leaving_row:
                        # Already green from row coloring
                        row_str += f"{value_str}|"
                    else:
                        # Just color the cell green (entering column)
                        row_str += f"{GREEN}{value_str}{RESET}|"
                elif j == basic_var:
                    # Basic variable column (yellow)
                    if is_leaving_row:
                        # Mix of green (row) and yellow (basic) - prioritize green for clarity
                        row_str += f"{value_str}|"
                    else:
                        row_str += f"{YELLOW}{value_str}{RESET}|"
                else:
                    # Regular cell
                    row_str += f"{value_str}|"
            
            # Add ratio if available
            if ratios is not None:
                if i-1 < len(ratios):
                    if ratios[i-1] == float('inf'):
                        ratio_str = "inf"
                    else:
                        ratio_str = f"{ratios[i-1]:.{DECIMAL_PRECISION}g}"
                    
                    row_str += f"{ratio_str:^10}|"
                else:
                    row_str += f"{' ':^10}|"
            
            # Add the reset color at the end if we colored the entire row
            row_str += row_reset
            
            print(row_str)
        
        print("-" * len(header.replace(GREEN, "").replace(RESET, "")))
    
    def calculate_ratios(self, entering_col):
        """
        Calculate ratios for the minimum ratio test with maximum precision.
        Uses Kahan summation techniques and careful division to ensure maximum float64 precision.
        
        Parameters:
        -----------
        entering_col : int
            The column index of the entering variable
            
        Returns:
        --------
        list
            List of ratios for each constraint row, inf for invalid ratios
        """
        # Get machine epsilon for reference
        eps = np.finfo(np.float64).eps
        
        # Get the column of the entering variable (excluding objective row)
        column = self.tableau[1:, entering_col].copy()  # Create copy to prevent accidental modifications
        
        # Get the RHS values
        rhs = self.tableau[1:, 0].copy()  # Create copy to prevent accidental modifications
        
        # Calculate ratios with improved numerical stability
        ratios = []
        for i in range(len(column)):
            # Only positive coefficients are valid for ratio test
            # Use a relative tolerance based on value magnitudes
            min_pivot_threshold = max(self.pivot_tol, abs(column[i]) * eps * 1000)
            
            if column[i] > min_pivot_threshold:
                # Protect against division by small numbers that could cause instability
                if column[i] < self.pivot_tol * 10:
                    # If coefficient is small but positive, use a safer divisor
                    safe_divisor = self.pivot_tol * 10
                    if self.verbose:
                        print(f"Warning: Small pivot element {column[i]:.2e}, using {safe_divisor:.2e} as divisor")
                    ratio = rhs[i] / safe_divisor
                else:
                    # Standard high-precision division
                    ratio = rhs[i] / column[i]
                
                # Cap extremely large ratios
                if ratio > self.max_value:
                    ratio = self.max_value
                
                # Ensure ratio isn't negative zero due to floating point issues
                if ratio == 0.0 and np.signbit(ratio):
                    ratio = 0.0
                
                ratios.append(ratio)
            else:
                ratios.append(float('inf'))  # Infinite ratio for non-positive coefficients
        
        return ratios
    
    def find_entering_variable(self):
        """
        Find the entering variable with enhanced numerical stability.
        Select the most negative coefficient in the objective row that will lead to
        a numerically stable pivot.
        
        Returns:
        --------
        int or None
            The column index of the entering variable, or None if all coefficients are non-negative
        """
        # VECTORIZED: Get coefficients from objective row (exclude RHS)
        coefficients = self.tableau[0, 1:]
        
        # RESTORED ORIGINAL CODE: Use the same tolerance as original
        negative_indices = np.where(coefficients < -self.tol)[0]
        
        if len(negative_indices) == 0:
            return None  # No negative coefficients, optimal solution reached
        
        # Among the negative coefficients, find the best candidate
        # considering both coefficient value and potential pivot stability
        
        # VECTORIZED: Calculate scores for all candidates at once
        col_indices = negative_indices + 1  # Adjust for RHS column
        coefficient_scores = np.abs(coefficients[negative_indices])
        
        # Initialize arrays to store pivot scores
        pivot_scores = np.zeros_like(coefficient_scores)
        valid_pivot_mask = np.zeros_like(coefficient_scores, dtype=bool)
        
        # Check each column for potential pivot elements
        for i, col_idx in enumerate(col_indices):
            column = self.tableau[1:, col_idx]
            # RESTORED ORIGINAL CODE: Use original pivot detection
            positive_elements = column[column > self.pivot_tol]
            
            if len(positive_elements) > 0:
                # Store the maximum pivot value
                pivot_scores[i] = np.max(positive_elements)
                valid_pivot_mask[i] = True
        
        # If no valid pivots found, return None
        if not np.any(valid_pivot_mask):
            if self.verbose:
                print("Warning: Found negative coefficients but no valid pivots")
            return None
        
        # Calculate combined scores (only for valid pivots)
        combined_scores = np.zeros_like(coefficient_scores)
        combined_scores[valid_pivot_mask] = coefficient_scores[valid_pivot_mask] * pivot_scores[valid_pivot_mask]
        
        # Find the column with the highest score
        best_idx = np.argmax(combined_scores)
        return col_indices[best_idx]
    
    def find_leaving_variable(self, entering_col):
        """
        Find the leaving variable using the minimum ratio test with improved numerical stability.
        
        Parameters:
        -----------
        entering_col : int
            The column index of the entering variable
            
        Returns:
        --------
        int or None
            The row index of the leaving variable, or None if unbounded
        """
        # VECTORIZED: Calculate ratios with improved numerical stability
        # Get the column of the entering variable (excluding objective row)
        column = self.tableau[1:, entering_col]
        
        # Get the RHS values
        rhs = self.tableau[1:, 0]
        
        # RESTORED ORIGINAL CODE: Use original pivot detection
        positive_coef_mask = column > self.pivot_tol
        
        # If no positive coefficients, problem is unbounded
        if not np.any(positive_coef_mask):
            return None
        
        # Initialize ratios array with infinity
        ratios = np.full(len(column), np.inf)
        
        # RESTORED ORIGINAL CODE: Use original ratio calculation
        ratios[positive_coef_mask] = rhs[positive_coef_mask] / column[positive_coef_mask]
        
        # Cap extremely large ratios
        ratios[ratios > self.max_value] = self.max_value
        
        # Find minimum finite ratio
        min_ratio = np.min(ratios)
        
        # If minimum ratio is infinity, problem is unbounded
        if np.isinf(min_ratio):
            return None
        
        # Find all rows with ratios close to the minimum (for tie detection)
        tied_rows_mask = np.abs(ratios - min_ratio) < self.tol * 10
        
        if np.sum(tied_rows_mask) > 1:
            # Multiple minimum ratios - select the most stable pivot
            tied_rows = np.where(tied_rows_mask)[0]
            tied_pivots = np.abs(column[tied_rows])
            best_pivot_idx = np.argmax(tied_pivots)
            min_ratio_row = tied_rows[best_pivot_idx]
        else:
            # Single minimum ratio
            min_ratio_row = np.argmin(ratios)
        
        return min_ratio_row + 1  # Add 1 to account for objective row
    
    def pivot(self, leaving_row, entering_col):
        """
        Perform a pivot operation on the tableau with maximum precision and improved numerical stability.
        
        Parameters:
        -----------
        leaving_row : int
            The row index of the leaving variable
        entering_col : int
            The column index of the entering variable
        """
        # Record the pivot for cycling detection
        self.pivot_history.append((leaving_row, entering_col))
        
        # Get the pivot element
        pivot_element = self.tableau[leaving_row, entering_col]
        
        # RESTORED ORIGINAL CODE: Simplified pivot handling with improved stability
        # Check for pivots that are too close to zero
        if abs(pivot_element) < self.pivot_tol:
            if self.verbose:
                print(f"Warning: Pivot element {pivot_element:.2e} is close to zero")
            # Force a minimum magnitude for the pivot to improve stability
            pivot_element = np.sign(pivot_element) * max(abs(pivot_element), self.pivot_tol)
        
        # Check if an artificial variable is leaving the basis
        leaving_var = self.basic_vars[leaving_row - 1]
        leaving_artificial = leaving_var in self.artificial_indices
        
        if self.verbose:
            print(f"\nPivoting: entering column {entering_col}, leaving row {leaving_row}")
            print(f"Pivot element: {pivot_element:.{DECIMAL_PRECISION}g}")
            # Show pre-pivot tableau with ratios
            self.print_tableau(entering_col)
        
        # Store original tableau for stable pivot calculation
        old_tableau = self.tableau.copy()
        
        # RESTORED ORIGINAL CODE: Use simpler pivot method from original
        # 1. Scale the pivot row
        self.tableau[leaving_row, :] = old_tableau[leaving_row, :] / pivot_element
        
        # 2. Update all other rows
        # Create mask for all rows except the pivot row
        non_pivot_rows = np.ones(self.tableau.shape[0], dtype=bool)
        non_pivot_rows[leaving_row] = False
        
        # Calculate the factors for all rows
        factors = old_tableau[:, entering_col].copy()
        
        # Create a mask for significant factors
        significant_factors = np.abs(factors) > self.tol
        significant_factors[leaving_row] = False  # Exclude pivot row
        
        # Combine masks to get rows that need updating
        rows_to_update = non_pivot_rows & significant_factors
        
        # Update all relevant rows at once using broadcasting
        # Create a 2D array of factors for broadcasting
        factors_2d = factors[rows_to_update, np.newaxis]
        
        # Update all rows at once
        self.tableau[rows_to_update, :] = old_tableau[rows_to_update, :] - factors_2d * self.tableau[leaving_row, :]
        
        # Update the basis
        self.basic_vars[leaving_row - 1] = entering_col
        
        # Apply numerical stabilization techniques
        self.apply_zero_tolerance()
        
        # If an artificial variable has left the basis, remove its column
        if leaving_artificial:
            self.remove_artificial_column(leaving_var)
        
        # Increment pivot count
        self.pivot_count += 1
        
        if self.verbose:
            print(f"\nAfter pivot:")
            self.print_tableau()
    
    def remove_artificial_column(self, artificial_var):
        """
        Remove an artificial variable column from the tableau after it leaves the basis.
        
        Parameters:
        -----------
        artificial_var : int
            The column index of the artificial variable to remove
        """
        if self.verbose:
            print(f"Removing artificial variable at column {artificial_var} from tableau")
        
        # Create a new tableau without the artificial column
        old_tableau = self.tableau.copy()
        rows, cols = old_tableau.shape
        
        # Create the new tableau with one less column
        self.tableau = np.zeros((rows, cols - 1))
        
        # Copy everything except the artificial column with high precision
        self.tableau[:, 0:artificial_var] = old_tableau[:, 0:artificial_var]  # Before the artificial
        if artificial_var < cols - 1:  # If not the last column
            self.tableau[:, artificial_var:] = old_tableau[:, artificial_var+1:]  # After the artificial
        
        # Update indices of all variables that come after the removed column
        for i in range(len(self.basic_vars)):
            if self.basic_vars[i] > artificial_var:
                self.basic_vars[i] -= 1
        
        # Update artificial_indices
        new_artificial_indices = []
        for idx in self.artificial_indices:
            if idx != artificial_var:  # Don't include the removed variable
                if idx > artificial_var:
                    # Adjust index for all artificial variables after the removed one
                    new_artificial_indices.append(idx - 1)
                else:
                    new_artificial_indices.append(idx)
        
        self.artificial_indices = new_artificial_indices
        
        # Update structure information
        self.num_artificial_vars -= 1
        self.total_vars -= 1
        
        # Apply numerical stabilization after column removal
        self.apply_numerical_stabilization()
        
        if self.verbose:
            print(f"Artificial variable removed. Remaining artificial variables: {self.artificial_indices}")
            print(f"Updated basis: {self.basic_vars}")
    
    def check_artificial_variables(self):
        """
        Check if any artificial variables remain in the basis with non-zero values.
        
        Returns:
        --------
        bool
            True if no artificial variables in basis or all have zero values, False otherwise
        """
        # If all artificial variables have been removed from the tableau, we're feasible
        if len(self.artificial_indices) == 0:
            return True
            
        for i, var in enumerate(self.basic_vars):
            # Check if this is an artificial variable
            if var in self.artificial_indices:
                # Check its value
                value = self.tableau[i + 1, 0]
                # Values are considered zero if their absolute value is less than feasibility_tol
                if abs(value) > self.feasibility_tol:  # Use feasibility tolerance
                    if self.verbose:
                        print(f"Artificial variable at position {var} still in basis with value {value}")
                    return False
        
        return True
    
    def check_numerical_health(self):
        """
        Check the numerical health of the tableau and report any issues.
        
        Returns:
        --------
        bool
            True if the tableau appears numerically healthy, False otherwise
        """
        # Check for very large values that might indicate numerical issues
        max_abs_value = np.max(np.abs(self.tableau))
        if max_abs_value > self.max_value * 0.1:
            if self.verbose:
                print(f"Warning: Large values detected in the tableau (max: {max_abs_value:.2e})")
            return False
            
        # Check for very small non-zero values
        nonzero_mask = self.tableau != 0
        if np.any(nonzero_mask):
            min_nonzero = np.min(np.abs(self.tableau[nonzero_mask]))
            if min_nonzero < self.min_value * 10:
                if self.verbose:
                    print(f"Warning: Very small non-zero values detected (min: {min_nonzero:.2e})")
                return False
        
        # Check basic columns to ensure they form a proper identity structure
        basic_col_issues = 0
        for i, var in enumerate(self.basic_vars):
            col = self.tableau[1:, var]
            # The basic column should have a 1 at position i and 0 elsewhere
            for j in range(len(col)):
                if j == i and abs(col[j] - 1.0) > self.tol:
                    basic_col_issues += 1
                elif j != i and abs(col[j]) > self.tol:
                    basic_col_issues += 1
        
        if basic_col_issues > 0:
            if self.verbose:
                print(f"Warning: {basic_col_issues} issues found in basic columns")
            return False
        
        return True
    
    def restore_basis_with_qr(self):
        """
        Use QR decomposition to restore numerical stability in the basis.
        
        This method is called when numerical instability is detected. It rebuilds
        the basis columns using QR decomposition for better numerical properties.
        
        Returns:
        --------
        bool
            True if the restoration was successful, False otherwise
        """
        if self.verbose:
            print("Applying QR decomposition to restore basis stability")
        
        try:
            # Extract the basis matrix (excluding objective row and RHS column)
            basis_matrix = np.zeros((self.m, self.m))
            for i, var in enumerate(self.basic_vars):
                basis_matrix[:, i] = self.tableau[1:, var]
            
            # Compute QR decomposition with higher precision
            q, r = np.linalg.qr(basis_matrix)
            
            # Use R to restore the identity basis structure
            old_tableau = self.tableau.copy()
            
            # Update the basis columns using the QR factors
            for i in range(self.m):
                self.tableau[1:, self.basic_vars[i]] = 0.0
                self.tableau[i+1, self.basic_vars[i]] = 1.0
            
            # Update the RHS vector
            rhs = old_tableau[1:, 0]
            self.tableau[1:, 0] = np.linalg.solve(r, q.T @ rhs)
            
            # Update the objective function coefficients
            for j in range(1, self.tableau.shape[1]):
                if j not in self.basic_vars:
                    coeffs = old_tableau[1:, j]
                    transformed_coeffs = np.linalg.solve(r, q.T @ coeffs)
                    for i in range(self.m):
                        self.tableau[1+i, j] = transformed_coeffs[i]
            
            # Apply numerical stabilization
            self.apply_numerical_stabilization()
            
            if self.verbose:
                print("Basis restoration successful")
            return True
            
        except np.linalg.LinAlgError as e:
            if self.verbose:
                print(f"Error in QR decomposition: {e}")
            return False
    
    def check_for_cycling(self):
        """
        Check if the current basis configuration has been seen before,
        which indicates cycling in the simplex algorithm.
        
        Returns:
        --------
        bool
            True if cycling is detected, False otherwise
        """
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
            self.cycling_iterations = 0  # Reset cycle iteration counter
            return True
        
        # Add the current basis to history
        self.basis_history.append(current_basis)
        return False
    
    def solve(self):
        """
        Solve the feasibility problem using Simplex Phase 1 with maximum precision.
        
        Returns:
        --------
        dict
            Dictionary containing solution status, iterations, and other information
        """
        start_time = time.time()
        
        # Print instructions for skipping
        sys.stderr.write("Press 'q' or ESC to skip this method and proceed to the next one\n")
        sys.stderr.flush()
        
        # Initialize pivot counter for stability checks
        self.pivot_count = 0
        
        # Initial tableau
        if self.verbose:
            print("\nInitial Tableau:")
            self.print_tableau()
        
        # Check if the problem is already feasible (no artificial variables)
        if len(self.artificial_indices) == 0:
            if self.verbose:
                print("Problem is already feasible - no artificial variables needed")
            
            return {
                'status': 'feasible',
                'iterations': 0,
                'time': time.time() - start_time,
                'message': 'Problem is already feasible',
                'cycling_detected': False,
                'cycling_count': 0,
                'condition_number': self.condition_number_after_scaling
            }
        
        # Counter for key press checking frequency
        check_counter = 0
        
        # Counter for numerical health check frequency
        health_check_counter = 0
        
        # Main simplex loop - RESTORE ORIGINAL LOOP STRUCTURE
        while self.iterations < self.max_iter:
            self.iterations += 1
            
            # Display progress indicator (a dot) after each iteration
            sys.stderr.write(".")
            sys.stderr.flush()
            # Print a newline every 50 iterations for readability
            if self.iterations % 50 == 0:
                sys.stderr.write(f" [{self.iterations}]\n")
                sys.stderr.flush()
            
            # Check for key press every 10 iterations
            check_counter += 1
            if check_counter >= KEY_PRESS_CHECK_FREQUENCY:
                check_counter = 0
                if check_key_press():
                    sys.stderr.write("\nSkipping Simplex Phase 1 method as requested by user\n")
                    sys.stderr.flush()
                    
                    # Print a newline at the end for clean output formatting
                    if self.iterations % 50 != 0:
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
                        'condition_number': self.condition_number_after_scaling
                    }
            
            if self.verbose:
                print(f"\nIteration {self.iterations}:")
            
            # Periodically check numerical health of the tableau
            health_check_counter += 1
            if health_check_counter >= NUMERICAL_HEALTH_CHECK_FREQUENCY:
                health_check_counter = 0
                if not self.check_numerical_health():
                    if self.verbose:
                        print("Numerical issues detected - applying enhanced stabilization")
                    # Apply the standard stabilization method first
                    self.apply_numerical_stabilization()
            
            # Monitor for cycling
            if self.check_for_cycling():
                # Cycling was detected or continued
                if self.cycling_iterations >= MAX_CYCLING_ITERATIONS:
                    # Skip to next method after max iterations of cycling
                    sys.stderr.write(f"\nSkipping to next method after {self.cycling_iterations} iterations of cycling\n")
                    sys.stderr.flush()
                    
                    # Print a newline at the end for clean output formatting
                    if self.iterations % 50 != 0:
                        sys.stderr.write("\n")
                        sys.stderr.flush()
                    
                    return {
                        'status': 'cycling_skipped',
                        'iterations': self.iterations,
                        'time': time.time() - start_time,
                        'message': f'Cycling detected and skipped after {self.cycling_iterations} iterations of cycling',
                        'cycling_detected': True,
                        'cycling_count': self.cycling_count,
                        'cycle_length': self.cycle_length,
                        'condition_number': self.condition_number_after_scaling
                    }
            
            # If cycling was previously detected, increment counter
            if self.cycling_detected:
                self.cycling_iterations += 1
            
            # Skip to next method after max iterations of cycling
            if self.cycling_iterations >= MAX_CYCLING_ITERATIONS:
                sys.stderr.write(f"\nSkipping to next method after {self.cycling_iterations} iterations of cycling\n")
                sys.stderr.flush()
                
                # Print a newline at the end for clean output formatting
                if self.iterations % 50 != 0:
                    sys.stderr.write("\n")
                    sys.stderr.flush()
                
                return {
                    'status': 'cycling_skipped',
                    'iterations': self.iterations,
                    'time': time.time() - start_time,
                    'message': f'Cycling detected and skipped after {self.cycling_iterations} iterations of cycling',
                    'cycling_detected': True,
                    'cycling_count': self.cycling_count,
                    'cycle_length': self.cycle_length,
                    'condition_number': self.condition_number_after_scaling
                }
            
            # Find entering variable
            entering_col = self.find_entering_variable()
            
            # If no entering variable, we've reached optimality for Phase 1
            if entering_col is None:
                if self.verbose:
                    print("No negative coefficients in objective row - Phase 1 complete")
                
                # Check if any artificial variables remain in the basis with non-zero values
                if self.check_artificial_variables():
                    if self.verbose:
                        print(f"Feasibility attained after {self.iterations} iterations")
                    
                    # Extract solution
                    solution = np.zeros(self.num_original_vars)
                    for i, var in enumerate(self.basic_vars):
                        if var <= self.num_original_vars:  # It's an original variable
                            solution[var - 1] = self.tableau[i + 1, 0]
                    
                    # Print a newline at the end of iterations for clean output formatting
                    if self.iterations % 50 != 0:
                        sys.stderr.write("\n")
                        sys.stderr.flush()
                    
                    return {
                        'status': 'feasible',
                        'iterations': self.iterations,
                        'time': time.time() - start_time,
                        'message': 'Feasibility attained',
                        'solution': solution,
                        'cycling_detected': self.cycling_detected,
                        'cycling_count': self.cycling_count,
                        'cycle_length': self.cycle_length if self.cycling_detected else 0,
                        'condition_number': self.condition_number_after_scaling
                    }
                else:
                    if self.verbose:
                        print(f"Problem is infeasible: artificial variables remain in basis "
                              f"with non-zero values after {self.iterations} iterations")
                    
                    # Print a newline at the end of iterations for clean output formatting
                    if self.iterations % 50 != 0:
                        sys.stderr.write("\n")
                        sys.stderr.flush()
                    
                    return {
                        'status': 'infeasible',
                        'iterations': self.iterations,
                        'time': time.time() - start_time,
                        'message': 'Problem is infeasible - artificial variables remain in basis',
                        'cycling_detected': self.cycling_detected,
                        'cycling_count': self.cycling_count,
                        'cycle_length': self.cycle_length if self.cycling_detected else 0,
                        'condition_number': self.condition_number_after_scaling
                    }
            
            if self.verbose:
                print(f"Entering variable: column {entering_col}")
            
            # Find leaving variable
            leaving_row = self.find_leaving_variable(entering_col)
            
            # If no leaving variable, the problem is unbounded
            if leaving_row is None:
                if self.verbose:
                    print(f"Problem is unbounded after {self.iterations} iterations")
                
                # Print a newline at the end of iterations for clean output formatting
                if self.iterations % 50 != 0:
                    sys.stderr.write("\n")
                    sys.stderr.flush()
                
                return {
                    'status': 'unbounded',
                    'iterations': self.iterations,
                    'time': time.time() - start_time,
                    'message': 'Problem is unbounded',
                    'cycling_detected': self.cycling_detected,
                    'cycling_count': self.cycling_count,
                    'cycle_length': self.cycle_length if self.cycling_detected else 0,
                    'condition_number': self.condition_number_after_scaling
                }
            
            if self.verbose:
                print(f"Leaving variable: row {leaving_row}")
            
            # Perform pivot
            self.pivot(leaving_row, entering_col)
        
        # If we exceed maximum iterations
        # Print a newline at the end of iterations for clean output formatting
        if self.iterations % 50 != 0:
            sys.stderr.write("\n")
            sys.stderr.flush()
            
        return {
            'status': 'max_iterations',
            'iterations': self.max_iter,
            'time': time.time() - start_time,
            'message': f'Reached maximum iterations ({self.max_iter})',
            'cycling_detected': self.cycling_detected,
            'cycling_count': self.cycling_count,
            'cycle_length': self.cycle_length if self.cycling_detected else 0,
            'condition_number': self.condition_number_after_scaling
        }


def solve_with_simplex_phase1(A, b, max_iter=MAX_ITERATIONS, tol=ZERO_TOLERANCE, verbose=False, use_sparse=None):
    """
    Solve LP feasibility problem using Simplex Phase 1 method with maximum precision.
    
    Parameters:
    -----------
    A : array_like
        The coefficient matrix of the constraints (m x n)
    b : array_like
        The right-hand side vector (m)
    tol : float, optional
        Numerical tolerance for feasibility check (default from global config)
    max_iter : int, optional
        Maximum number of iterations (default from global config)
    verbose : bool, optional
        Whether to print step-by-step information
    use_sparse : bool, optional
        Whether to use sparse matrices for large problems.
        If None, will be determined automatically based on problem size.
        
    Returns:
    --------
    dict
        Solution information including status, iterations, time, etc.
    """
    # Auto-detect if sparse matrices should be used for large problems
    if use_sparse is None:
        m, n = A.shape
        estimated_size = (m + 1) * (n + m + min(m, 1000) + 1) * 8  # Size in bytes (float64 = 8 bytes)
        use_sparse = estimated_size > SPARSE_MATRIX_SIZE_THRESHOLD
        if verbose and use_sparse:
            print(f"Using sparse matrices for large problem: {m} constraints, {n} variables")
    
    try:
        solver = SimplexPhase1(A, b, tol=tol, max_iter=max_iter, verbose=verbose, use_sparse=use_sparse)
        result = solver.solve()
        return result
    except MemoryError:
        # If still getting memory error with auto-detection, force sparse mode
        print("Memory error encountered. Forcing sparse matrix usage.")
        solver = SimplexPhase1(A, b, tol=tol, max_iter=max_iter, verbose=verbose, use_sparse=True)
        result = solver.solve()
        return result


# Example usage
if __name__ == "__main__":
    # Print configuration info
    print("\nSimplex Phase 1 Configuration:")
    print(f"ZERO_TOLERANCE: {ZERO_TOLERANCE:.2e}")
    print(f"FEASIBILITY_TOLERANCE: {FEASIBILITY_TOLERANCE:.2e}")
    print(f"PIVOT_TOLERANCE: {PIVOT_TOLERANCE:.2e}")
    print(f"DECIMAL_PRECISION: {DECIMAL_PRECISION}")
    print(f"MAX_ITERATIONS: {MAX_ITERATIONS}")
    
    # Example 1: A feasible problem
    print("\nExample 1: Feasible problem")
    print("Constraints: x1 ≤ 4, x2 ≤ 6, 3x1 + 2x2 ≥ 18")
    
    A1 = np.array([
        [1, 0],    # x1 ≤ 4
        [0, 1],    # x2 ≤ 6
        [-3, -2]   # 3x1 + 2x2 ≥ 18 (converted to -3x1 - 2x2 ≤ -18)
    ])
    b1 = np.array([4, 6, -18])
    
    result1 = solve_with_simplex_phase1(A1, b1, verbose=True)
    print("\nResult:", result1['status'])
    print(f"Iterations: {result1['iterations']}")
    if 'solution' in result1:
        print(f"Solution: {result1['solution']}")
    print(f"Condition Number: {result1.get('condition_number', 'N/A')}")
    
    # Example 2: An infeasible problem
    print("\n\nExample 2: Infeasible problem")
    print("Constraints: x1 ≤ 4, x2 ≤ 6, 3x1 + 2x2 ≥ 30")
    
    A2 = np.array([
        [1, 0],     # x1 ≤ 4
        [0, 1],     # x2 ≤ 6
        [-3, -2]    # 3x1 + 2x2 ≥ 30 (converted to -3x1 - 2x2 ≤ -30)
    ])
    b2 = np.array([4, 6, -30])
    
    result2 = solve_with_simplex_phase1(A2, b2, verbose=True)
    print("\nResult:", result2['status'])
    print(f"Iterations: {result2['iterations']}")
    print(f"Message: {result2['message']}")
    print(f"Condition Number: {result2.get('condition_number', 'N/A')}")
    
    # Example 3: Ill-conditioned problem
    print("\n\nExample 3: Ill-conditioned problem")
    print("Testing numerical stability with an ill-conditioned matrix")
    
    # Create an ill-conditioned matrix by having nearly parallel constraints
    epsilon = 1e-10
    A3 = np.array([
        [1, 1],           # x1 + x2 ≤ 10
        [1, 1+epsilon],   # x1 + (1+ε)x2 ≤ 5 - nearly parallel to first constraint
        [0, 1]            # x2 ≤ 3
    ])
    b3 = np.array([10, 5, 3])
    
    result3 = solve_with_simplex_phase1(A3, b3, verbose=True)
    print("\nResult:", result3['status'])
    print(f"Iterations: {result3['iterations']}")
    if 'solution' in result3:
        print(f"Solution: {result3['solution']}")
    print(f"Condition Number: {result3.get('condition_number', 'N/A')}")