import numpy as np
import time
import sys
import inspect
import os
from iam_norm_utils import compute_dot_product, compute_normalized_dot_product_l2

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
    print("Using NETLIB high precision settings for IAM-1")
else:
    # Use standard IAM precision settings
    from config_iam import (
        ZERO_TOLERANCE, 
        FEASIBILITY_TOLERANCE,
        MAX_ITERATIONS, 
        ENABLE_ROW_SCALING,
        ENABLE_DYNAMIC_ROW_SCALING,
        IAM1_USE_EXACT_NORM,
        CYCLING_CHECK_FREQUENCY,
        CYCLING_CHECK_START,
        PIVOT_MIN_VALUE,
        PIVOT_SAFETY_FACTOR,
        KEY_PRESS_CHECK_FREQUENCY,
        PROGRESS_DOT_FREQUENCY,
        PROGRESS_LINE_FREQUENCY,
        MIN_ROW_SCALE_VALUE,
        MAX_ROW_SCALE_VALUE,
        MAX_CYCLING_ITERATIONS,
        ADMISSIBLE_COL_TOLERANCE,
        DEFAULT_INFEASIBLE_PROJECTION_VALUE,
        # Floating-point precision parameters
        MACHINE_EPSILON,
        EPSILON_SMALL,
        EPSILON_MEDIUM,
        EPSILON_LARGE,
        # Display parameters
        COLUMN_WIDTH,
        DECIMAL_PRECISION,
        ENABLE_COLOR_OUTPUT,
        get_config_info
    )
    print("Using standard precision settings for IAM-1")

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

class IAM1:
    """
    Implementation of Inayatullah's Angle Method Type 1 (IAM-1) for linear programming feasibility.
    
    IAM-1 is a projection-based algorithm that eliminates the need for artificial variables by 
    using a minimum angle criterion to select pivot elements. This implementation includes
    max value (L∞-norm) row scaling as a preprocessing step for improved numerical stability.
    """
    
    def __init__(self, A, b, tol=ZERO_TOLERANCE, max_iter=MAX_ITERATIONS, verbose=False):
        """
        Initialize the IAM-1 solver with problem data.
        
        Parameters:
        -----------
        A : numpy.ndarray
            The coefficient matrix of the constraints (m x n)
        b : numpy.ndarray
            The right-hand side vector (m)
        tol : float, optional
            Numerical tolerance for feasibility check (default from config_iam)
        max_iter : int, optional
            Maximum number of iterations (default from config_iam)
        verbose : bool, optional
            Whether to print step-by-step information
        """
        # Store reference to configuration
        import config_iam
        self.config = config_iam
        
        # Copy IAM1_USE_EXACT_NORM from config for easy access
        self.IAM1_USE_EXACT_NORM = IAM1_USE_EXACT_NORM
        
        self.A_original = np.array(A, dtype=float)
        self.b_original = np.array(b, dtype=float)
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        
        self.m, self.n = self.A_original.shape
        self.is_feasible = None
        self.iterations = 0
        
        # Scaling factors
        self.row_scaling = None
        
        # Apply scaling and prepare the problem
        self.preprocess_problem()
        
        # Cycling detection
        self.basis_history = []
        self.pivot_history = []  # Use for logging pivots
        self.pivot_count = 0     # Track number of pivots
        self.cycling_detected = False
        self.cycling_count = 0
        self.cycle_length = 0
        self.cycling_details = []
        
        # Convert problem to standard form dictionary
        self.initialize_dictionary()
    
    def preprocess_problem(self):
        """
        Scale problem using max value (L∞-norm) row scaling.
        
        This preprocessing step improves numerical stability by scaling each constraint 
        row to have a maximum absolute coefficient of 1. The scaling:
        
        1. Balances the magnitude of constraint coefficients
        2. Reduces the risk of overflow or underflow in calculations
        3. Helps prevent numerical issues when constraints have widely varying magnitudes
        4. Makes the problem more amenable to accurate angle calculations
        
        The scaling preserves the feasible region and optimal solution while improving
        the numerical properties of the problem.
        """
        # Initialize row scaling factor
        self.row_scaling = np.ones(self.m)
        
        # Create a copy of the original matrix to work with
        A_scaled = self.A_original.copy()
        b_scaled = self.b_original.copy()
        
        # Use machine epsilon to determine minimum scaling value
        min_scale = max(MIN_ROW_SCALE_VALUE, EPSILON_MEDIUM)
        max_scale = min(MAX_ROW_SCALE_VALUE, 1.0/MACHINE_EPSILON)
        
        # Apply row scaling based on max absolute value (L∞-norm)
        for i in range(self.m):
            # Extract row
            row = A_scaled[i, :]
            
            # Compute maximum absolute value in the row with high precision
            max_abs_val = 0.0
            for val in np.abs(row):
                if val > max_abs_val:
                    max_abs_val = val
            
            # Only scale rows with significant values
            if max_abs_val > EPSILON_MEDIUM:
                # Apply scaling with bounds to prevent extreme values
                scale_factor = 1.0 / max_abs_val
                scale_factor = max(min(scale_factor, max_scale), min_scale)
                
                self.row_scaling[i] = scale_factor
                
                # Apply scaling to the row with maximum precision
                for j in range(len(row)):
                    A_scaled[i, j] = A_scaled[i, j] * scale_factor
                
                # Also scale corresponding b element
                b_scaled[i] = b_scaled[i] * scale_factor
        
        # Use the scaled matrix and scaled b
        self.A = A_scaled
        self.b = b_scaled
        
        if self.verbose:
            print("=== Applied Max Value (L∞-norm) Row Scaling with Maximum Precision ===")
            print(f"Row scaling range: {np.min(self.row_scaling):.16e} to {np.max(self.row_scaling):.16e}")
            print(f"Using Min/Max Scale Values: {min_scale:.16e} to {max_scale:.16e}")
    
    def initialize_dictionary(self):
        """Initialize the dictionary for the LP problem."""
        self.basic_vars = list(range(self.n, self.n + self.m))  # Initially, slack variables are basic
        self.nonbasic_vars = list(range(self.n))  # Original variables are non-basic
        
        # Initialize dictionary: first column is RHS, other columns correspond to non-basic variables
        self.dict = np.zeros((self.m + 1, self.n + 1))
        self.dict[1:, 0] = self.b  # RHS
        self.dict[1:, 1:] = self.A  # Coefficients for non-basic variables
        
        # Apply zero adjustment
        self.zero_adjust()
        
        if self.verbose:
            print("=== Initial Dictionary ===")
            self.print_dictionary()
            print()
    
    def zero_adjust(self):
        """
        Set values with absolute magnitude less than tolerance to zero.
        Uses machine epsilon based tolerances for maximum precision.
        """
        # Get machine epsilon for reference
        eps = np.finfo(np.float64).eps
        
        # Use dynamic tolerance based on maximum absolute values in the dictionary
        max_abs_value = np.max(np.abs(self.dict))
        min_abs_value = np.min(np.abs(self.dict[np.abs(self.dict) > 0]))
        
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
    
    def find_max_projection(self, r):
        """
        Find the column with maximum projection onto the right-hand side
        among admissible columns from row r.
        
        Args:
            r: Row index with most negative RHS value (1-indexed)
            
        Returns:
            tuple: (entering_col, max_projection_value)
            entering_col will be -1 if no suitable column is found
        """
        max_proj = -float('inf')
        entering_col = -1
        
        # Find all columns with negative coefficients in row r
        # In IAM-1, a column j is admissible if dict[r, j] < -tol
        admissible_cols = []
        for j in range(1, self.n+1):
            if self.dict[r, j] < -self.tol:
                admissible_cols.append(j)
        
        # If no admissible columns exist, return -1 to indicate no solution
        if len(admissible_cols) == 0:
            if self.verbose:
                print(f"No negative coefficients in row {r}, problem appears infeasible")
            return -1, -float('inf')
        
        if self.verbose:
            print(f"Evaluating {len(admissible_cols)} columns for row {r}")
        
        # Compute projections for all candidate columns
        for j in admissible_cols:
            # Use enhanced precision projection calculation
            proj = self.compute_projection(j)
            
            # We want the column with maximum projection value
            if proj > max_proj:
                max_proj = proj
                entering_col = j
        
        # Verify if we found a suitable column
        if entering_col == -1 and self.verbose:
            print("No suitable column found, problem appears infeasible.")
        elif self.verbose:
            print(f"Selected column {entering_col} with projection value {max_proj:.6f}")
        
        return entering_col, max_proj
    
    def find_leaving_row(self):
        """
        Find the basic variable with the most negative RHS value.
        For IAM-1, the leaving variable is the basic variable with the most negative RHS value.
        If all RHS values are non-negative, then the current solution is feasible.
            
        Returns:
            Index of the leaving row (1-indexed), or -1 if solution is feasible
        """
        # Check if there are any negative RHS values
        most_negative_val = 0
        leaving_row = -1
        
        for i in range(1, self.m+1):
            rhs_val = self.dict[i, 0]
            if rhs_val < most_negative_val:
                most_negative_val = rhs_val
                leaving_row = i
        
        if self.verbose:
            if leaving_row != -1:
                print(f"Most negative RHS value at row {leaving_row}: {most_negative_val:.6f}")
            else:
                print("No negative RHS values found, solution is feasible.")
        
        return leaving_row
    
    def _kahan_dot_product(self, v1, v2):
        """
        Compute dot product with Kahan summation algorithm for maximum precision.
        
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
        
        sum_value = 0.0
        comp = 0.0  # Compensation term for lost low-order bits
        
        for i in range(len(v1)):
            product = v1[i] * v2[i]
            y = product - comp  # Compensated summand
            t = sum_value + y   # Next approximation
            comp = (t - sum_value) - y  # Update compensation term
            sum_value = t
        
        return sum_value
    
    def compute_projection(self, idx):
        """
        Compute the precise projection value of a given column onto the right hand side.
        This uses Kahan summation for maximum numerical precision.
        
        Parameters:
        -----------
        idx : int
            The index of the column to compute the projection for
            
        Returns:
        --------
        float
            The unnormalized projection value (dot product / column norm)
        """
        # Extract the column and rhs directly with maximum precision
        col = self.dict[1:, idx].copy()  # Skip the objective row
        rhs = self.dict[1:, 0].copy()    # Right hand side
        
        # Get machine epsilon for precise comparisons
        eps = np.finfo(np.float64).eps
        
        # Always use maximum precision for numerical stability
        # Use Kahan summation for maximum precision in dot products and norms
        col_dot_rhs = self._kahan_dot_product(col, rhs)
        col_squared_norm = self._kahan_dot_product(col, col)
        
        # Check for near-zero norm to prevent division by zero
        # Use a threshold based on machine epsilon - increase threshold in problematic cases
        col_norm_threshold = max(eps * 100, eps * 1000 * abs(col_dot_rhs))
        
        # Ensure norm is positive and above threshold
        if col_squared_norm < col_norm_threshold:
            if self.verbose:
                print(f"Warning: Column norm too small ({col_squared_norm:.2e}), using minimum threshold")
            col_squared_norm = col_norm_threshold
        
        # Compute norm with maximum precision
        col_norm = np.sqrt(max(0.0, col_squared_norm))  # Ensure no negative values due to numerical error
    
        # Calculate unnormalized projection with maximum precision
        # This is the dot product divided by the column norm
        # (rather than the cosine similarity which also divides by the RHS norm)
        projection = col_dot_rhs / col_norm
        
        # Check for extremely large values that could cause overflow
        max_safe_value = 1.0 / (eps * 1000)
        if abs(projection) > max_safe_value:
            # Cap projection to a safe but large value
            return np.sign(projection) * max_safe_value
            
        return projection
    
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
            print()
    
    def check_for_cycling(self):
        """
        Check if the current basis configuration has been seen before,
        using a more efficient representation for basis comparison.
        """
        # OPTIMIZED: Only check for cycling periodically to reduce overhead
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
        self.basis_history.append(current_basis)
        return False
    
    def pivot(self, r, s):
        """
        Perform a pivot operation on element (r,s) of the dictionary.
        Combines the robust pivot selection from our implementation with 
        the efficient vectorized operations from iam1_original.py.
        
        Parameters:
        -----------
        r : int
            Row index of the pivot element
        s : int
            Column index of the pivot element
        """
        # Check if r and s are valid indices
        if r <= 0 or r > self.m or s <= 0 or s > self.n:
            raise ValueError(f"Invalid pivot indices: ({r}, {s})")
            
        # Record pivot operation for tracking
        self.pivot_count += 1
        self.pivot_history.append((r, s))
        
        # Get pivot element
        pivot_element = self.dict[r, s]
        
        # NUMERICAL STABILITY: Check for very small pivot element and search for better pivots
        if abs(pivot_element) < self.tol:
            if self.verbose:
                print(f"Warning: Very small pivot element {pivot_element:.2e}")
                print("Attempting to find a better pivot...")
            
            # Try all columns systematically to find a suitable pivot
            all_columns = list(range(1, self.n+1))
            all_columns.remove(s)  # Remove the current column
            
            max_abs_value = 0.0
            best_col = -1
            
            # First try to find a column with a non-zero coefficient in the same row
            for alt_c in all_columns:
                if abs(self.dict[r, alt_c]) > max_abs_value:
                    max_abs_value = abs(self.dict[r, alt_c])
                    best_col = alt_c
            
            if max_abs_value > self.tol:
                if self.verbose:
                    print(f"Found better pivot element in column {best_col} with value {max_abs_value:.6e}")
                s = best_col
                pivot_element = self.dict[r, s]
            else:
                # If no better column for this row, try different rows with any column
                best_row = -1
                for alt_r in range(1, self.m+1):
                    if alt_r != r and self.dict[alt_r, 0] < 0:  # Only consider infeasible rows
                        for alt_c in range(1, self.n+1):
                            if abs(self.dict[alt_r, alt_c]) > max_abs_value:
                                max_abs_value = abs(self.dict[alt_r, alt_c])
                                best_row = alt_r
                                best_col = alt_c
                
                if best_row != -1 and max_abs_value > self.tol:
                    if self.verbose:
                        print(f"Found better pivot element in row {best_row}, column {best_col} with value {max_abs_value:.6e}")
                    r = best_row
                    s = best_col
                    pivot_element = self.dict[r, s]
                else:
                    if self.verbose:
                        print("Could not find a suitable alternative pivot. Using a small perturbation.")
                    # Apply a small perturbation to the pivot element to avoid division by zero
                    pivot_element = np.sign(pivot_element) * max(abs(pivot_element), self.tol/100)
                    self.dict[r, s] = pivot_element
        
        if self.verbose:
            print(f"Pivoting: ({r}, {s})")
            
        # EFFICIENT PIVOTING: Use the optimized vectorized implementation
        # Create a copy of the dictionary to store old values while updating
        old_dict = self.dict.copy()
        
        # 1. Replace the pivot element with its reciprocal
        self.dict[r, s] = 1.0 / pivot_element
        
        # OPTIMIZED: 2+3. Update all rows and columns with reduced checks
        # Create efficient masks for non-pivot rows and columns
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
        
        # Create update matrix in one operation
        update_matrix = np.outer(row_factors.flatten(), col_factors.flatten()) / pivot_element
        
        # Apply updates in one operation
        self.dict[np.ix_(rows[non_pivot_rows], cols[non_pivot_cols])] -= update_matrix
        
        # Apply zero adjustment after pivoting
        self.zero_adjust()
        
        # Update basic/nonbasic variable sets
        leaving_var = self.basic_vars[r - 1]
        entering_var = self.nonbasic_vars[s - 1]
        
        self.basic_vars[r - 1] = entering_var
        self.nonbasic_vars[s - 1] = leaving_var
        
        return True
    
    def solve(self):
        """
        Solve the LP feasibility problem using IAM-1.
        
        This is the original IAM-1 algorithm that:
        1. Finds the most infeasible row (most negative RHS)
        2. Determines admissible columns (coefficients < -tol)
        3. Selects entering variable with maximum projection
        4. Performs pivot and continues until feasible or max iterations reached
        """
        if self.verbose:
            print("Solving LP feasibility problem using IAM-1...")
        
        # Initialize performance tracking
        start_time = time.time()
        projection_time = 0
        self.pivot_count = 0
        self.pivot_history = []
        
        # Print instructions for skipping
        sys.stderr.write("Press 'q' or ESC to skip this method and proceed to the next one\n")
        sys.stderr.flush()
        
        # Counter for key press checking and progress display
        check_counter = 0
        
        # Initialize precision monitoring stats
        precision_monitor_frequency = 50  # Check numerical health periodically
        precision_stats_history = []
        precision_recoveries = 0
        
        # Check initial feasibility
        if self.is_primal_feasible():
            if self.verbose:
                print("Initial dictionary is feasible.")
            return True, 0, time.time() - start_time, self.dict
        
        # Main iteration loop
        for iteration in range(self.max_iter):
            self.iterations = iteration + 1  # 1-indexed iterations
            
            # Monitor numerical precision periodically
            if iteration % precision_monitor_frequency == 0:
                precision_stats = self.monitor_numerical_precision()
                precision_stats_history.append((iteration, precision_stats))
                
                # Take action based on precision health
                if hasattr(precision_stats, 'numerical_health') and precision_stats.get('numerical_health') in ['poor', 'critical']:
                    if self.verbose:
                        print(f"Taking corrective action for numerical stability...")
                    
                    # Apply row scaling for numerical stability if enabled
                    if ENABLE_DYNAMIC_ROW_SCALING:
                        self.apply_row_scaling()
                        precision_recoveries += 1
                    elif self.verbose:
                        print("Dynamic row scaling is disabled, skipping correction")
            
            # Apply zero adjustment at the start of each iteration
            self.zero_adjust()
            
            # Check if the solution is feasible
            if self.is_primal_feasible():
                if self.verbose:
                    print(f"Found feasible solution after {self.iterations} iterations.")
                return True, self.iterations, time.time() - start_time, self.dict
            
            # STEP 1-2: Find most infeasible row (most negative RHS value)
            r = self.find_leaving_row()
            
            if r == -1:
                # No negative RHS values found, solution is feasible
                if self.verbose:
                    print("No negative RHS values found, solution is feasible.")
                return True, self.iterations, time.time() - start_time, self.dict
            
            # STEP 3-6: Find column with maximum projection among admissible columns
            proj_start = time.time()
            entering_col, max_proj = self.find_max_projection(r)
            projection_time += time.time() - proj_start
            
            if entering_col == -1:
                # No suitable column found, problem is infeasible
                if self.verbose:
                    print(f"No suitable column found. Problem is infeasible.")
                return False, self.iterations, time.time() - start_time, self.dict
            
            # STEP 7: Perform pivot
            if self.verbose:
                print(f"Pivoting: ({r}, {entering_col})")
            self.pivot(r, entering_col)
            self.pivot_count += 1  # Increment pivot count
            
            # Check for cycling - just detect, don't prevent
            if self.iterations % CYCLING_CHECK_FREQUENCY == 0:
                self.check_for_cycling()
            
            # Check for key press to allow skipping
            check_counter += 1
            if check_counter >= KEY_PRESS_CHECK_FREQUENCY:
                check_counter = 0
                if check_key_press():
                    sys.stderr.write("\nSkipping IAM1 method as requested by user\n")
                    sys.stderr.flush()
                    return {
                        'status': 'skipped',
                        'iterations': self.iterations,
                        'time': time.time() - start_time,
                        'message': 'Method skipped by user',
                        'cycling_detected': self.cycling_detected,
                        'cycling_count': self.cycling_count,
                        'cycle_length': self.cycle_length if self.cycling_detected else 0
                    }
            
            # Display progress indicator
            if self.iterations % PROGRESS_DOT_FREQUENCY == 0:
                sys.stderr.write(".")
                sys.stderr.flush()
            if self.iterations % PROGRESS_LINE_FREQUENCY == 0:
                sys.stderr.write(f" [{self.iterations}]\n")
                sys.stderr.flush()
            
            # Verbose output on iteration milestones
            if self.verbose and self.iterations % 100 == 0:
                print(f"Iteration {self.iterations}: Projection value = {max_proj:.6f}")
        
        # If we reach here, we've hit the maximum iteration limit
        if self.verbose:
            print(f"Reached maximum iterations {self.max_iter} without finding a feasible solution.")
            print(f"Numerical precision recoveries applied: {precision_recoveries}")
        
        # Print a newline at the end for clean output formatting
        if self.iterations % PROGRESS_LINE_FREQUENCY != 0:
            sys.stderr.write("\n")
            sys.stderr.flush()
        
        return False, self.max_iter, time.time() - start_time, self.dict

    def check_numerical_stability(self):
        """
        Check the numerical stability of the dictionary and apply corrective measures if needed.
        Uses enhanced precision controls from configuration if available.
        
        Returns:
        --------
        bool
            True if instability was detected and corrective measures were applied
        """
        # Get machine epsilon for reference
        eps = np.finfo(np.float64).eps
        applied_corrections = False
        
        # Check for large values that might indicate numerical issues
        max_abs_val = np.max(np.abs(self.dict))
        safe_max = 1.0 / (eps * 1000)
        if max_abs_val > safe_max:
            if self.verbose:
                print(f"Warning: Large values detected (max: {max_abs_val:.2e})")
            applied_corrections = True
            
            # Apply row scaling if enabled
            if hasattr(self.config, 'ENABLE_ROW_SCALING') and self.config.ENABLE_ROW_SCALING:
                self.apply_row_scaling()
        
        # Check for very small non-zero values
        non_zero_mask = np.abs(self.dict) > eps
        if np.any(non_zero_mask):
            min_nonzero = np.min(np.abs(self.dict[non_zero_mask]))
            min_safe = eps * 100
            if min_nonzero < min_safe:
                if self.verbose:
                    print(f"Warning: Very small non-zero values detected (min: {min_nonzero:.2e})")
                applied_corrections = True
        
        # Check for errors in basic columns (should form identity submatrix)
        basic_col_issues = 0
        for i, var in enumerate(self.basic_vars):
            if var >= self.n:  # Only check real variables, not slack
                continue  # Skip if variable is out of range (slack variable)
                
            # Check that column forms proper identity element
            for row in range(1, self.m + 1):
                expected = 1.0 if row == i + 1 else 0.0
                if abs(self.dict[row, var % self.n + 1] - expected) > eps * 100:
                    basic_col_issues += 1
        
        if basic_col_issues > 0:
            if self.verbose:
                print(f"Warning: {basic_col_issues} issues found in basic columns")
            
            # Restore identity structure if advanced options enabled
            if hasattr(self.config, 'ENABLE_ITERATIVE_REFINEMENT') and self.config.ENABLE_ITERATIVE_REFINEMENT:
                self.restore_basis_with_qr()
            else:
                # Simpler approach - directly fix identity structure
                for i, var in enumerate(self.basic_vars):
                    if var < self.n:  # Only real variables
                        col_idx = var % self.n + 1  # Calculate the corresponding column index
                        # Zero out the column
                        self.dict[1:, col_idx] = 0.0
                        # Set identity element
                        self.dict[i+1, col_idx] = 1.0
                    
            applied_corrections = True
        
        # If corrections were made, apply zero adjustment
        if applied_corrections:
            self.zero_adjust()
        
        return applied_corrections

    def apply_row_scaling(self):
        """
        Apply row scaling to the current dictionary to improve numerical stability.
        This is useful when numerical issues arise during the solution process.
        """
        # Save the basic and nonbasic variables
        basic_vars = self.basic_vars.copy()
        nonbasic_vars = self.nonbasic_vars.copy()
        
        # Compute row scaling factors based on the maximum absolute value in each row
        row_scaling = np.ones(self.m)
        for i in range(1, self.m + 1):
            row = self.dict[i, :]
            max_abs_val = np.max(np.abs(row))
            if max_abs_val > self.tol:
                row_scaling[i-1] = 1.0 / max_abs_val
        
        # Apply scaling to each row
        for i in range(1, self.m + 1):
            if row_scaling[i-1] != 1.0:
                self.dict[i, :] *= row_scaling[i-1]
        
        # Apply zero adjustment after scaling
        self.zero_adjust()
        
        # Restore basic and nonbasic variables
        self.basic_vars = basic_vars
        self.nonbasic_vars = nonbasic_vars
        
        if self.verbose:
            print("Applied row scaling for numerical stability")
            print(f"Row scaling range: {np.min(row_scaling):.2e} to {np.max(row_scaling):.2e}")
            
        return True

    def monitor_numerical_precision(self):
        """
        Monitor numerical precision and identify potential instability issues.
        This method analyzes the current dictionary for signs of numerical instability.
        
        Returns:
        --------
        dict
            Dictionary with numerical health metrics
        """
        # Get machine epsilon for reference
        eps = np.finfo(np.float64).eps
        max_abs_val = np.max(np.abs(self.dict))
        metrics = {}
        
        # Check for extremely large values
        if max_abs_val > 1e12:
            metrics['max_value'] = max_abs_val
            metrics['numerical_health'] = 'critical'
        elif max_abs_val > 1e8:
            metrics['max_value'] = max_abs_val
            metrics['numerical_health'] = 'poor'
        else:
            metrics['max_value'] = max_abs_val
            metrics['numerical_health'] = 'good'
            
        # Check for very small non-zero values that might indicate instability
        non_zero_mask = np.abs(self.dict) > eps
        if np.any(non_zero_mask):
            min_nonzero = np.min(np.abs(self.dict[non_zero_mask]))
            metrics['min_nonzero'] = min_nonzero
            
            if min_nonzero < eps * 100:
                metrics['small_values_present'] = True
                if metrics['numerical_health'] != 'critical':
                    metrics['numerical_health'] = 'poor'
            else:
                metrics['small_values_present'] = False
        
        # Check for basic variable identity violations
        identity_violations = 0
        for i in range(self.m):
            for j in range(self.m):
                expected = 1.0 if i == j else 0.0
                var_idx = self.basic_vars[j] % self.n + 1 if self.basic_vars[j] >= self.n else self.basic_vars[j] + 1
                if i+1 < self.dict.shape[0] and var_idx < self.dict.shape[1]:
                    error = abs(self.dict[i+1, var_idx] - expected)
                    if error > self.tol:
                        identity_violations += 1
        
        metrics['identity_violations'] = identity_violations
        if identity_violations > 0 and metrics['numerical_health'] != 'critical':
            metrics['numerical_health'] = 'poor'
        
        return metrics
        
    def restore_basis_with_qr(self):
        """
        Restore numerical stability of the basis matrix using QR factorization.
        This more robust approach is used when enabled in config_netlib.py.
        """
        try:
            if self.verbose:
                print("Applying QR factorization to restore basis matrix...")
            
            # Build the basis matrix from the dictionary
            B = np.zeros((self.m, self.m))
            for i, var in enumerate(self.basic_vars):
                if var < self.dict.shape[1]:
                    B[:, i] = self.dict[1:, var]
            
            # Perform QR factorization
            Q, R = np.linalg.qr(B)
            
            # Check for near-singularity
            eps = np.finfo(np.float64).eps
            diag_R = np.abs(np.diag(R))
            min_diag = np.min(diag_R)
            
            if min_diag < eps * 1000:
                if self.verbose:
                    print(f"Warning: Near-singular basis detected (min diagonal: {min_diag:.2e})")
                    print("Using partial pivoting for better numerical stability")
            
            # Apply column pivoting to improve conditioning
            P = np.eye(self.m)
            B_copy = B.copy()
            
            for k in range(self.m - 1):
                # Find max element in remaining subcolumn for pivot
                max_idx = k + np.argmax(np.abs(B_copy[k:, k]))
                if max_idx != k:
                    # Swap rows
                    B_copy[[k, max_idx], :] = B_copy[[max_idx, k], :]
                    P[[k, max_idx], :] = P[[max_idx, k], :]
            
            # Recompute QR with better pivoting
            Q, R = np.linalg.qr(B_copy)
        
            # Update the basis columns with corrected identity
            for i, var in enumerate(self.basic_vars):
                if var < self.dict.shape[1]:
                    # Set exact identity column
                    self.dict[1:, var] = 0.0
                    self.dict[i+1, var] = 1.0
        
            # Update RHS to maintain solution equivalence
            self.dict[1:, 0] = np.dot(Q.T, self.dict[1:, 0])
            
            if self.verbose:
                print("Basis restoration with QR completed")
            
        except Exception as e:
            if self.verbose:
                print(f"QR factorization failed: {str(e)}")
                print("Falling back to simple identity restoration")
            
            # Simple identity restoration as fallback
            for i, var in enumerate(self.basic_vars):
                if var < self.dict.shape[1]:
                    self.dict[1:, var] = 0.0
                    self.dict[i+1, var] = 1.0

    def get_solution(self):
        """
        Extract the solution vector from the current dictionary.
        
        Returns:
        --------
        numpy.ndarray
            Solution vector for the original variables
        """
        # Create a solution vector for all variables (original + slack)
        full_solution = np.zeros(self.n + self.m)
        
        # Fill in the values for basic variables from the RHS
        for i, var in enumerate(self.basic_vars):
            full_solution[var] = self.dict[i+1, 0]
        
        # Return only the original variables (not slack)
        return full_solution[:self.n]


def solve_with_iam1(A, b, tol=ZERO_TOLERANCE, max_iter=MAX_ITERATIONS, verbose=False):
    """
    Solve a linear programming feasibility problem using IAM-1.
    
    Parameters:
    -----------
    A : array_like
        The coefficient matrix of the constraints (m x n)
    b : array_like
        The right-hand side vector (m)
    tol : float, optional
        Numerical tolerance for feasibility check (default from config_iam)
    max_iter : int, optional
        Maximum number of iterations (default from config_iam)
    verbose : bool, optional
        Whether to print step-by-step information
        
    Returns:
    --------
    dict
        Solution information including status, iterations, computation time, and solution vector
        
    Notes:
    ------
    This implementation applies max value (L∞-norm) row scaling as a preprocessing
    step to improve numerical stability. Each row is divided by its maximum absolute
    value before the algorithm proceeds.
    """
    # Initialize solver
    solver = IAM1(A, b, tol, max_iter, verbose)
    
    # Run the solve method
    is_feasible, iterations, time_taken, dictionary = solver.solve()
    
    # Calculate computation time
    computation_time = time_taken
    
    # Extract solution if feasible
    if is_feasible:
        # Construct the solution vector
        solution = np.zeros(solver.n)  # Only original variables
        for i, var in enumerate(solver.basic_vars):
            if var < solver.n:  # Only include original variables
                solution[var] = dictionary[i+1, 0]
        
        return {
            'status': 'feasible',
            'iterations': iterations,
            'time': computation_time,
            'solution': solution,  # Return only original variables
            'cycling_detected': solver.cycling_detected,
            'cycling_count': solver.cycling_count,
            'cycle_length': solver.cycle_length if solver.cycling_detected else 0
        }
    elif isinstance(is_feasible, dict):  # Handle skipped case
        return is_feasible
    else:
        return {
            'status': 'infeasible' if iterations < max_iter else 'max_iterations',
            'iterations': iterations,
            'time': computation_time,
            'cycling_detected': solver.cycling_detected,
            'cycling_count': solver.cycling_count,
            'cycle_length': solver.cycle_length if solver.cycling_detected else 0
        }


# Example usage
if __name__ == "__main__":
    # Print configuration info
    print("\n" + "="*60)
    print("IAM-1 CONFIGURATION".center(60))
    print("="*60)
    print(get_config_info())
    print("="*60 + "\n")
    
    # Example from the paper
    A = np.array([
        [-8, 1, -5],
        [8, 8, 0],
        [-7, 9, 0],
        [-7, 7, -9]
    ])
    b = np.array([6, 8, 1, 6])
    
    # Convert to standard form (Ax >= b becomes -Ax <= -b)
    A = -A
    b = -b
    
    # Set verbose=True to see step-by-step results
    result = solve_with_iam1(A, b, verbose=True)
    
    print("\n=== Final Result ===")
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    
    if result['status'] == 'feasible':
        print(f"Solution: {result['solution']}") 
    elif result['status'] == 'infeasible':
        print("Problem is infeasible")
    elif result['status'] == 'max_iterations':
        print(f"Maximum iterations ({MAX_ITERATIONS}) reached without finding feasible solution")
    
    if result.get('cycling_detected', False):
        print(f"Cycling detected ({result['cycling_count']} occurrences)")
        print(f"Cycle length: {result['cycle_length']}") 