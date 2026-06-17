import numpy as np
import time
import sys
from iam_norm_utils import compute_dot_product, compute_normalized_dot_product_l2

# Import global configuration parameters
from config import (
    ZERO_TOLERANCE, 
    MAX_ITERATIONS, 
    ENABLE_ROW_SCALING,
    IAM1_USE_EXACT_NORM
)

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
        
        # Scaling factors
        self.row_scaling = None
        
        # Apply scaling and prepare the problem
        self.preprocess_problem()
        
        # Cycling detection
        self.basis_history = []
        self.pivot_history = []
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
        
        # Apply row scaling based on max absolute value (L∞-norm)
        for i in range(self.m):
            # Extract row
            row = A_scaled[i, :]
            
            # Compute maximum absolute value in the row
            max_abs_val = np.max(np.abs(row))
            
            if max_abs_val > self.tol:  # Avoid scaling nearly-zero rows
                self.row_scaling[i] = 1.0 / max_abs_val
                # Apply scaling to the row
                A_scaled[i, :] *= self.row_scaling[i]
                # Also scale corresponding b element
                b_scaled[i] *= self.row_scaling[i]
        
        # Use the scaled matrix and scaled b
        self.A = A_scaled
        self.b = b_scaled
        
        if self.verbose:
            print("=== Applied Max Value (L∞-norm) Row Scaling ===")
            print(f"Row scaling range: {np.min(self.row_scaling):.2e} to {np.max(self.row_scaling):.2e}")
    
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
        """Set values with absolute magnitude less than tolerance to zero."""
        # OPTIMIZED: Apply zero adjustment in one efficient operation
        small_mask = np.abs(self.dict) < self.tol
        self.dict[small_mask] = 0.0
        
    def is_primal_feasible(self):
        """Check if the current dictionary is primal feasible."""
        # OPTIMIZED: Simple direct comparison for faster feasibility check
        return np.all(self.dict[1:, 0] >= -self.tol)
    
    def compute_projection(self, j):
        """
        Compute the projection of column j onto the RHS vector.
        
        Parameters:
        -----------
        j : int
            Index of the non-basic variable (column)
            
        Returns:
        --------
        float
            Projection value
        """
        # OPTIMIZED: Extract column j and RHS with direct indexing
        col_j = self.dict[1:, j]
        rhs = self.dict[1:, 0]
        
        # OPTIMIZED: Compute normalized dot product efficiently
        # Use faster L2 norm calculation
        col_norm = np.linalg.norm(col_j)
        rhs_norm = np.linalg.norm(rhs)
        
        # Check for near-zero norms to avoid division by zero
        if col_norm < self.tol or rhs_norm < self.tol:
            return -float('inf')
            
        # Compute the dot product and normalize
        return np.dot(col_j, rhs) / (col_norm * rhs_norm)
    
    def print_dictionary(self):
        """Print the current dictionary in a readable format."""
        # Create labels for variables
        basic_labels = [f"x{i+1}" if i < self.n else f"s{i-self.n+1}" for i in self.basic_vars]
        nonbasic_labels = [f"x{i+1}" if i < self.n else f"s{i-self.n+1}" for i in self.nonbasic_vars]
        
        # Print header
        print(f"{'Basic':<10} | {'RHS':<10}", end="")
        for label in nonbasic_labels:
            print(f" | {label:<10}", end="")
        print()
        print("-" * (10 + 12 * (len(nonbasic_labels) + 1)))
        
        # Print each row
        for i in range(1, self.m + 1):
            print(f"{basic_labels[i-1]:<10} | {self.dict[i, 0]:<10.4f}", end="")
            for j in range(1, self.n + 1):
                print(f" | {self.dict[i, j]:<10.4f}", end="")
            print()
    
    def check_for_cycling(self):
        """
        Check if the current basis configuration has been seen before,
        using a more efficient representation for basis comparison.
        """
        # OPTIMIZED: Only check for cycling periodically to reduce overhead
        if self.iterations < 20 or self.iterations % 10 != 0:
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
        
        Parameters:
        -----------
        r : int
            Row index of the pivot element
        s : int
            Column index of the pivot element
        """
        # Record the pivot for cycling detection
        self.pivot_history.append((r, s))
        
        # Store pivot element
        pivot_element = self.dict[r, s]
        
        # Handle extremely small pivot elements if necessary
        if abs(pivot_element) < self.tol / 100:
            pivot_element = np.sign(pivot_element) * self.tol / 100
        
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
    
    def solve(self):
        """
        Solve the LP feasibility problem using IAM-1.
        
        Returns:
        --------
        dict
            Dictionary containing solve status, iterations, computation time, and solution if feasible
        """
        # Start timer for computation time only (excluding initialization)
        start_time = time.time()
        
        # Print instructions for skipping
        sys.stderr.write("Press 'q' or ESC to skip this method and proceed to the next one\n")
        sys.stderr.flush()
        
        # Counter for key press checking frequency
        check_counter = 0
        
        # Pre-allocate arrays for frequently used operations
        rhs_values = np.zeros(self.m)
        
        # Main iteration loop
        for self.iterations in range(self.max_iter):
            # Apply zero adjustment at the start of each iteration
            self.zero_adjust()
            
            # OPTIMIZED: Check feasibility with direct access to RHS values
            rhs_values = self.dict[1:, 0]
            if np.all(rhs_values >= -self.tol):
                self.is_feasible = True
                break
                
            # Monitor for cycling - just detect, don't prevent (now optimized)
            self.check_for_cycling()
            
            # Check for key press every 10 iterations
            check_counter += 1
            if check_counter >= 10:
                check_counter = 0
                if check_key_press():
                    sys.stderr.write("\nSkipping IAM1 method as requested by user\n")
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
                        'cycle_length': self.cycle_length if self.cycling_detected else 0
                    }
            
            # OPTIMIZED: Step 2-3 - Find most infeasible row
            # Get index directly from the previously obtained rhs_values
            r_idx = np.argmin(rhs_values)
            r = r_idx + 1  # Adjust for dictionary indexing
            
            # OPTIMIZED: Step 4 - Find admissible columns
            row_coeffs = self.dict[r, 1:]
            admissible_mask = row_coeffs < -self.tol
            
            # If no admissible columns, problem is infeasible
            if not np.any(admissible_mask):
                self.is_feasible = False
                break
                
            # Get indices of admissible columns
            admissible_cols = np.where(admissible_mask)[0] + 1  # +1 to adjust for RHS column
            
            # OPTIMIZED: Step 6-7 - Compute projections and find maximum angle
            # Compute projections for all admissible columns
            projections = np.array([self.compute_projection(j) for j in admissible_cols])
            
            # Find the column with maximum projection
            if len(projections) > 0:
                best_idx = np.argmax(projections)
                best_col = admissible_cols[best_idx]
            
            # Step 8: Perform pivot
                self.pivot(r, best_col)
            else:
                # This shouldn't happen if the algorithm is implemented correctly
                self.is_feasible = False
                break
            
            # Display progress indicator (a dot) after each iteration
            # Use sys.stderr which won't be suppressed when stdout is redirected
            sys.stderr.write(".")
            sys.stderr.flush()
            # Print a newline every 50 iterations for readability
            if (self.iterations + 1) % 50 == 0:
                sys.stderr.write(f" [{self.iterations + 1}]\n")
                sys.stderr.flush()
        
        # Print a newline at the end of iterations for clean output formatting
        if self.iterations % 50 != 0:
            sys.stderr.write("\n")
            sys.stderr.flush()
        
        # Calculate computation time
        computation_time = time.time() - start_time
        
        # Extract solution if feasible
        if self.is_feasible:
            # Apply final zero adjustment
            self.zero_adjust()
            
            # Construct the solution vector
            solution = np.zeros(self.n + self.m)
            for i, var in enumerate(self.basic_vars):
                if var < solution.size:
                    solution[var] = self.dict[i+1, 0]
            
            # No need to unscale by row_scaling since row scaling doesn't affect the solution vector
            # Row scaling affects constraints, not the variables
            
            return {
                'status': 'feasible',
                'iterations': self.iterations,
                'time': computation_time,
                'solution': solution[:self.n],  # Return only original variables
                'cycling_detected': self.cycling_detected,
                'cycling_count': self.cycling_count,
                'cycle_length': self.cycle_length if self.cycling_detected else 0
            }
        elif self.is_feasible is False:
            return {
                'status': 'infeasible',
                'iterations': self.iterations,
                'time': computation_time,
                'cycling_detected': self.cycling_detected,
                'cycling_count': self.cycling_count,
                'cycle_length': self.cycle_length if self.cycling_detected else 0
            }
        else:
            return {
                'status': 'max_iterations',
                'iterations': self.max_iter,
                'time': computation_time,
                'cycling_detected': self.cycling_detected,
                'cycling_count': self.cycling_count,
                'cycle_length': self.cycle_length if self.cycling_detected else 0
            }


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
        Numerical tolerance for feasibility check (default from global config)
    max_iter : int, optional
        Maximum number of iterations (default from global config)
    verbose : bool, optional
        Whether to print step-by-step information
        
    Returns:
    --------
    dict
        Solution information including cycling statistics
        
    Notes:
    ------
    This implementation applies max value (L∞-norm) row scaling as a preprocessing
    step to improve numerical stability. Each row is divided by its maximum absolute
    value before the algorithm proceeds.
    """
    solver = IAM1(A, b, tol, max_iter, verbose)
    return solver.solve()


# Example usage
if __name__ == "__main__":
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