import os
import pickle
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
import sys
from scipy.optimize import linprog
from contextlib import contextmanager

# Import centralized NETLIB configuration parameters
from config_netlib import (
    ZERO_TOLERANCE, 
    FEASIBILITY_TOLERANCE,
    PIVOT_TOLERANCE, 
    MAX_ITERATIONS, 
    ENABLE_ROW_SCALING,
    USE_SPARSE_MATRICES,
    RESUME_INTERRUPTED_TESTS,
    GENERATE_VISUALIZATIONS,
    MAX_PROBLEMS_PER_BATCH,
    IAM1_MAX_ITERATIONS,
    SIMPLEX_MAX_ITERATIONS,
    PAN_MAX_ITERATIONS,
    get_netlib_config_info
)

# Import config_iam to control IAM1 row scaling
import config_iam

# Suppress all warnings
warnings.filterwarnings("ignore")
# Specific warning suppressions for completeness
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Global verbosity control - Set to False by default
VERBOSE = False  # Set to False to suppress most non-essential output

# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

# Import solver methods
from iam1 import solve_with_iam1 as iam1_standard
# Commented out to fix import error - needs updating to use config_iam instead of config
# from iam1_l1_approximation import solve_with_iam1_l1_approximation 
# Disabled Pure L1 implementation
# from iam1_pure_l1 import solve_with_iam1_pure_l1
from iam_1b_l2_scaling import solve_with_iam1b_l2_scaling
from iam_1b_l1_scaling import solve_with_iam1b_l1_scaling
# Add simplex phase 1 method
from simplex_phase_1 import solve_with_simplex_phase1
# Add Pan's method with row scaling
from Pans_method_row_scale import check_feasibility_pan_row_scaled
# Add IAM-1B without column scaling
from iam1b_without_column_scaling import solve_with_iam1b_without_column_scaling

# No longer need the enhanced HiGHS solver
# from highs_wrapper import solve_with_highs_enhanced

# Function to redirect stdout temporarily (to suppress output)
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def clear_screen():
    """Clear the terminal screen."""
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For Mac and Linux
    else:
        _ = os.system('clear')

def load_problem(file_path):
    """Load a problem from a pickle file"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        # Verify data has required fields
        required_fields = ['name', 'A', 'b']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Problem data missing required field: {field}")
                
        return data
    except Exception as e:
        print(f"Error loading problem file {file_path}: {str(e)}")
        raise

def list_available_problems(problem_dir):
    """List available problems from the netlib directory."""
    if not os.path.exists(problem_dir):
        print(f"Directory not found: {problem_dir}")
        return []
    
    # Get problem files
    try:
        problem_files = [f for f in os.listdir(problem_dir) if f.endswith('.pkl')]
        problems = []
        
        for filename in problem_files:
            # Remove '.pkl' extension and handle NAME_ prefix and _reduced suffix
            problem_name = filename.replace('.pkl', '')
            if problem_name.startswith('NAME_'):
                problem_name = problem_name[5:]  # Remove 'NAME_' prefix
            if problem_name.endswith('_reduced'):
                problem_name = problem_name[:-8]  # Remove '_reduced' suffix
            problems.append(problem_name)
        
        return sorted(problems)
    except Exception as e:
        print(f"Error listing problems: {str(e)}")
        return []

def generate_feasible_lp(m, n, sparsity=0.0, bound=50, max_attempts=1500):
    """Generate a random feasible linear programming problem."""
    for _ in range(max_attempts):
        A, b = generate_random_lp(m, n, sparsity, bound)
        
        # Try to solve with Simplex Phase 1 to check feasibility
        try:
            result = solve_with_simplex_phase1(A, b)
            if result['status'] == 'feasible':
                return A, b
        except:
            continue
    
    # If we couldn't generate a feasible problem, return None
    return None


def generate_infeasible_lp(m, n, sparsity=0.0, bound=50, max_attempts=1500):
    """Generate a random infeasible linear programming problem."""
    for _ in range(max_attempts):
        A, b = generate_random_lp(m, n, sparsity, bound)
        
        # Try to solve with Simplex Phase 1 to check infeasibility
        try:
            result = solve_with_simplex_phase1(A, b)
            if result['status'] == 'infeasible':
                return A, b
        except:
            continue
    
    # If we couldn't generate an infeasible problem, return None
    return None

def solve_with_highs(A, b, max_iter=None, tol=ZERO_TOLERANCE, use_presolve=False, verbose=False):
    """
    Solve LP with standard HiGHS using SciPy's linprog implementation
    with presolve disabled and zero objective function for pure feasibility
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        max_iter: Maximum iterations (default None to let HiGHS determine automatically)
        tol: Tolerance (default from global config)
        use_presolve: Whether to use presolve
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with solver results
    """
    m, n = A.shape
    
    # Use a zero objective vector for pure feasibility detection
    c = np.zeros(n)
    
    # Set bounds for all variables (default is x >= 0)
    bounds = [(0, None) for _ in range(n)]
    
    start_time = time.time()
    try:
        # Suppress output unless verbose is True
        with suppress_stdout() if not verbose else contextmanager(lambda: (yield))():
            # Check if 'highs' method is available in SciPy's linprog
            try:
                # Create options dictionary
                options = {
                    'tol': tol,
                    'disp': verbose,
                    'presolve': use_presolve  # Allow presolve to be disabled
                }
                
                # Only add maxiter if it's specified (otherwise let HiGHS decide)
                if max_iter is not None:
                    options['maxiter'] = max_iter
                
                # Use HiGHS method with presolve disabled
                result = linprog(
                    c=c,
                    A_ub=A,
                    b_ub=b,
                    bounds=bounds,
                    method='highs',
                    options=options
                )
            except ValueError as e:
                # If 'highs' method is not available, fall back to 'interior-point'
                if "method" in str(e) and "highs" in str(e):
                    print("Warning: HiGHS method not available in your SciPy version. Falling back to 'interior-point'.")
                    result = linprog(
                        c=c,
                        A_ub=A,
                        b_ub=b,
                        bounds=bounds,
                        method='interior-point',
                        options={
                            'maxiter': max_iter,
                            'tol': tol,
                            'disp': verbose
                        }
                    )
                else:
                    raise
        
        solve_time = time.time() - start_time
        
        # Extract the number of iterations
        iterations = result.nit if hasattr(result, 'nit') else 0
        
        # Print summary if verbose
        if verbose:
            print("\n" + "="*60)
            print("HIGHS SOLVER RESULTS".center(60))
            print("="*60)
            print(f"• Status: {'feasible' if result.success else 'infeasible'}")
            print(f"• Iterations: {iterations}")
            print(f"• Time: {solve_time:.6f} seconds")
            print("="*60)
        
        return {
            'solver': 'highs',
            'success': result.success,
            'status': 'feasible' if result.success else 'infeasible',
            'iterations': iterations,
            'simplex_iterations': iterations,  # Standard HiGHS doesn't provide detailed iteration breakdown
            'time': solve_time,
            'is_feasible': result.success
        }
        
    except Exception as e:
        solve_time = time.time() - start_time
        if verbose:
            print(f"Error in HiGHS solver: {str(e)}")
        return {
            'solver': 'highs',
            'success': False,
            'status': f"Error: {str(e)}",
            'iterations': -1,
            'simplex_iterations': -1,
            'time': solve_time,
            'is_feasible': False
        }

def check_bounds_feasibility(A, b, tol=1e-7):
    """
    Check if the LP Ax <= b, x >= 0 has a trivial feasible solution (zero vector).
    
    Parameters:
    -----------
    A : array_like
        The constraint matrix
    b : array_like
        The right-hand side vector
    tol : float, optional
        Tolerance for feasibility check
        
    Returns:
    --------
    feasible : bool
        True if the problem is feasible with x = 0, False otherwise
    """
    # Check if b >= 0 (within tolerance)
    infeasible_indices = np.where(b < -tol)[0]
    
    if len(infeasible_indices) > 0:
        if VERBOSE:
            print(f"Bounds check found {len(infeasible_indices)} infeasible constraints (b_i < 0)")
        
        # For small problems, trying the simplex method
        if A.shape[0] <= 1000 and A.shape[1] <= 1000:
            print("Problem has negative b values. Attempting to find feasibility with Simplex Phase 1...")
            try:
                # Use sparse matrices for better memory efficiency
                result = solve_with_simplex_phase1(A, b, use_sparse=True)
                if result['status'] == 'feasible':
                    return True
                else:
                    return False
            except Exception as e:
                print(f"Error in Simplex feasibility check: {e}")
                return False
        return False  # For large problems with negative b values, assume infeasibility
    
    return True

def is_feasible(A, b, tol=1e-7):
    """
    Determine if the system Ax <= b, x >= 0 is feasible.
    
    Parameters:
    -----------
    A : array_like
        The constraint matrix
    b : array_like
        The right-hand side vector
    tol : float, optional
        Tolerance for feasibility check
        
    Returns:
    --------
    bool
        True if the system is feasible, False otherwise
    """
    # First check if x = 0 is a feasible solution
    if check_bounds_feasibility(A, b, tol):
        if VERBOSE:
            print("Zero vector is feasible")
        return True
    
    # For small problems, always try Simplex Phase 1
    if A.shape[0] <= 1000 and A.shape[1] <= 1000:
        try:
            # Use sparse matrices for better memory efficiency
            result = solve_with_simplex_phase1(A, b, use_sparse=True)
            return result['status'] == 'feasible'
        except Exception as e:
            print(f"Error in simplex method: {e}")
            return False
    
    # For large problems that couldn't be solved with x = 0,
    # we'll try other methods or return false
    print("Problem is too large for direct feasibility check and bounds check failed.")
    return False

def solve_with_method(method_name, A, b, max_iter=MAX_ITERATIONS, tol=ZERO_TOLERANCE, 
                 disable_row_scaling=False, disable_dynamic_row_scaling=False,
                 disable_column_scaling=True, disable_dynamic_column_scaling=True):
    """
    Solve a problem using the specified method.
    
    Parameters:
    -----------
    method_name : str
        Name of the method to use
    A : numpy.ndarray
        Constraint matrix
    b : numpy.ndarray
        Right-hand side vector
    max_iter : int
        Maximum number of iterations (default from global config)
    tol : float
        Numerical tolerance (default from global config)
    disable_row_scaling : bool
        Whether to disable initial row scaling for IAM1 method (default False)
    disable_dynamic_row_scaling : bool
        Whether to disable dynamic row scaling during pivoting for IAM1 method (default False)
    disable_column_scaling : bool
        Whether to disable initial column scaling for IAM1 method (default True)
    disable_dynamic_column_scaling : bool
        Whether to disable dynamic column scaling during pivoting for IAM1 method (default True)
        
    Returns:
    --------
    dict
        Result of the solver
    """
    # Method name mapping to actual function names
    method_mapping = {
        "simplex": "simplex_phase1",  # Use our simplex phase 1 implementation instead of linprog simplex
        "highs": "highs",  # Add HiGHS method
        "iam1": "iam1_standard",
        # "iam1_l1approx": "solve_with_iam1_l1_approximation",
        # "iam1_pure_l1": "solve_with_iam1_pure_l1",  # Disabled Pure L1 implementation
        "iam1b_l1scaling": "solve_with_iam1b_l1_scaling",
        "iam1b": "solve_with_iam1b_l2_scaling",  # Renamed from iam1b_l2scaling to iam1b
        "pan_row_scaled": "check_feasibility_pan_row_scaled",  # Add Pan's method with row scaling
        "iam1b_nocolscaling": "solve_with_iam1b_without_column_scaling",  # Add IAM-1B without column scaling
    }
    
    # Remove hardcoded iteration limits - use the provided max_iter instead
    method_max_iter = max_iter
    print(f"Using max_iter = {method_max_iter} for method {method_name}")
    
    # Check if method exists
    if method_name not in method_mapping:
        raise ValueError(f"Unknown method: {method_name}")
        
    start_time = time.time()
    
    # Ensure b is a 1D array
    if isinstance(b, np.ndarray) and len(b.shape) > 1:
        b = b.flatten()
    
    # Create a custom progress callback function
    def progress_callback(iterations, residual):
        if iterations % 100 == 0 or iterations <= 5:  # Show more detail in early iterations
            print(f"\r  {method_name}: Iteration {iterations}, residual: {residual:.6e}", end="", flush=True)
    
    try:
        if method_name == "simplex":
            # Use our simplex phase 1 implementation with sparse matrices
            with suppress_stdout():
                result = solve_with_simplex(A, b, method_name, method_max_iter, tol, VERBOSE)
            
            # Ensure all required fields are present
            result_dict = {
                'solver': method_name,
                'success': result.get('status') == 'success',
                'status': 'feasible' if result.get('status') == 'success' else result.get('status', 'error'),
                'iterations': result.get('iterations', 0),
                'simplex_iterations': result.get('iterations', 0),
                'time': result.get('time', 0),
                'is_feasible': result.get('status') == 'success',
                'message': result.get('message', '')
            }
            return result_dict
        
        elif method_name == "highs":
            # Use standard HiGHS solver with presolve disabled
            result = solve_with_highs(
                A=A, 
                b=b, 
                max_iter=None,  # No iteration limit for HiGHS
                tol=tol, 
                use_presolve=False,  # Disable presolve
                verbose=VERBOSE      # Use global verbosity setting
            )
            return result
        
        elif method_name == "iam1":
            # For IAM1, we need to handle both row and column scaling options
            # Store original settings
            original_row_scaling = config_iam.ENABLE_ROW_SCALING
            original_dynamic_row_scaling = config_iam.ENABLE_DYNAMIC_ROW_SCALING
            # We'll need to add column scaling to config_iam
            original_column_scaling = getattr(config_iam, 'ENABLE_COLUMN_SCALING', False)
            original_dynamic_column_scaling = getattr(config_iam, 'ENABLE_DYNAMIC_COLUMN_SCALING', False)
            
            # Apply requested settings
            if disable_row_scaling:
                config_iam.ENABLE_ROW_SCALING = False
                print("IAM1: Initial row scaling disabled")
            else:
                config_iam.ENABLE_ROW_SCALING = True
                print("IAM1: Initial row scaling enabled (default)")
                
            if disable_dynamic_row_scaling:
                config_iam.ENABLE_DYNAMIC_ROW_SCALING = False
                print("IAM1: Dynamic row scaling during pivoting disabled")
            else:
                config_iam.ENABLE_DYNAMIC_ROW_SCALING = True
                print("IAM1: Dynamic row scaling during pivoting enabled (default)")
                
            # Set column scaling options if they exist in config_iam
            if hasattr(config_iam, 'ENABLE_COLUMN_SCALING'):
                if disable_column_scaling:
                    config_iam.ENABLE_COLUMN_SCALING = False
                    print("IAM1: Initial column scaling disabled")
                else:
                    config_iam.ENABLE_COLUMN_SCALING = True
                    print("IAM1: Initial column scaling enabled")
                    
            if hasattr(config_iam, 'ENABLE_DYNAMIC_COLUMN_SCALING'):
                if disable_dynamic_column_scaling:
                    config_iam.ENABLE_DYNAMIC_COLUMN_SCALING = False
                    print("IAM1: Dynamic column scaling during pivoting disabled")
                else:
                    config_iam.ENABLE_DYNAMIC_COLUMN_SCALING = True
                    print("IAM1: Dynamic column scaling during pivoting enabled")
            
            # Run the solver
            with suppress_stdout():
                result = iam1_standard(A, b, tol=tol, max_iter=method_max_iter, verbose=False)
            
            # Restore original settings
            config_iam.ENABLE_ROW_SCALING = original_row_scaling
            config_iam.ENABLE_DYNAMIC_ROW_SCALING = original_dynamic_row_scaling
            
            if hasattr(config_iam, 'ENABLE_COLUMN_SCALING'):
                config_iam.ENABLE_COLUMN_SCALING = original_column_scaling
                
            if hasattr(config_iam, 'ENABLE_DYNAMIC_COLUMN_SCALING'):
                config_iam.ENABLE_DYNAMIC_COLUMN_SCALING = original_dynamic_column_scaling
        
        # Disabled Pure L1 implementation
        # elif method_name == "iam1_pure_l1":
        #     with suppress_stdout():
        #         result = solve_with_iam1_pure_l1(A, b, tol=tol, max_iter=method_max_iter, verbose=False)
        
        elif method_name == "iam1b_l1scaling":
            with suppress_stdout():
                result = solve_with_iam1b_l1_scaling(A, b, tol=tol, max_iter=method_max_iter, verbose=False)
        
        elif method_name == "iam1b" or method_name == "iam1b_l2scaling":  # Support both names for backward compatibility
            with suppress_stdout():
                result = solve_with_iam1b_l2_scaling(A, b, tol=tol, max_iter=method_max_iter, verbose=False)
        
        elif method_name == "pan_row_scaled":
            with suppress_stdout():
                result = check_feasibility_pan_row_scaled(A, b, tol=tol, max_iter=method_max_iter, verbose=False)
        
        elif method_name == "iam1b_nocolscaling":
            with suppress_stdout():
                result = solve_with_iam1b_without_column_scaling(A, b, tol=tol, max_iter=method_max_iter, verbose=False)
        
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        # Print a newline to ensure following output starts on a new line
        print()
        
        solve_time = time.time() - start_time
        
        # Standardize result structure
        if isinstance(result, dict):
            result['solver'] = method_name
            result['time'] = solve_time
            result['success'] = result.get('status') == 'feasible'
            result['is_feasible'] = result.get('status') == 'feasible'
            
            # Ensure status is consistently "feasible" or "infeasible" rather than optimization messages
            if 'status' in result:
                if result.get('status') == 'feasible' or result['success']:
                    result['status'] = 'feasible'
                elif isinstance(result['status'], str) and 'infeasible' in result['status'].lower():
                    result['status'] = 'infeasible'
                # Keep the original message for errors/other statuses
            else:
                result['status'] = 'unknown'
                
            # Make sure iterations is present and valid
            if 'iterations' not in result or not isinstance(result['iterations'], (int, float)):
                if VERBOSE:
                    print(f"Warning: Invalid or missing iterations for {method_name}. Using default value.")
                result['iterations'] = -1
            
            # Ensure simplex_iterations is present for consistency
            if 'simplex_iterations' not in result:
                result['simplex_iterations'] = result['iterations']
        else:
            if VERBOSE:
                print(f"Warning: Unexpected result format from {method_name}. Method returned: {type(result)}")
            result = {
                'solver': method_name,
                'success': False,
                'status': 'Error: unexpected result format',
                'iterations': -1,
                'simplex_iterations': -1,
                'time': solve_time,
                'is_feasible': False
            }
        
        return result
        
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"Error in {method_name} solver: {str(e)}")
        if VERBOSE:
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        return {
            'solver': method_name,
            'success': False,
            'status': str(e),
            'iterations': -1,
            'simplex_iterations': -1,
            'time': solve_time,
            'is_feasible': False
        }

def check_problem_feasibility(data, methods_and_limits, tol=ZERO_TOLERANCE, 
                         disable_row_scaling=False, disable_dynamic_row_scaling=False,
                         disable_column_scaling=True, disable_dynamic_column_scaling=True):
    """
    Check feasibility of a problem with different methods.
    
    Parameters:
    -----------
    data : dict
        Problem data with 'A' and 'b' fields
    methods_and_limits : dict
        Dictionary mapping method names to iteration limits
    tol : float
        Tolerance for feasibility check (default from global config)
    disable_row_scaling : bool
        Whether to disable initial row scaling for IAM1 method (default False)
    disable_dynamic_row_scaling : bool
        Whether to disable dynamic row scaling during pivoting for IAM1 method (default False)
    disable_column_scaling : bool
        Whether to disable initial column scaling for IAM1 method (default True)
    disable_dynamic_column_scaling : bool
        Whether to disable dynamic column scaling during pivoting for IAM1 method (default True)
        
    Returns:
    --------
    pandas.DataFrame
        Feasibility results for each method
    """
    problem_name = data['name']
    A = data['A']
    b = data['b']
    
    if isinstance(b, np.ndarray) and len(b.shape) > 1:
        b = b.flatten()
    
    rows, cols = A.shape
    original_shape = data.get('original_shape', (rows, cols))
    nonzeros = np.count_nonzero(A)
    sparsity = nonzeros / A.size
    
    print(f"Problem: {problem_name}")
    if VERBOSE:
        print(f"Current dimensions (inequality form): {A.shape}")
        print(f"Original dimensions (before conversion): {original_shape}")
        print(f"Size change: {original_shape[0]}x{original_shape[1]} → {A.shape[0]}x{A.shape[1]}")
        print(f"Sparsity: {nonzeros}/{A.size} = {sparsity:.6f}")
    
    results = {
        'problem': problem_name,
        'rows': rows,
        'cols': cols,
        'original_rows': original_shape[0],
        'original_cols': original_shape[1],
        'nonzeros': nonzeros,
        'sparsity': sparsity
    }
    
    simplex_status = None
    highs_status = None
    
    for method, max_iter in methods_and_limits.items():
        print(f"\nChecking feasibility with {method} method (max iterations: {max_iter})...")
        try:
            # Pass all scaling parameters only for IAM1 method
            if method == 'iam1':
                result = solve_with_method(
                    method, A, b, max_iter=max_iter, tol=tol, 
                    disable_row_scaling=disable_row_scaling, 
                    disable_dynamic_row_scaling=disable_dynamic_row_scaling,
                    disable_column_scaling=disable_column_scaling,
                    disable_dynamic_column_scaling=disable_dynamic_column_scaling
                )
            else:
                result = solve_with_method(method, A, b, max_iter=max_iter, tol=tol)
            
            print(f"Solver: {result['solver']}")
            print(f"Feasible: {result['is_feasible']}")
            
            if method == 'highs':
                print(f"Simplex iterations: {result['simplex_iterations']}")
            else:
                print(f"Iterations: {result['iterations']}")
                
            print(f"Time: {result['time']:.6f} seconds")
            if VERBOSE:
                print(f"Status: {result['status']}")
                
            results[f'{method}_iterations'] = result['iterations']
            if method == 'highs':
                results[f'{method}_simplex_iterations'] = result['simplex_iterations']
            results[f'{method}_time'] = result['time']
            results[f'{method}_feasible'] = result['is_feasible']
            results[f'{method}_status'] = result['status']
            
            if method == 'simplex':
                simplex_status = 'feasible' if result['is_feasible'] else 'infeasible'
            elif method == 'highs':
                highs_status = 'feasible' if result['is_feasible'] else 'infeasible'
                
        except Exception as e:
            print(f"Error processing method {method}: {str(e)}")
            results[f'{method}_iterations'] = -1
            if method == 'highs':
                results[f'{method}_simplex_iterations'] = -1
            results[f'{method}_time'] = -1
            results[f'{method}_feasible'] = False
            results[f'{method}_status'] = f"Error: {str(e)}"
    
    if simplex_status is not None:
        for method in methods_and_limits.keys():
            if method != 'simplex':
                method_status = 'feasible' if results.get(f'{method}_feasible', False) else 'infeasible'
                results[f'{method}_agrees_with_simplex'] = method_status == simplex_status
    
    if highs_status is not None:
        for method in methods_and_limits.keys():
            if method != 'highs':
                method_status = 'feasible' if results.get(f'{method}_feasible', False) else 'infeasible'
                results[f'{method}_agrees_with_highs'] = method_status == highs_status
    
    return results

def select_multiple_problems(problems):
    """Allow user to select multiple problems from the list."""
    selected_problems = []
    PAGE_SIZE = 20
    total_pages = (len(problems) + PAGE_SIZE - 1) // PAGE_SIZE
    page = 1
    
    while True:
        clear_screen()
        print(f"Available problems (Page {page}/{total_pages}):")
        print("-" * 60)
        print("Currently selected problems:", ", ".join(selected_problems) if selected_problems else "None")
        print("-" * 60)
        
        start_idx = (page - 1) * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, len(problems))
        
        for i, problem in enumerate(problems[start_idx:end_idx], start=start_idx + 1):
            status = "[X]" if problem in selected_problems else "[ ]"
            print(f"{i}. {status} {problem}")
        
        print("-" * 60)
        print("Commands:")
        print("- Enter a number to toggle problem selection")
        print("- 'n' for next page")
        print("- 'p' for previous page")
        print("- 'a' to select all problems")
        print("- 'd' to deselect all")
        print("- Press Enter (empty input) to finish selection")
        print("- 'q' to return to main menu")
        
        selection = input("Your selection: ").lower()
        
        if selection == 'q':
            return None
        elif selection == 'n':
            page = min(page + 1, total_pages)
        elif selection == 'p':
            page = max(page - 1, 1)
        elif selection == 'd':
            selected_problems = []
        elif selection == 'a':
            selected_problems = problems.copy()
        elif selection == '' and selected_problems:
            return selected_problems
        else:
            try:
                idx = int(selection) - 1
                if 0 <= idx < len(problems):
                    problem = problems[idx]
                    if problem in selected_problems:
                        selected_problems.remove(problem)
                    else:
                        selected_problems.append(problem)
            except ValueError:
                continue

def select_methods_and_limits(available_methods):
    """Allow user to select methods and set their iteration limits."""
    selected_methods = {}
    
    while True:
        clear_screen()
        print("Available methods:")
        print("-" * 60)
        print("Currently selected methods:")
        for method, limit in selected_methods.items():
            print(f"- {method} (max iterations: {limit})")
        print("-" * 60)
        
        for i, method in enumerate(available_methods, 1):
            status = "[X]" if method in selected_methods else "[ ]"
            print(f"{i}. {status} {method}")
        
        print("-" * 60)
        print("Commands:")
        print("- Enter a number to toggle method selection")
        print("- 'l' to set iteration limits for selected methods")
        print("- 'a' to select all methods")
        print("- 'd' to deselect all")
        print("- Press Enter (empty input) to finish selection")
        print("- 'q' to return to main menu")
        
        selection = input("Your selection: ").lower()
        
        if selection == 'q':
            return None
        elif selection == 'd':
            selected_methods.clear()
        elif selection == 'a':
            # Select all methods with default limits
            for method in available_methods:
                if method not in selected_methods:
                    default_limit = MAX_ITERATIONS
                    selected_methods[method] = default_limit
        elif selection == '' and selected_methods:
            return selected_methods
        elif selection == 'l':
            for method in list(selected_methods.keys()):
                try:
                    limit = int(input(f"Enter max iterations for {method} (current: {selected_methods[method]}): "))
                    if limit > 0:
                        selected_methods[method] = limit
                except ValueError:
                    print(f"Invalid input. Keeping current limit for {method}")
        else:
            try:
                idx = int(selection) - 1
                if 0 <= idx < len(available_methods):
                    method = available_methods[idx]
                    if method in selected_methods:
                        del selected_methods[method]
                    else:
                        # Set default iteration limit based on method type - use MAX_ITERATIONS for all methods
                        default_limit = MAX_ITERATIONS
                        limit = default_limit
                        try:
                            user_limit = input(f"Enter max iterations for {method} (default: {default_limit}): ")
                            if user_limit.strip():
                                limit = int(user_limit)
                                if limit <= 0:
                                    raise ValueError
                        except ValueError:
                            print(f"Invalid input. Using default limit: {default_limit}")
                        selected_methods[method] = limit
            except ValueError:
                continue

def run_batch_comparison(problems, problem_dir, output_file, methods_and_limits, max_iter=MAX_ITERATIONS, tol=ZERO_TOLERANCE, resume=False, 
                      disable_row_scaling=False, disable_dynamic_row_scaling=False,
                      disable_column_scaling=True, disable_dynamic_column_scaling=True):
    """
    Run batch comparison of multiple problems with multiple methods.
    
    Parameters:
    -----------
    problems : list
        List of problem names
    problem_dir : str
        Directory containing problem files
    output_file : str
        File to save results to
    methods_and_limits : dict
        Dictionary mapping method names to iteration limits
    max_iter : int
        Maximum number of iterations (default from global config)
    tol : float
        Numerical tolerance (default from global config)
    resume : bool
        Whether to resume from a previous run
    disable_row_scaling : bool
        Whether to disable initial row scaling for IAM1 method (default False)
    disable_dynamic_row_scaling : bool
        Whether to disable dynamic row scaling during pivoting for IAM1 method (default False)
    disable_column_scaling : bool
        Whether to disable initial column scaling for IAM1 method (default True)
    disable_dynamic_column_scaling : bool
        Whether to disable dynamic column scaling during pivoting for IAM1 method (default True)
        
    Returns:
    --------
    pandas.DataFrame
        Comparison results
    """
    # Setup DataFrame for storing results
    if os.path.exists(output_file):
        print(f"Loading existing results file: {output_file}")
        output_df = pd.read_csv(output_file)
        existing_problems = set(output_df['problem'].tolist())
    else:
        # Create empty DataFrame with columns for each method
        columns = ['problem', 'rows', 'cols', 'original_rows', 'original_cols', 'nonzeros', 'sparsity']
        for method in methods_and_limits.keys():
            # Add specific simplex_iterations column for highs
            if method == 'highs':
                columns.extend([
                    f'{method}_iterations', 
                    f'{method}_simplex_iterations',
                    f'{method}_time', 
                    f'{method}_feasible', 
                    f'{method}_status'
                ])
            else:
                columns.extend([
                    f'{method}_iterations', 
                    f'{method}_time', 
                    f'{method}_feasible', 
                    f'{method}_status'
                ])
                
            if method != 'simplex':
                columns.append(f'{method}_agrees_with_simplex')
        output_df = pd.DataFrame(columns=columns)
        existing_problems = set()
    
    for i, problem_name in enumerate(problems):
        # Only process problems that are not in the existing data when resuming
        if problem_name in existing_problems and resume:
            print(f"\nSkipping already processed problem: {problem_name}")
            continue
            
        print(f"\n{'-'*50}")
        print(f"Processing {problem_name} ({i+1}/{len(problems)})...")
        
        # Try different filename patterns
        problem_file = os.path.join(problem_dir, f"{problem_name}.pkl")
        if not os.path.exists(problem_file):
            problem_file = os.path.join(problem_dir, f"NAME_{problem_name}.pkl")
            if not os.path.exists(problem_file):
                problem_file = os.path.join(problem_dir, f"NAME_{problem_name}_reduced.pkl")
                if not os.path.exists(problem_file):
                    print(f"Problem file not found for {problem_name}, skipping...")
                    continue
        
        try:
            data = load_problem(problem_file)
            results = check_problem_feasibility(
                data, 
                methods_and_limits, 
                tol, 
                disable_row_scaling, 
                disable_dynamic_row_scaling,
                disable_column_scaling,
                disable_dynamic_column_scaling
            )
            
            # Add new row for this problem
            output_df = pd.concat([output_df, pd.DataFrame([results])], ignore_index=True)
            existing_problems.add(problem_name)
            print(f"Added new results for {problem_name}")
            
            # Save after each problem
            output_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"Error processing {problem_name}: {str(e)}")
    
    return output_df

def create_summary_visualizations(output_file, output_dir):
    """Create visualizations comparing the different methods."""
    # Check if visualization libraries are available
    try:
        import matplotlib
        import seaborn
    except ImportError:
        print("Warning: matplotlib or seaborn not available. Cannot create visualizations.")
        print("Please install them with: pip install matplotlib seaborn")
        return
        
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the results
        df = pd.read_csv(output_file)
        
        if len(df) == 0:
            print("No data to visualize.")
            return
            
        # Extract method names directly from column names
        time_cols = [col for col in df.columns if col.endswith('_time')]
        methods = []
        for col in time_cols:
            method = col.replace('_time', '')
            if f"{method}_iterations" in df.columns and f"{method}_feasible" in df.columns:
                methods.append(method)
        
        print(f"Detected methods for visualization: {methods}")
        
        if not methods:
            print("No valid methods found in the results file.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Time comparison (box plot)
        time_data = []
        for method in methods:
            col = f'{method}_time'
            method_data = df[df[col] > 0][[col, 'problem']]
            if not method_data.empty:
                method_data['method'] = method
                method_data.rename(columns={col: 'time'}, inplace=True)
                time_data.append(method_data[['problem', 'method', 'time']])
        
        if time_data:
            time_df = pd.concat(time_data)
            sns.boxplot(x='method', y='time', data=time_df, ax=axes[0, 0])
            axes[0, 0].set_title('Solving Time Comparison')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].set_xlabel('Method')
        
        # 2. Iterations comparison (box plot)
        iter_data = []
        for method in methods:
            col = f'{method}_iterations'
            method_data = df[df[col] > 0][[col, 'problem']]
            if not method_data.empty:
                method_data['method'] = method
                method_data.rename(columns={col: 'iterations'}, inplace=True)
                iter_data.append(method_data[['problem', 'method', 'iterations']])
        
        if iter_data:
            iter_df = pd.concat(iter_data)
            sns.boxplot(x='method', y='iterations', data=iter_df, ax=axes[0, 1])
            axes[0, 1].set_title('Iterations Comparison')
            axes[0, 1].set_ylabel('Number of Iterations')
            axes[0, 1].set_xlabel('Method')
        
        # 3. Feasibility success rate
        feasible_counts = []
        for method in methods:
            col = f'{method}_feasible'
            if col in df.columns:
                success_rate = df[col].mean() * 100
                feasible_counts.append({'method': method, 'success_rate': success_rate})
        
        if feasible_counts:
            feasible_df = pd.DataFrame(feasible_counts)
            sns.barplot(x='method', y='success_rate', data=feasible_df, ax=axes[1, 0])
            axes[1, 0].set_title('Feasibility Success Rate')
            axes[1, 0].set_ylabel('Success Rate (%)')
            axes[1, 0].set_xlabel('Method')
            axes[1, 0].set_ylim(0, 100)
        
        # 4. Agreement with simplex
        agreement_counts = []
        for method in methods:
            if method == 'simplex':
                continue
            col = f'{method}_agrees_with_simplex'
            if col in df.columns:
                agreement_rate = df[col].mean() * 100
                agreement_counts.append({'method': method, 'agreement_rate': agreement_rate})
        
        if agreement_counts:
            agreement_df = pd.DataFrame(agreement_counts)
            sns.barplot(x='method', y='agreement_rate', data=agreement_df, ax=axes[1, 1])
            axes[1, 1].set_title('Agreement with Simplex')
            axes[1, 1].set_ylabel('Agreement Rate (%)')
            axes[1, 1].set_xlabel('Method')
            axes[1, 1].set_ylim(0, 100)
        
        # Adjust layout and save
        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'method_comparison.png')
        plt.savefig(fig_path)
        print(f"Visualization saved to {fig_path}")
        
        # Create detailed time comparison by problem size
        plt.figure(figsize=(15, 10))
        ax = plt.subplot(111)
        
        # Get time ratios compared to simplex
        for method in methods:
            if method != 'simplex':
                df[f'{method}_ratio'] = np.where(
                    df['simplex_time'] > 0, 
                    df[f'{method}_time'] / df['simplex_time'],
                    np.nan
                )
        
        # Plot ratio data
        for method in [m for m in methods if m != 'simplex']:
            ratio_col = f'{method}_ratio'
            valid_data = df[(df[ratio_col] > 0) & (df[ratio_col] < 10)]  # Filter extreme values
            if not valid_data.empty:
                ax.plot(valid_data['problem'], valid_data[ratio_col], 'o-', label=f'{method} vs simplex')
        
        ax.axhline(y=1, color='r', linestyle='--', label='Equal to simplex')
        ax.set_title('Method Speed Relative to Simplex (lower is better)')
        ax.set_ylabel('Time Ratio (method time / simplex time)')
        ax.set_xlabel('Problem')
        ax.set_xticks(range(len(df['problem'])))
        ax.set_xticklabels(df['problem'], rotation=90)
        ax.legend()
        
        # Save ratio chart
        plt.tight_layout()
        ratio_path = os.path.join(output_dir, 'time_ratio_comparison.png')
        plt.savefig(ratio_path)
        print(f"Ratio visualization saved to {ratio_path}")
        
        # Create a new figure for HiGHS comparison
        if 'highs' in methods:
            plt.figure(figsize=(15, 10))
            ax = plt.subplot(111)
            
            # Get time ratios compared to HiGHS
            for method in methods:
                if method != 'highs':
                    df[f'{method}_vs_highs_ratio'] = np.where(
                        df['highs_time'] > 0, 
                        df[f'{method}_time'] / df['highs_time'],
                        np.nan
                    )
            
            # Plot ratio data
            for method in [m for m in methods if m != 'highs']:
                ratio_col = f'{method}_vs_highs_ratio'
                valid_data = df[(df[ratio_col] > 0) & (df[ratio_col] < 10)]  # Filter extreme values
                if not valid_data.empty:
                    ax.plot(valid_data['problem'], valid_data[ratio_col], 'o-', label=f'{method} vs highs')
            
            ax.axhline(y=1, color='r', linestyle='--', label='Equal to HiGHS')
            ax.set_title('Method Speed Relative to HiGHS (lower is better)')
            ax.set_ylabel('Time Ratio (method time / highs time)')
            ax.set_xlabel('Problem')
            ax.set_xticks(range(len(df['problem'])))
            ax.set_xticklabels(df['problem'], rotation=90)
            ax.legend()
            
            # Save ratio chart
            plt.tight_layout()
            highs_ratio_path = os.path.join(output_dir, 'time_ratio_vs_highs.png')
            plt.savefig(highs_ratio_path)
            print(f"HiGHS ratio visualization saved to {highs_ratio_path}")
            
            # Create a figure for agreement with HiGHS
            plt.figure(figsize=(12, 8))
            
            # Agreement with HiGHS
            highs_agreement_counts = []
            for method in methods:
                if method == 'highs':
                    continue
                col = f'{method}_agrees_with_highs'
                if col in df.columns:
                    agreement_rate = df[col].mean() * 100
                    highs_agreement_counts.append({'method': method, 'agreement_rate': agreement_rate})
            
            if highs_agreement_counts:
                highs_agreement_df = pd.DataFrame(highs_agreement_counts)
                sns.barplot(x='method', y='agreement_rate', data=highs_agreement_df)
                plt.title('Agreement with HiGHS')
                plt.ylabel('Agreement Rate (%)')
                plt.xlabel('Method')
                plt.ylim(0, 100)
                
                plt.tight_layout()
                highs_agreement_path = os.path.join(output_dir, 'agreement_with_highs.png')
                plt.savefig(highs_agreement_path)
                print(f"HiGHS agreement visualization saved to {highs_agreement_path}")
        
        # Create a line graph comparing iterations for each problem across all methods
        plt.figure(figsize=(16, 10))
        
        # Create a line plot for each method showing iterations across all problems
        for method in methods:
            iter_col = f'{method}_iterations'
            # Filter out problems where the method didn't run or had negative iterations
            valid_data = df[df[iter_col] > 0]
            if not valid_data.empty:
                # Sort by problem name for consistency
                valid_data = valid_data.sort_values('problem')
                plt.plot(valid_data['problem'], valid_data[iter_col], 'o-', label=method, linewidth=2, markersize=8)
        
        plt.title('Number of Iterations by Method for Each Problem', fontsize=16)
        plt.ylabel('Number of Iterations', fontsize=14)
        plt.xlabel('Problem', fontsize=14)
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Add some padding to prevent clipping
        plt.tight_layout(pad=3)
        
        # Save the iterations comparison line graph
        iterations_path = os.path.join(output_dir, 'iterations_comparison_line.png')
        plt.savefig(iterations_path)
        print(f"Iterations comparison line graph saved to {iterations_path}")
        
        # Create a log-scale version for better visibility with wide-ranging iteration counts
        plt.figure(figsize=(16, 10))
        
        for method in methods:
            iter_col = f'{method}_iterations'
            valid_data = df[df[iter_col] > 0]
            if not valid_data.empty:
                valid_data = valid_data.sort_values('problem')
                plt.plot(valid_data['problem'], valid_data[iter_col], 'o-', label=method, linewidth=2, markersize=8)
        
        plt.title('Number of Iterations by Method (Log Scale)', fontsize=16)
        plt.ylabel('Number of Iterations (log scale)', fontsize=14)
        plt.xlabel('Problem', fontsize=14)
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        plt.tight_layout(pad=3)
        
        # Save the log-scale iterations comparison
        log_iterations_path = os.path.join(output_dir, 'iterations_comparison_log.png')
        plt.savefig(log_iterations_path)
        print(f"Log-scale iterations comparison saved to {log_iterations_path}")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        import traceback
        print(traceback.format_exc())

def create_sorted_batches(problem_dir, batch_size=10):
    """Create batches of problems sorted by file size (from smallest to largest)."""
    all_problems = list_available_problems(problem_dir)
    if not all_problems:
        return []
    
    # Get file sizes for each problem
    problem_sizes = []
    for problem in all_problems:
        # Try to find the problem file
        problem_file = os.path.join(problem_dir, f"{problem}.pkl")
        if not os.path.exists(problem_file):
            problem_file = os.path.join(problem_dir, f"NAME_{problem}.pkl")
            if not os.path.exists(problem_file):
                continue
        
        # Get file size in KB
        file_size = os.path.getsize(problem_file) / 1024
        problem_sizes.append((problem, file_size))
    
    # Sort by file size (smallest to largest)
    problem_sizes.sort(key=lambda x: x[1])
    
    # Divide into batches
    batches = []
    for i in range(0, len(problem_sizes), batch_size):
        batch = problem_sizes[i:i+batch_size]
        batches.append(batch)
    
    return batches

def display_batch_menu(batches):
    """Display menu for selecting a batch of problems."""
    clear_screen()
    print("=" * 60)
    print("SELECT BATCH TO PROCESS".center(60))
    print("=" * 60)
    print(f"Found {len(batches)} batches of problems")
    print("-" * 60)
    
    for i, batch in enumerate(batches):
        # Calculate total size of the batch (in KB and MB)
        total_kb = sum(size for _, size in batch)
        if total_kb >= 1024:
            size_str = f"{total_kb/1024:.2f} MB"
        else:
            size_str = f"{total_kb:.2f} KB"
        
        # Get size range
        min_size = min(size for _, size in batch)
        max_size = max(size for _, size in batch)
        
        if min_size >= 1024:
            min_size_str = f"{min_size/1024:.2f} MB"
        else:
            min_size_str = f"{min_size:.2f} KB"
            
        if max_size >= 1024:
            max_size_str = f"{max_size/1024:.2f} MB"
        else:
            max_size_str = f"{max_size:.2f} KB"
        
        print(f"Batch {i+1}: {len(batch)} problems, Size: {size_str}")
        print(f"    Range: {min_size_str} - {max_size_str}")
        print(f"    Problems: {', '.join(problem for problem, _ in batch)}")
        print()
    
    print("-" * 60)
    print("Enter batch number to select, or 0 to return to main menu")
    choice = input("Your selection: ")
    
    if choice.isdigit():
        batch_num = int(choice)
        if 1 <= batch_num <= len(batches):
            return batches[batch_num-1]
    
    return None

def process_batch(batch, problem_dir, methods_and_limits, options, output_dir, 
                 disable_row_scaling=False, disable_dynamic_row_scaling=False,
                 disable_column_scaling=True, disable_dynamic_column_scaling=True):
    """Process a batch of problems."""
    # Extract just the problem names from the batch (remove size information)
    problems = [problem for problem, _ in batch]
    
    # Set up the output file to include the batch information
    batch_problems_str = '_'.join(problems[:3])  # Use first 3 problem names in filename
    if len(problems) > 3:
        batch_problems_str += f"_and_{len(problems)-3}_more"
    
    output_file = os.path.join(output_dir, f"batch_{batch_problems_str}_maxiter{options['max_iter']}_results.csv")
    
    # Initialize resume to True by default to only process missing problems
    resume = True
    
    # Ask about visualizing results - default to no
    visualize = input("\nCreate visualizations after running? (y/n, default=n): ")
    visualize = visualize.lower() == 'y'
    
    print(f"\nRunning comparison on batch of {len(problems)} problems...")
    print(f"IAM1 initial row scaling: {'Disabled' if disable_row_scaling else 'Enabled'}")
    print(f"IAM1 dynamic row scaling: {'Disabled' if disable_dynamic_row_scaling else 'Enabled'}")
    print(f"IAM1 initial column scaling: {'Disabled' if disable_column_scaling else 'Enabled'}")
    print(f"IAM1 dynamic column scaling: {'Disabled' if disable_dynamic_column_scaling else 'Enabled'}")
    output_df = run_batch_comparison(
        problems,
        problem_dir,
        output_file,
        methods_and_limits,
        options['max_iter'],
        options['tol'],
        resume,
        disable_row_scaling,
        disable_dynamic_row_scaling,
        disable_column_scaling,
        disable_dynamic_column_scaling
    )
    
    # Create visualizations if requested
    if visualize:
        create_summary_visualizations(output_file, output_dir)
    
    return output_file

def display_main_menu(disable_row_scaling, disable_dynamic_row_scaling, disable_column_scaling, disable_dynamic_column_scaling):
    """Display the main menu and get user choice."""
    clear_screen()
    print("=" * 60)
    print("IAM-1 VARIANTS NETLIB LP COMPARISON MENU".center(60))
    print("=" * 60)
    print("FEASIBILITY CHECKING ONLY - NO OPTIMIZATION".center(60))
    print("-" * 60)
    print("1. Run small batch of test problems")
    print("2. Run all available problems")
    print("3. Select a specific problem")
    print("4. Visualize existing results")
    print("5. Change PKL files directory")
    print("6. Toggle quiet mode (currently " + ("OFF" if VERBOSE else "ON") + ")")
    print("7. Run problems by size-sorted batches")
    print("8. Toggle IAM1 initial row scaling (currently " + ("OFF" if disable_row_scaling else "ON") + ")")
    print("9. Toggle IAM1 dynamic row scaling (currently " + ("OFF" if disable_dynamic_row_scaling else "ON") + ")")
    print("10. Toggle IAM1 initial column scaling (currently " + ("OFF" if disable_column_scaling else "ON") + ")")
    print("11. Toggle IAM1 dynamic column scaling (currently " + ("OFF" if disable_dynamic_column_scaling else "ON") + ")")
    print("12. Run a single example from examples.py (with detailed output)")
    print("13. Run all examples from examples.py in sequence")
    print("14. Exit")
    print("-" * 60)
    print("Note: Only simplex, highs, iam1, and pan_row_scaled methods are enabled")
    print("Note: 'simplex' refers to our custom simplex_phase_1 implementation")
    print("-" * 60)
    choice = input("Enter your choice (1-14): ")
    return choice

def select_advanced_options(output_dir):
    """Get advanced options from user."""
    clear_screen()
    print("Advanced Options:")
    print("-" * 60)
    print(f"1. Maximum iterations (default: {MAX_ITERATIONS})")
    print(f"2. Tolerance (default: {ZERO_TOLERANCE})")
    print("3. Output file name (default: netlib_feasibility_results_maxiterXXXX.csv)")
    print("4. Continue with defaults")
    print("-" * 60)
    
    options = {
        'max_iter': MAX_ITERATIONS,
        'tol': ZERO_TOLERANCE,
        'output_file': os.path.join(output_dir, f"netlib_feasibility_results_maxiter{MAX_ITERATIONS}.csv")
    }
    
    choice = input("Select an option (1-4): ")
    
    if choice == '1':
        try:
            max_iter = int(input(f"Enter maximum iterations (default: {MAX_ITERATIONS}): "))
            if max_iter > 0:
                options['max_iter'] = max_iter
                # Update output filename with new max_iter if using default name
                if options['output_file'] == os.path.join(output_dir, f"netlib_feasibility_results_maxiter{MAX_ITERATIONS}.csv"):
                    options['output_file'] = os.path.join(output_dir, f"netlib_feasibility_results_maxiter{max_iter}.csv")
        except ValueError:
            print(f"Invalid value. Using default: {MAX_ITERATIONS}")
            time.sleep(1)
    
    elif choice == '2':
        try:
            tol = float(input(f"Enter tolerance (default: {ZERO_TOLERANCE}): "))
            if tol > 0:
                options['tol'] = tol
        except ValueError:
            print(f"Invalid value. Using default: {ZERO_TOLERANCE}")
            time.sleep(1)
    
    elif choice == '3':
        output_file = input(f"Enter output file name (default: netlib_feasibility_results_maxiter{options['max_iter']}.csv): ").strip()
        if output_file:
            # Make the path relative to the output_dir
            options['output_file'] = os.path.join(output_dir, output_file)
    
    return options

def solve_with_simplex(A, b, method_name, method_max_iter, tol, verbose=False):
    """
    Solve linear program using standard simplex method (Phase 1).
    
    Parameters:
    -----------
    A : array_like
        The constraint matrix
    b : array_like
        The right-hand side vector
    method_name : str
        The name of the simplex method (for output)
    method_max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for numerical calculations
    verbose : bool
        Whether to print detailed information
        
    Returns:
    --------
    dict
        Result information including status, iterations, time, and solution (if found)
    """
    try:
        # Call simplex solver with sparse matrices for large problems
        result = solve_with_simplex_phase1(A, b, max_iter=method_max_iter, tol=tol, verbose=VERBOSE, use_sparse=True)
        
        # Format result
        status = result["status"]
        if status == "feasible":
            return {
                "status": "success",
                "iterations": result["iterations"],
                "time": result["time"],
                "message": f"Simplex successfully found a feasible solution after {result['iterations']} iterations",
                "cycling_detected": result.get("cycling_detected", False)
            }
        elif status == "cycling_skipped":
            return {
                "status": "cycling",
                "iterations": result["iterations"],
                "time": result["time"],
                "message": f"Cycling detected during Simplex, skipped after {result['iterations']} iterations",
                "cycling_detected": True,
                "cycle_length": result.get("cycle_length", 0)
            }
        elif status == "infeasible":
            return {
                "status": "infeasible",
                "iterations": result["iterations"],
                "time": result["time"],
                "message": f"Problem is infeasible (confirmed by Simplex after {result['iterations']} iterations)"
            }
        elif status == "max_iterations":
            return {
                "status": "max_iterations",
                "iterations": result["iterations"],
                "time": result["time"],
                "message": f"Simplex reached maximum iterations ({result['iterations']})"
            }
        elif status == "skipped":
            return {
                "status": "skipped",
                "iterations": result["iterations"],
                "time": result["time"],
                "message": f"Simplex Phase 1 skipped by user after {result['iterations']} iterations"
            }
        else:
            return {
                "status": "error",
                "iterations": result.get("iterations", 0),  # Get iterations or default to 0
                "message": f"Simplex returned unknown status: {status}",
                "time": result["time"]
            }
    except Exception as e:
        # Handle memory errors specifically
        if isinstance(e, MemoryError):
            error_message = f"Memory error in simplex: {str(e)}"
            print(f"\nERROR: {error_message}")
            return {
                "status": "error",
                "iterations": 0,  # Add iterations field with default value
                "message": error_message,
                "time": 0
            }
        else:
            error_message = f"Error in simplex: {str(e)}"
            print(f"\nERROR: {error_message}")
            return {
                "status": "error",
                "iterations": 0,  # Add iterations field with default value
                "message": error_message,
                "time": 0
            }

def main():
    # Print global configuration info
    print("\n" + "="*60)
    print("GLOBAL CONFIGURATION".center(60))
    print("="*60)
    print(get_netlib_config_info())
    print("="*60 + "\n")
    
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_problem_dir = os.path.join(script_dir, 'Netlib_files', 'inequality_form_splited')
    
    # Prompt for Netlib problems directory
    print("\n" + "="*60)
    print("NETLIB PROBLEMS DIRECTORY SELECTION".center(60))
    print("="*60)
    print(f"\nDefault directory for PKL files: {default_problem_dir}")
    print("Enter a custom directory path or press Enter to use the default:")
    custom_dir = input().strip()
    
    if custom_dir:
        problem_dir = custom_dir
    else:
        problem_dir = default_problem_dir
        
    print(f"\nUsing Netlib problems directory: {problem_dir}")
    
    # Create output directory inside the specified problem directory
    output_dir = os.path.join(problem_dir, 'computational_results_cpu')
    print(f"Computational results and graphs will be saved in: {output_dir}")
    print("="*60 + "\n")
    
    # Check if problem directory exists
    if not os.path.exists(problem_dir):
        print(f"Error: Problem directory not found: {problem_dir}")
        print("Please ensure the directory exists and contains PKL files.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define methods to compare - Enable only simplex, highs, iam-1, pan's method
    methods_with_limits = {
        "simplex": MAX_ITERATIONS,
        "highs": None,  # No iteration limit for HiGHS
        "iam1": MAX_ITERATIONS,
        "pan_row_scaled": MAX_ITERATIONS
    }
    
    # For backward compatibility (some parts of code might still use the list)
    methods = list(methods_with_limits.keys())
    
    # Small batch of problems for testing
    SMALL_PROBLEMS = ['AFIRO','ADLITTLE','SCSD1', 'SHIP08S', 'BRANDY', 'SHIP12S', 'FFFFF800']
    
    # Allow access to the global VERBOSE flag
    global VERBOSE
    
    # Create flags for IAM1 scaling options
    disable_row_scaling = not config_iam.ENABLE_ROW_SCALING  # By default, use the configuration value
    disable_dynamic_row_scaling = not config_iam.ENABLE_DYNAMIC_ROW_SCALING  # By default, use the configuration value
    disable_column_scaling = True  # Default to disabled since it's a new feature
    disable_dynamic_column_scaling = True  # Default to disabled since it's a new feature
    
    while True:
        choice = display_main_menu(disable_row_scaling, disable_dynamic_row_scaling, disable_column_scaling, disable_dynamic_column_scaling)
        
        if choice == '1':
            # Run small batch
            print(f"Testing on a batch of {len(SMALL_PROBLEMS)} small problems")
            
            # Ask for advanced options
            advanced = input("\nDo you want to set advanced options? (y/n): ")
            if advanced.lower() == 'y':
                options = select_advanced_options(output_dir)
            else:
                options = {
                    'max_iter': MAX_ITERATIONS,
                    'tol': ZERO_TOLERANCE,
                    'output_file': os.path.join(output_dir, "netlib_small_batch_results.csv")
                }
                
            # Initialize resume to True by default to only process missing problems
            resume = True
                
            # Ask about visualizing results - default to no
            visualize = input("\nCreate visualizations after running? (y/n, default=n): ")
            visualize = visualize.lower() == 'y'
            
            print("\nRunning comparison on small batch...")
            print(f"IAM1 initial row scaling: {'Disabled' if disable_row_scaling else 'Enabled'}")
            print(f"IAM1 dynamic row scaling: {'Disabled' if disable_dynamic_row_scaling else 'Enabled'}")
            print(f"IAM1 initial column scaling: {'Disabled' if disable_column_scaling else 'Enabled'}")
            print(f"IAM1 dynamic column scaling: {'Disabled' if disable_dynamic_column_scaling else 'Enabled'}")
            output_df = run_batch_comparison(
                SMALL_PROBLEMS, 
                problem_dir, 
                options['output_file'], 
                methods_with_limits, 
                options['max_iter'], 
                options['tol'], 
                resume,
                disable_row_scaling,
                disable_dynamic_row_scaling,
                disable_column_scaling,
                disable_dynamic_column_scaling
            )
            
            if visualize:
                create_summary_visualizations(options['output_file'], output_dir)
                
            input("\nPress Enter to return to menu...")
            
        elif choice == '2':
            # Run all problems
            print("\nWARNING: Running all problems may take a very long time.")
            confirm = input("Are you sure you want to continue? (y/n): ")
            
            if confirm.lower() == 'y':
                # Get list of all problems
                all_problems = list_available_problems(problem_dir)
                
                if not all_problems:
                    print("No problems found.")
                    input("\nPress Enter to return to menu...")
                    continue
                
                print(f"Found {len(all_problems)} problems.")
                
                # Ask for advanced options
                advanced = input("\nDo you want to set advanced options? (y/n): ")
                if advanced.lower() == 'y':
                    options = select_advanced_options(output_dir)
                else:
                    options = {
                        'max_iter': MAX_ITERATIONS,
                        'tol': ZERO_TOLERANCE,
                        'output_file': os.path.join(output_dir, "netlib_all_results.csv")
                    }
                
                # Initialize resume to True by default to only process missing problems
                resume = True
                    
                # Ask if they want to create visualizations - default to no
                visualize = input("\nCreate visualizations after running? (y/n, default=n): ")
                visualize = visualize.lower() == 'y'
                
                print("\nRunning comparison on all problems...")
                print(f"IAM1 initial row scaling: {'Disabled' if disable_row_scaling else 'Enabled'}")
                print(f"IAM1 dynamic row scaling: {'Disabled' if disable_dynamic_row_scaling else 'Enabled'}")
                print(f"IAM1 initial column scaling: {'Disabled' if disable_column_scaling else 'Enabled'}")
                print(f"IAM1 dynamic column scaling: {'Disabled' if disable_dynamic_column_scaling else 'Enabled'}")
                output_df = run_batch_comparison(
                    all_problems, 
                    problem_dir, 
                    options['output_file'], 
                    methods_with_limits, 
                    options['max_iter'], 
                    options['tol'], 
                    resume,
                    disable_row_scaling,
                    disable_dynamic_row_scaling,
                    disable_column_scaling,
                    disable_dynamic_column_scaling
                )
                
                if visualize:
                    create_summary_visualizations(options['output_file'], output_dir)
                    
                input("\nPress Enter to return to menu...")
            
        elif choice == '3':
            # Select specific problems
            problems = list_available_problems(problem_dir)
            
            if not problems:
                print("No problems found.")
                input("\nPress Enter to return to menu...")
                continue
            
            # Select multiple problems
            selected_problems = select_multiple_problems(problems)
            if not selected_problems:
                continue
            
            # Select methods and their iteration limits - only show enabled methods
            available_methods = ["simplex", "highs", "iam1", "pan_row_scaled"]
            methods_and_limits = select_methods_and_limits(available_methods)
            if not methods_and_limits:
                    continue
                
            # Ask for tolerance
            tol = ZERO_TOLERANCE
            try:
                user_tol = input(f"\nEnter tolerance (default: {tol}): ")
                if user_tol.strip():
                    tol = float(user_tol)
            except ValueError:
                print(f"Invalid tolerance. Using default: {tol}")
            
            # Set output file
            timestamp = int(time.time())
            output_file = os.path.join(output_dir, f"custom_comparison_results_{timestamp}.csv")
                            
            # Ask about visualizing results - default to no
            visualize = input("\nCreate visualizations after running? (y/n, default=n): ")
            visualize = visualize.lower() == 'y'
            
            # Initialize resume to True to only process missing problems 
            resume = True
                        
            print(f"\nRunning comparison on {len(selected_problems)} problems...")
            print(f"IAM1 initial row scaling: {'Disabled' if disable_row_scaling else 'Enabled'}")
            print(f"IAM1 dynamic row scaling: {'Disabled' if disable_dynamic_row_scaling else 'Enabled'}")
            print(f"IAM1 initial column scaling: {'Disabled' if disable_column_scaling else 'Enabled'}")
            print(f"IAM1 dynamic column scaling: {'Disabled' if disable_dynamic_column_scaling else 'Enabled'}")
            output_df = run_batch_comparison(
                selected_problems,
                problem_dir, 
                output_file,
                methods_and_limits,
                tol=tol,
                resume=resume,
                disable_row_scaling=disable_row_scaling,
                disable_dynamic_row_scaling=disable_dynamic_row_scaling,
                disable_column_scaling=disable_column_scaling,
                disable_dynamic_column_scaling=disable_dynamic_column_scaling
            )
            
            if visualize:
                create_summary_visualizations(output_file, output_dir)
            
            input("\nPress Enter to return to menu...")
            
        elif choice == '4':
            # Handle visualize existing results option
            try:
                clear_screen()
                print("\nVisualize Existing Results:")
                print("-" * 60)
                print(f"1. Visualize default results (netlib_feasibility_results_maxiter{MAX_ITERATIONS}.csv)")
                print("2. Select a file to visualize")
                vis_choice = input("Enter your choice (1-2): ")

                if vis_choice == '1':
                    # Use timestamp for time-based output directory
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    output_file = os.path.join(output_dir, f"netlib_feasibility_results_maxiter{MAX_ITERATIONS}.csv")
                elif vis_choice == '2':
                    output_file_name = input("Enter the name of the results file to visualize: ")
                    output_file = os.path.join(output_dir, output_file_name)
                else:
                    # Default if user enters something else
                    output_file = os.path.join(output_dir, f"netlib_feasibility_results_maxiter{MAX_ITERATIONS}.csv")

                if not os.path.exists(output_file):
                    print(f"Error: Results file {output_file} not found.")
                    input("\nPress Enter to return to menu...")
                    continue
                
                create_summary_visualizations(output_file, output_dir)
                input("\nPress Enter to return to menu...")
            except Exception as e:
                print(f"Error processing visualization choice: {str(e)}")
                input("\nPress Enter to return to menu...")
            
        elif choice == '5':
            # Change PKL files directory
            clear_screen()
            print("\n" + "="*60)
            print("CHANGE PKL FILES DIRECTORY".center(60))
            print("="*60)
            print(f"\nCurrent PKL files directory: {problem_dir}")
            print("Enter a new directory path or press Enter to keep the current directory:")
            new_dir = input().strip()
            
            if new_dir:
                # Check if the new directory exists
                if os.path.exists(new_dir):
                    problem_dir = new_dir
                    print(f"\nChanging PKL files directory to: {problem_dir}")
                    
                    # Update the output directory to match the new problem directory
                    output_dir = os.path.join(problem_dir, 'computational_results_cpu')
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"Computational results will now be saved in: {output_dir}")
                    
                    # Check if the directory contains PKL files
                    pkl_files = [f for f in os.listdir(problem_dir) if f.endswith('.pkl')]
                    if pkl_files:
                        print(f"Found {len(pkl_files)} PKL files in the directory.")
                    else:
                        print("Warning: No PKL files found in this directory.")
                else:
                    print(f"\nError: Directory '{new_dir}' not found.")
            else:
                print("\nKeeping current PKL files directory.")
            
            print("="*60)
            input("\nPress Enter to return to menu...")
            
        elif choice == '6':
            # Toggle quiet mode
            VERBOSE = not VERBOSE
            print("\nQuiet mode is now " + ("OFF" if VERBOSE else "ON"))
            input("\nPress Enter to return to menu...")
            
        elif choice == '7':
            # Run problems by size-sorted batches
            clear_screen()
            print("\n" + "="*60)
            print("RUN PROBLEMS BY SIZE-SORTED BATCHES".center(60))
            print("="*60)
            
            # Create batches
            print("\nCreating batches of problems sorted by file size...")
            batch_size = input("Enter batch size (default: 10): ").strip()
            if batch_size.isdigit() and int(batch_size) > 0:
                batches = create_sorted_batches(problem_dir, int(batch_size))
            else:
                batches = create_sorted_batches(problem_dir)
            
            if not batches:
                print("No problems found to create batches.")
                input("\nPress Enter to return to menu...")
                continue
            
            # Ask for method selection
            print("\nMethod Selection:")
            print("1. Use default methods (simplex, highs, iam1, pan_row_scaled)")
            print("2. Select specific methods")
            method_choice = input("Enter your choice (1-2): ")
            
            # Define batch methods based on user choice
            batch_methods = {}
            if method_choice == '2':
                # Define all available methods
                all_available_methods = ["simplex", "highs", "iam1", "iam1b", 
                                       "iam1b_l1scaling", "pan_row_scaled", "iam1b_nocolscaling"]
                
                # Let user select methods
                batch_methods = select_methods_and_limits(all_available_methods)
                if not batch_methods:
                    print("No methods selected, using default methods.")
                    batch_methods = methods_with_limits
            else:
                # Use default methods
                batch_methods = methods_with_limits
            
            # Ask for advanced options
            advanced = input("\nDo you want to set advanced options? (y/n): ")
            if advanced.lower() == 'y':
                options = select_advanced_options(output_dir)
            else:
                options = {
                    'max_iter': MAX_ITERATIONS,
                    'tol': ZERO_TOLERANCE,
                    'output_file': None  # Will be set during batch processing
                }
            
            # Show batch selection menu
            selected_batch = display_batch_menu(batches)
            if selected_batch:
                # Process the selected batch with the selected methods
                print(f"IAM1 initial row scaling: {'Disabled' if disable_row_scaling else 'Enabled'}")
                print(f"IAM1 dynamic row scaling: {'Disabled' if disable_dynamic_row_scaling else 'Enabled'}")
                print(f"IAM1 initial column scaling: {'Disabled' if disable_column_scaling else 'Enabled'}")
                print(f"IAM1 dynamic column scaling: {'Disabled' if disable_dynamic_column_scaling else 'Enabled'}")
                output_file = process_batch(
                    selected_batch, 
                    problem_dir, 
                    batch_methods, 
                    options, 
                    output_dir, 
                    disable_row_scaling, 
                    disable_dynamic_row_scaling,
                    disable_column_scaling,
                    disable_dynamic_column_scaling
                )
                print(f"\nBatch processing complete. Results saved to {output_file}")
            
            input("\nPress Enter to return to menu...")
        
        elif choice == '8':
            # Toggle IAM1 row scaling
            disable_row_scaling = not disable_row_scaling
            print(f"\nIAM1 row scaling is now {'DISABLED' if disable_row_scaling else 'ENABLED'}")
            input("\nPress Enter to return to menu...")
            
        elif choice == '9':
            # Toggle IAM1 dynamic row scaling
            disable_dynamic_row_scaling = not disable_dynamic_row_scaling
            print(f"\nIAM1 dynamic row scaling is now {'DISABLED' if disable_dynamic_row_scaling else 'ENABLED'}")
            input("\nPress Enter to return to menu...")
            
        elif choice == '10':
            # Toggle IAM1 column scaling
            disable_column_scaling = not disable_column_scaling
            print(f"\nIAM1 column scaling is now {'DISABLED' if disable_column_scaling else 'ENABLED'}")
            input("\nPress Enter to return to menu...")
            
        elif choice == '11':
            # Toggle IAM1 dynamic column scaling
            disable_dynamic_column_scaling = not disable_dynamic_column_scaling
            print(f"\nIAM1 dynamic column scaling is now {'DISABLED' if disable_dynamic_column_scaling else 'ENABLED'}")
            input("\nPress Enter to return to menu...")
            
        elif choice == '12':
            # Run a single example from examples.py (with detailed output)
            try:
                # Select example and methods
                selected_example, selected_methods = select_example_and_methods()
                
                if selected_example and selected_methods:
                    # Run the selected example with detailed output
                    run_example_with_details(
                        selected_example, 
                        selected_methods,
                        disable_row_scaling,
                        disable_dynamic_row_scaling,
                        disable_column_scaling,
                        disable_dynamic_column_scaling
                    )
                    
            except ImportError:
                print("Error: examples.py not found or could not be imported.")
                print("Please make sure examples.py exists in the current directory.")
            except Exception as e:
                print(f"Error running example: {str(e)}")
                
            input("\nPress Enter to return to menu...")
        
        elif choice == '13':
            # Run all examples from examples.py in sequence
            try:
                run_all_examples(
                    methods_and_limits=None, 
                    disable_row_scaling=disable_row_scaling, 
                    disable_dynamic_row_scaling=disable_dynamic_row_scaling,
                    disable_column_scaling=disable_column_scaling, 
                    disable_dynamic_column_scaling=disable_dynamic_column_scaling
                )
            except Exception as e:
                print(f"Error running all examples: {str(e)}")
            
            input("\nPress Enter to return to menu...")
        
        elif choice == '14':
            # Exit
            print("\nExiting the program. Goodbye!")
            sys.exit(0)
            
        else:
            print("\nInvalid choice.")
            input("\nPress Enter to try again...")

def run_example_with_details(example_data, methods_and_limits, 
                     disable_row_scaling=False, disable_dynamic_row_scaling=False,
                     disable_column_scaling=True, disable_dynamic_column_scaling=True):
    """
    Run a specific example and show detailed step-by-step dictionaries.
    
    Parameters:
    -----------
    example_data : dict
        Example data containing 'A', 'b', 'name', etc.
    methods_and_limits : dict
        Dictionary mapping method names to iteration limits
    disable_row_scaling : bool
        Whether initial row scaling is disabled
    disable_dynamic_row_scaling : bool
        Whether dynamic row scaling is disabled
    disable_column_scaling : bool
        Whether initial column scaling is disabled
    disable_dynamic_column_scaling : bool
        Whether dynamic column scaling is disabled
        
    Returns:
    --------
    dict
        Results of solving the example with different methods
    """
    A = example_data['A']
    b = example_data['b']
    example_name = example_data['name']
    expected_feasible = example_data.get('feasible', 'Unknown')
    
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {example_name}".center(60))
    print(f"{'='*60}")
    print(f"Size: {A.shape[0]} constraints x {A.shape[1]} variables")
    print(f"Expected: {'Feasible' if expected_feasible else 'Infeasible'}")
    print(f"{'='*60}\n")
    
    # Print problem details
    print("Problem Details:")
    print("-" * 60)
    print("A matrix:")
    np.set_printoptions(precision=2, suppress=True, linewidth=100)
    print(A)
    print("\nb vector:")
    print(b)
    print("-" * 60)
    
    # Prepare results dictionary
    results = {}
    
    # Import needed functions for showing dictionary
    from iam1 import IAM1
    
    for method_name, max_iter in methods_and_limits.items():
        print(f"\n{'-'*60}")
        print(f"Solving with {method_name} method (max iterations: {max_iter})")
        print(f"{'-'*60}")
        
        if method_name == 'iam1':
            # Show step-by-step dictionary for IAM1
            print("\nIAM1 Step-by-Step Details:")
            
            # Apply scaling settings
            original_row_scaling = config_iam.ENABLE_ROW_SCALING
            original_dynamic_row_scaling = config_iam.ENABLE_DYNAMIC_ROW_SCALING
            
            # Set scaling parameters
            config_iam.ENABLE_ROW_SCALING = not disable_row_scaling
            config_iam.ENABLE_DYNAMIC_ROW_SCALING = not disable_dynamic_row_scaling
            
            # Set column scaling if available
            if hasattr(config_iam, 'ENABLE_COLUMN_SCALING'):
                original_column_scaling = config_iam.ENABLE_COLUMN_SCALING
                config_iam.ENABLE_COLUMN_SCALING = not disable_column_scaling
            else:
                original_column_scaling = None
            
            if hasattr(config_iam, 'ENABLE_DYNAMIC_COLUMN_SCALING'):
                original_dynamic_column_scaling = config_iam.ENABLE_DYNAMIC_COLUMN_SCALING
                config_iam.ENABLE_DYNAMIC_COLUMN_SCALING = not disable_dynamic_column_scaling
            else:
                original_dynamic_column_scaling = None
            
            try:
                # Create a custom IAM1 wrapper class to show dictionaries with projections
                class CustomIAM1(IAM1):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.iteration_count = 0
                    
                    def print_dictionary_with_projections(self):
                        """Print dictionary with 2 decimal places and projection values"""
                        # Create labels for variables
                        basic_labels = [f"x{i+1}" if i < self.n else f"s{i-self.n+1}" for i in self.basic_vars]
                        nonbasic_labels = [f"x{i+1}" if i < self.n else f"s{i-self.n+1}" for i in self.nonbasic_vars]
                        
                        # Print header
                        print(f"{'Basic':<8} | {'RHS':<8}", end="")
                        for label in nonbasic_labels:
                            print(f" | {label:<8}", end="")
                        print()
                        print("-" * (8 + 10 * (len(nonbasic_labels) + 1)))
                        
                        # Print each row - using 2 decimal places
                        for i in range(1, self.m + 1):
                            print(f"{basic_labels[i-1]:<8} | {self.dict[i, 0]:<8.2f}", end="")
                            for j in range(1, self.n + 1):
                                print(f" | {self.dict[i, j]:<8.2f}", end="")
                            print()
                        
                        # Print projection values in bottom row if not initial dictionary
                        if self.iteration_count > 0:
                            print("-" * (8 + 10 * (len(nonbasic_labels) + 1)))
                            print(f"{'Proj':<8} | {'---':<8}", end="")
                            
                            # Calculate projection for each column
                            for j in range(1, self.n + 1):
                                projection = self.compute_projection(j)
                                print(f" | {projection:<8.2f}", end="")
                            print()
                    
                    def solve(self):
                        """Override solve method to show dictionary at each iteration"""
                        if self.verbose:
                            print("Solving LP feasibility problem using IAM-1...")
                        
                        # Initialize performance tracking
                        start_time = time.time()
                        self.pivot_count = 0
                        self.pivot_history = []
                        
                        # Print initial dictionary
                        print("\nInitial Dictionary (Iteration 0):")
                        self.print_dictionary_with_projections()
                        
                        # Check initial feasibility
                        if self.is_primal_feasible():
                            if self.verbose:
                                print("Initial dictionary is feasible.")
                            return True, 0, time.time() - start_time, self.dict
                        
                        # Main iteration loop
                        for iteration in range(self.max_iter):
                            self.iterations = iteration + 1  # 1-indexed iterations
                            self.iteration_count = iteration + 1
                            
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
                            entering_col, max_proj = self.find_max_projection(r)
                            
                            if entering_col == -1:
                                # No suitable column found, problem is infeasible
                                if self.verbose:
                                    print(f"No suitable column found. Problem is infeasible.")
                                return False, self.iterations, time.time() - start_time, self.dict
                            
                            # Print dictionary before pivoting
                            print(f"\nDictionary before Iteration {self.iterations}:")
                            self.print_dictionary_with_projections()
                            print(f"Pivoting on element ({r}, {entering_col})")
                            
                            # STEP 7: Perform pivot
                            self.pivot(r, entering_col)
                            self.pivot_count += 1  # Increment pivot count
                            
                            # Check for cycling - just detect, don't prevent
                            if self.iterations % config_iam.CYCLING_CHECK_FREQUENCY == 0:
                                self.check_for_cycling()
                        
                        # If we reach here, we've hit the maximum iteration limit
                        if self.verbose:
                            print(f"Reached maximum iterations {self.max_iter} without finding a feasible solution.")
                        
                        return False, self.max_iter, time.time() - start_time, self.dict
                
                # Create our custom IAM1 solver with verbose=True
                solver = CustomIAM1(A, b, max_iter=max_iter, verbose=True)
                
                # Show solver steps
                print("\nSolving step-by-step:")
                is_feasible, iterations, time_taken, final_dict = solver.solve()
                
                # Display final dictionary
                print("\nFinal Dictionary:")
                solver.print_dictionary_with_projections()
                
                # Print summary
                print(f"\nResult: {'Feasible' if is_feasible else 'Infeasible'}")
                print(f"Iterations: {iterations}")
                print(f"Time taken: {time_taken:.6f} seconds")
                
                if is_feasible:
                    # Get solution if feasible
                    solution = solver.get_solution()
                    print("\nSolution vector:")
                    print(solution)
                
                # Save results
                results[method_name] = {
                    'is_feasible': is_feasible,
                    'iterations': iterations,
                    'time': time_taken
                }
            except Exception as e:
                print(f"Error running IAM1 solver: {str(e)}")
                results[method_name] = {
                    'is_feasible': False,
                    'iterations': -1,
                    'time': 0,
                    'error': str(e)
                }
            finally:
                # Restore original scaling settings
                config_iam.ENABLE_ROW_SCALING = original_row_scaling
                config_iam.ENABLE_DYNAMIC_ROW_SCALING = original_dynamic_row_scaling
                
                if original_column_scaling is not None:
                    config_iam.ENABLE_COLUMN_SCALING = original_column_scaling
                    
                if original_dynamic_column_scaling is not None:
                    config_iam.ENABLE_DYNAMIC_COLUMN_SCALING = original_dynamic_column_scaling
        else:
            # Run other methods using solve_with_method
            result = solve_with_method(
                method_name, 
                A, 
                b, 
                max_iter=max_iter, 
                tol=ZERO_TOLERANCE,
                disable_row_scaling=disable_row_scaling,
                disable_dynamic_row_scaling=disable_dynamic_row_scaling,
                disable_column_scaling=disable_column_scaling,
                disable_dynamic_column_scaling=disable_dynamic_column_scaling
            )
            
            # Show results
            print(f"\nResult: {'Feasible' if result['is_feasible'] else 'Infeasible'}")
            print(f"Iterations: {result['iterations']}")
            print(f"Time taken: {result['time']:.6f} seconds")
            
            # Save results
            results[method_name] = {
                'is_feasible': result['is_feasible'],
                'iterations': result['iterations'],
                'time': result['time']
            }
    
    # Print comparative summary
    print(f"\n{'='*60}")
    print("COMPARATIVE SUMMARY".center(60))
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Feasible':<10} {'Iterations':<12} {'Time (s)':<10}")
    print(f"{'-'*60}")
    
    for method, result in results.items():
        feasible_str = "Yes" if result['is_feasible'] else "No"
        print(f"{method:<15} {feasible_str:<10} {result['iterations']:<12} {result['time']:<10.6f}")
    
    print(f"{'='*60}")
    
    return results

def select_example_and_methods():
    """
    Let the user select an example from examples.py and choose which methods to run.
    
    Returns:
    --------
    tuple
        (selected_example, selected_methods) or (None, None) if cancelled
    """
    # Import example list and functions
    from examples import list_examples, get_example, EXAMPLES
    
    # Display available examples
    clear_screen()
    list_examples()
    
    # Select an example
    print("\nSelect an example to run:")
    try:
        example_idx = int(input("Enter example number (or 0 to cancel): "))
        if example_idx == 0:
            return None, None
            
        selected_example = get_example(example_idx)
    except (ValueError, IndexError) as e:
        print(f"Invalid selection: {str(e)}")
        input("\nPress Enter to return to menu...")
        return None, None
    
    # Ask about method selection approach
    print("\nMethod selection:")
    print("1. Use all methods with default iteration limits")
    print("2. Select specific methods and iteration limits")
    method_choice = input("Enter choice (1-2, default=1): ").strip()
    
    if method_choice == '2':
        # Select methods to run manually
        available_methods = ["simplex", "highs", "iam1", "pan_row_scaled"]
        print("\nSelect methods to run:")
        selected_methods = select_methods_and_limits(available_methods)
        
        if not selected_methods:
            return None, None
    else:
        # Use all methods with default iteration limits
        available_methods = ["simplex", "highs", "iam1", "pan_row_scaled"]
        selected_methods = {}
        for method in available_methods:
            if method == "highs":
                selected_methods[method] = None  # No iteration limit for HiGHS
            else:
                selected_methods[method] = MAX_ITERATIONS
        
        print(f"\nUsing all methods with default iteration limits:")
        for method, limit in selected_methods.items():
            print(f"- {method}: {limit if limit is not None else 'No limit'}")
        
    return selected_example, selected_methods

def run_all_examples(methods_and_limits=None, 
                  disable_row_scaling=False, disable_dynamic_row_scaling=False,
                  disable_column_scaling=True, disable_dynamic_column_scaling=True):
    """
    Run all examples from examples.py in sequence.
    
    Parameters:
    -----------
    methods_and_limits : dict, optional
        Dictionary mapping method names to iteration limits.
        If None, user will be prompted to select methods.
    disable_row_scaling : bool
        Whether initial row scaling is disabled
    disable_dynamic_row_scaling : bool
        Whether dynamic row scaling is disabled
    disable_column_scaling : bool
        Whether initial column scaling is disabled
    disable_dynamic_column_scaling : bool
        Whether dynamic column scaling is disabled
    """
    try:
        # Import examples directly
        from examples import EXAMPLES
    except ImportError:
        print("Error: examples.py not found or could not be imported.")
        print("Please make sure examples.py exists in the current directory.")
        return
    
    # If methods not provided, prompt for method selection
    if methods_and_limits is None:
        # Ask about method selection approach
        print("\nMethod selection for all examples:")
        print("1. Use all methods with default iteration limits")
        print("2. Select specific methods and iteration limits")
        method_choice = input("Enter choice (1-2, default=1): ").strip()
        
        if method_choice == '2':
            # Select methods to run manually
            available_methods = ["simplex", "highs", "iam1", "pan_row_scaled"]
            print("\nSelect methods to run:")
            methods_and_limits = select_methods_and_limits(available_methods)
            
            if not methods_and_limits:
                print("No methods selected, cancelling examples.")
                return
        else:
            # Use all methods with default iteration limits
            available_methods = ["simplex", "highs", "iam1", "pan_row_scaled"]
            methods_and_limits = {}
            for method in available_methods:
                if method == "highs":
                    methods_and_limits[method] = None  # No iteration limit for HiGHS
                else:
                    methods_and_limits[method] = MAX_ITERATIONS
            
            print(f"\nUsing all methods with default iteration limits:")
            for method, limit in methods_and_limits.items():
                print(f"- {method}: {limit if limit is not None else 'No limit'}")
        
    # Display header
    print(f"\n{'='*60}")
    print("RUNNING ALL EXAMPLES FROM EXAMPLES.PY".center(60))
    print(f"{'='*60}")
    print(f"Total examples: {len(EXAMPLES)}")
    print(f"Methods: {', '.join(methods_and_limits.keys())}")
    
    # Run each example
    for i, example in enumerate(EXAMPLES, 1):
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i}/{len(EXAMPLES)}: {example['name']}".center(60))
        print(f"{'='*60}")
        
        # Get user confirmation to proceed
        proceed = input(f"Run example {i} - {example['name']}? (Y/n, default=Y): ")
        if proceed.lower() == 'n':
            print(f"Skipping example {i}")
            continue
            
        # Run the example
        try:
            run_example_with_details(
                example,
                methods_and_limits,
                disable_row_scaling,
                disable_dynamic_row_scaling,
                disable_column_scaling,
                disable_dynamic_column_scaling
            )
        except Exception as e:
            print(f"Error running example {i}: {str(e)}")
        
        # Get user confirmation to continue after each example
        if i < len(EXAMPLES):
            next_example = input("\nPress Enter to continue to next example or 'q' to quit: ")
            if next_example.lower() == 'q':
                print("Exiting examples")
                break
    
    print(f"\n{'='*60}")
    print("COMPLETED RUNNING EXAMPLES".center(60))
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print("Press Enter to exit...")
        input()
        sys.exit(1)
