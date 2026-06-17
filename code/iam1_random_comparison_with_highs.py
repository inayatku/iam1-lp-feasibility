import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import warnings
import random
import os

# Import centralized Random problem configuration parameters
from config_random import (
    ZERO_TOLERANCE, 
    FEASIBILITY_TOLERANCE,
    MAX_ITERATIONS, 
    ENABLE_ROW_SCALING,
    DEFAULT_SPARSITY,
    DEFAULT_BOUND,
    MAX_GENERATION_ATTEMPTS,
    PROBLEMS_PER_SIZE,
    TIMEOUT_SECONDS,
    DEFAULT_PROBLEM_SIZES,
    PLOT_DPI,
    PLOT_SIZE_INCHES,
    SHOW_PLOTS,
    IAM1_MAX_ITERATIONS,
    SIMPLEX_MAX_ITERATIONS,
    PAN_MAX_ITERATIONS,
    get_random_config_info
)

# Create output directory if it doesn't exist
OUTPUT_DIR = "random_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add this line to suppress the simplex deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

# Import solver methods from separate files
from iam1 import solve_with_iam1 as iam1_standard
# Commented out IAM variants that are no longer used
# from iam1_l1_approximation import solve_with_iam1_l1_approximation
# Disabled: IAM-1 with Pure L1
# from iam1_pure_l1 import solve_with_iam1_pure_l1
# from iam_1b_l2_scaling import solve_with_iam1b_l2_scaling
# from iam_1b_l1_scaling import solve_with_iam1b_l1_scaling
from simplex_phase_1 import solve_with_simplex_phase1
# Temporarily disabled due to indentation errors in the file
# from avf_simplex_phase1 import solve_with_avf_simplex_phase1
# Add Pan's method with row scaling
from Pans_method_row_scale import check_feasibility_pan_row_scaled

# We will use only SciPy's HiGHS implementation
HIGHS_WRAPPER_AVAILABLE = False

def generate_random_lp(m, n, sparsity=0.0, bound=50):
    """Generate a random linear programming problem."""
    # Generate random matrix A with specified sparsity
    A = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if np.random.random() >= sparsity:  # Only create non-zero with probability (1-sparsity)
                A[i, j] = np.random.randint(-bound, bound+1)
    
    # Generate random right-hand side b
    b = np.random.randint(-bound, bound+1, size=m)
    
    return A, b


def generate_feasible_lp(m, n, sparsity=DEFAULT_SPARSITY, bound=DEFAULT_BOUND, max_attempts=MAX_GENERATION_ATTEMPTS):
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


def generate_infeasible_lp(m, n, sparsity=DEFAULT_SPARSITY, bound=DEFAULT_BOUND, max_attempts=MAX_GENERATION_ATTEMPTS):
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


def run_comparison(problem_sizes, num_problems=PROBLEMS_PER_SIZE, problem_type='feasible', sparsity=DEFAULT_SPARSITY, timeout=TIMEOUT_SECONDS, max_iter=MAX_ITERATIONS, tol=ZERO_TOLERANCE):
    """
    Run comparison tests for all IAM-1 variants on random problems.
    
    Parameters:
    -----------
    problem_sizes : list of tuples
        List of (m, n) tuples representing problem sizes to test
    num_problems : int
        Number of problems to test for each size (default from config)
    problem_type : str
        Either 'feasible' or 'infeasible'
    sparsity : float
        Sparsity level (0-1) for constraint matrices (default from config)
    timeout : int
        Maximum seconds allowed for each method on a problem (default from config)
    max_iter : int
        Maximum number of iterations for each solver (default from config)
    tol : float
        Numerical tolerance for feasibility checking (default from config)
        
    Returns:
    --------
    pandas.DataFrame
        Results of the comparison tests
    """
    # Methods to compare (excluding Simplex Phase 1 and HiGHS as they're treated separately)
    methods = [
        # Removed IAM variants, keeping only IAM-1 Standard
        ('IAM-1 Standard', iam1_standard),
        # Temporarily disabled due to indentation errors in the file
        # ('AVF Simplex Phase 1', solve_with_avf_simplex_phase1),
        # Add Pan's method with row scaling
        ('Pan\'s Method with Row Scaling', check_feasibility_pan_row_scaled)
    ]
    
    # Results dictionary
    results = {
        'Size': [],
        'Problem_Type': []
    }
    
    # Initialize columns for each method
    for method_name, _ in methods:
        results[f'{method_name}_Success_Rate'] = []
        results[f'{method_name}_Agreement_with_Simplex'] = []  # Agreement with Simplex
        results[f'{method_name}_Agreement_with_HiGHS'] = []   # Agreement with HiGHS
        results[f'{method_name}_Max_Iter_Reached'] = []
        results[f'{method_name}_Avg_Iterations'] = []
        results[f'{method_name}_Avg_Iterations_All'] = []  # Also track average with all iterations
        results[f'{method_name}_Avg_Time'] = []
        results[f'{method_name}_Time_Per_Iter'] = []
    
    # Add Simplex method columns
    results['Simplex_Avg_Iterations'] = []
    results['Simplex_Avg_Time'] = []
    results['Simplex_Time_Per_Iter'] = []
    
    # Add HiGHS method columns
    results['HiGHS_Avg_Iterations'] = []
    results['HiGHS_Avg_Time'] = []
    results['HiGHS_Time_Per_Iter'] = []
    
    # Test each problem size
    for m, n in problem_sizes:
        print(f"\n{Colors.BOLD}Testing {m}x{n} {problem_type} problems with sparsity {sparsity}{Colors.END}")
        
        # Initialize statistics for this size
        method_stats = {method_name: {
            'success_count': 0,
            'agreement_with_simplex': 0,
            'agreement_with_highs': 0,  # New counter for agreement with HiGHS
            'iterations': [],
            'times': []
        } for method_name, _ in methods}
        
        # Add Simplex method stats
        simplex_stats = {
            'iterations': [],
            'times': []
        }
        
        # Add HiGHS method stats
        highs_stats = {
            'iterations': [],
            'times': []
        }
        
        # Generate and test problems
        for i in range(num_problems):
            print(f"  Problem {i+1}/{num_problems}...", end='', flush=True)
            
            # Generate problem based on type
            if problem_type == 'feasible':
                problem = generate_feasible_lp(m, n, sparsity)
            else:
                problem = generate_infeasible_lp(m, n, sparsity)
            
            if problem is None:
                print(f" {Colors.YELLOW}Skipped (couldn't generate){Colors.END}")
                continue
                
            A, b = problem
            
            # Solve with Simplex first to establish one benchmark
            simplex_start_time = time.time()
            try:
                simplex_result = solve_with_simplex_phase1(A, b, max_iter=max_iter, tol=tol)
                simplex_time = simplex_result['time']
                simplex_status = 'feasible' if simplex_result['status'] == 'feasible' else 'infeasible'
                simplex_iterations = simplex_result['iterations']
                
                simplex_stats['iterations'].append(simplex_iterations)
                simplex_stats['times'].append(simplex_time)
            except Exception as e:
                print(f" {Colors.RED}Simplex error: {str(e)}{Colors.END}", end='')
                continue
            
            # Solve with HiGHS as second benchmark
            highs_start_time = time.time()
            try:
                # Use SciPy's linprog with HiGHS method
                highs_result = linprog(
                    np.zeros(A.shape[1]), 
                    A_ub=A, 
                    b_ub=b, 
                    method='highs', 
                    options={'presolve': False}  # No maxiter limit for HiGHS
                )
                highs_time = time.time() - highs_start_time
                highs_status = 'feasible' if highs_result.success else 'infeasible'
                highs_iterations = highs_result.nit if hasattr(highs_result, 'nit') else 0
                
                highs_stats['iterations'].append(highs_iterations)
                highs_stats['times'].append(highs_time)
            except Exception as e:
                print(f" {Colors.RED}HiGHS error: {str(e)}{Colors.END}", end='')
                continue
            
            # Test each method
            for method_name, solve_func in methods:
                try:
                    # Get the result from solver with specified max_iter and tolerance
                    result = solve_func(A, b, max_iter=max_iter, tol=tol)
                    
                    # Use internal time reported by solver instead of wall clock time
                    solve_time = result['time']
                    
                    if solve_time > timeout:
                        print(f" {Colors.YELLOW}{method_name} timeout{Colors.END}", end='')
                        continue
                        
                    # Record statistics
                    method_stats[method_name]['iterations'].append(result['iterations'])
                    method_stats[method_name]['times'].append(solve_time)
                    
                    # Track if max iterations was reached
                    if result['iterations'] >= max_iter:
                        method_stats[method_name].setdefault('max_iter_reached', 0)
                        method_stats[method_name]['max_iter_reached'] += 1
                    
                    if result['status'] in ['feasible', 'infeasible']:
                        # Check if the result matches the expected result for the problem type
                        if (problem_type == 'feasible' and result['status'] == 'feasible') or \
                           (problem_type == 'infeasible' and result['status'] == 'infeasible'):
                            method_stats[method_name]['success_count'] += 1
                        
                        # Check if the result agrees with simplex
                        if result['status'] == simplex_status:
                            method_stats[method_name]['agreement_with_simplex'] += 1
                            
                        # Check if the result agrees with HiGHS
                        if result['status'] == highs_status:
                            method_stats[method_name]['agreement_with_highs'] += 1
                    
                except Exception as e:
                    print(f" {Colors.RED}{method_name} error: {str(e)}{Colors.END}", end='')
            
            print(f" {Colors.GREEN}Done{Colors.END}")
        
        # Calculate averages and success rates for this size
        results['Size'].append(f"{m}x{n}")
        results['Problem_Type'].append(problem_type)
        
        # Process simplex results
        simplex_avg_iterations = np.mean(simplex_stats['iterations']) if simplex_stats['iterations'] else float('nan')
        simplex_avg_time = np.mean(simplex_stats['times']) if simplex_stats['times'] else float('nan')
        
        # Calculate simplex time per iteration
        simplex_time_per_iters = []
        for time_val, iter_val in zip(simplex_stats['times'], simplex_stats['iterations']):
            if iter_val > 0:  # Avoid division by zero
                simplex_time_per_iters.append(time_val / iter_val)
        simplex_time_per_iter = np.mean(simplex_time_per_iters) if simplex_time_per_iters else float('nan')
        
        results['Simplex_Avg_Iterations'].append(round(simplex_avg_iterations, 2))
        results['Simplex_Avg_Time'].append(round(simplex_avg_time, 4))
        results['Simplex_Time_Per_Iter'].append(round(simplex_time_per_iter, 6))
        
        # Process HiGHS results
        highs_avg_iterations = np.mean(highs_stats['iterations']) if highs_stats['iterations'] else float('nan')
        highs_avg_time = np.mean(highs_stats['times']) if highs_stats['times'] else float('nan')
        
        # Calculate HiGHS time per iteration
        highs_time_per_iters = []
        for time_val, iter_val in zip(highs_stats['times'], highs_stats['iterations']):
            if iter_val > 0:  # Avoid division by zero
                highs_time_per_iters.append(time_val / iter_val)
        highs_time_per_iter = np.mean(highs_time_per_iters) if highs_time_per_iters else float('nan')
        
        results['HiGHS_Avg_Iterations'].append(round(highs_avg_iterations, 2))
        results['HiGHS_Avg_Time'].append(round(highs_avg_time, 4))
        results['HiGHS_Time_Per_Iter'].append(round(highs_time_per_iter, 6))
        
        # Print simplex results
        print(f"  Simplex (Benchmark):")
        print(f"    Avg Iterations: {simplex_avg_iterations:.2f}")
        print(f"    Avg Time: {simplex_avg_time:.4f}s")
        print(f"    Time Per Iteration: {simplex_time_per_iter:.6f}s")
        
        # Print HiGHS results
        print(f"  HiGHS (Benchmark):")
        print(f"    Avg Iterations: {highs_avg_iterations:.2f}")
        print(f"    Avg Time: {highs_avg_time:.4f}s")
        print(f"    Time Per Iteration: {highs_time_per_iter:.6f}s")
        
        for method_name, _ in methods:
            stats = method_stats[method_name]
            
            # Success rate
            success_rate = (stats['success_count'] / num_problems) * 100 if num_problems > 0 else 0
            results[f'{method_name}_Success_Rate'].append(round(success_rate, 1))
            
            # Agreement with simplex rate
            agreement_with_simplex_rate = (stats['agreement_with_simplex'] / num_problems) * 100 if num_problems > 0 else 0
            results[f'{method_name}_Agreement_with_Simplex'].append(round(agreement_with_simplex_rate, 1))
            
            # Agreement with HiGHS rate
            agreement_with_highs_rate = (stats['agreement_with_highs'] / num_problems) * 100 if num_problems > 0 else 0
            results[f'{method_name}_Agreement_with_HiGHS'].append(round(agreement_with_highs_rate, 1))
            
            # Max iterations reached rate
            max_iter_reached = stats.get('max_iter_reached', 0)
            max_iter_rate = (max_iter_reached / num_problems) * 100 if num_problems > 0 else 0
            results[f'{method_name}_Max_Iter_Reached'].append(round(max_iter_rate, 1))
            
            # Average iterations - modified to exclude max iteration cases
            non_max_iter_iterations = []
            for iter_val in stats['iterations']:
                if iter_val < max_iter:  # Only include if less than max_iter
                    non_max_iter_iterations.append(iter_val)
                
            # Calculate average with non-max iteration cases
            if non_max_iter_iterations:
                avg_iterations = np.mean(non_max_iter_iterations)
            else:
                avg_iterations = float('nan')  # If all cases hit max iter
                
            results[f'{method_name}_Avg_Iterations'].append(round(avg_iterations, 2))
            
            # Also record average including all iterations 
            avg_iterations_all = np.mean(stats['iterations']) if stats['iterations'] else float('nan')
            results[f'{method_name}_Avg_Iterations_All'].append(round(avg_iterations_all, 2))
            
            # Average time
            avg_time = np.mean(stats['times']) if stats['times'] else float('nan')
            results[f'{method_name}_Avg_Time'].append(round(avg_time, 4))
            
            # Calculate time per iteration for each individual problem
            problem_time_per_iters = []
            for time_val, iter_val in zip(stats['times'], stats['iterations']):
                if iter_val > 0:  # Avoid division by zero
                    problem_time_per_iters.append(time_val / iter_val)
            
            # Calculate the average time per iteration across problems
            if problem_time_per_iters:
                time_per_iter = np.mean(problem_time_per_iters)
            else:
                time_per_iter = float('nan')
            results[f'{method_name}_Time_Per_Iter'].append(round(time_per_iter, 6))
            
            # Print method summary with agreement indicators
            simplex_agreement_indicator = ""
            if agreement_with_simplex_rate < 100:
                simplex_agreement_indicator = f" {Colors.RED}[DISAGREES with Simplex: {100-agreement_with_simplex_rate:.1f}% of problems]{Colors.END}"
                
            highs_agreement_indicator = ""
            if agreement_with_highs_rate < 100:
                highs_agreement_indicator = f" {Colors.RED}[DISAGREES with HiGHS: {100-agreement_with_highs_rate:.1f}% of problems]{Colors.END}"
            
            print(f"  {method_name}:{simplex_agreement_indicator}{highs_agreement_indicator}")
            
            # Only show success rate if not 100%
            if success_rate < 100:
                print(f"    Success Rate: {success_rate:.1f}%")
                
            # Only show agreement with simplex if not 100%
            if agreement_with_simplex_rate < 100:
                print(f"    Agreement with Simplex: {agreement_with_simplex_rate:.1f}%")
                
            # Only show agreement with HiGHS if not 100%
            if agreement_with_highs_rate < 100:
                print(f"    Agreement with HiGHS: {agreement_with_highs_rate:.1f}%")
                
            # Only show max iterations reached if greater than 0%
            if max_iter_rate > 0:
                print(f"    Max Iterations Reached: {max_iter_rate:.1f}%")
                
            print(f"    Avg Iterations (excluding max iter cases): {avg_iterations:.2f}")
            print(f"    Avg Iterations (all cases): {avg_iterations_all:.2f}")
            print(f"    Avg Time: {avg_time:.4f}s")
            print(f"    Time Per Iteration: {time_per_iter:.6f}s")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def plot_comparison(results_df, metric='Success_Rate', problem_type='feasible', save_file=None):
    """
    Create visualizations of the comparison results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results of the comparison tests
    metric : str
        Which metric to visualize: 'Success_Rate', 'Avg_Iterations', or 'Avg_Time'
    problem_type : str
        Either 'feasible' or 'infeasible'
    save_file : str, optional
        File path to save the plot (if None, will use default naming)
    """
    # Filter results for the specific problem type
    df = results_df[results_df['Problem_Type'] == problem_type].copy()
    
    # Extract sizes and convert to string for better plotting
    df['Size_Str'] = df['Size'].apply(lambda x: f"{x[0]}x{x[1]}")
    
    # Get methods (excluding Simplex and HiGHS)
    methods = []
    for col in df.columns:
        if col.endswith(f"_{metric}"):
            method = col.replace(f"_{metric}", "")
            if method not in ['Simplex', 'HiGHS']:
                methods.append(method)
    
    # Create a figure with configurable size from config
    plt.figure(figsize=PLOT_SIZE_INCHES)
    
    # Plot each method
    for method in methods:
        metric_col = f"{method}_{metric}"
        plt.plot(df['Size_Str'], df[metric_col], 'o-', linewidth=2, markersize=8, label=method)
    
    # Add Simplex Phase 1 as a reference
    if metric == 'Success_Rate':
        # For success rate, Simplex is always 100% by construction
        plt.plot(df['Size_Str'], [1.0] * len(df), 'k--', linewidth=2, label='Simplex (Reference)')
    else:
        metric_col = f"Simplex_{metric.replace('Success_Rate', 'Avg_Iterations' if metric == 'Success_Rate' else metric)}"
        if metric_col in df.columns:
            plt.plot(df['Size_Str'], df[metric_col], 'k--', linewidth=2, label='Simplex Phase 1')
    
    # Add HiGHS as a reference
    if metric == 'Success_Rate':
        # For success rate, HiGHS is always 100% by construction
        plt.plot(df['Size_Str'], [1.0] * len(df), 'r--', linewidth=2, label='HiGHS (Reference)')
    else:
        metric_col = f"HiGHS_{metric.replace('Success_Rate', 'Avg_Iterations' if metric == 'Success_Rate' else metric)}"
        if metric_col in df.columns:
            plt.plot(df['Size_Str'], df[metric_col], 'r--', linewidth=2, label='HiGHS')
    
    # Set up the plot
    metric_labels = {
        'Success_Rate': 'Success Rate (%)',
        'Avg_Iterations': 'Average Iterations',
        'Avg_Time': 'Average Time (s)',
        'Time_Per_Iter': 'Time per Iteration (ms)'
    }
    
    plt.title(f"{metric_labels.get(metric, metric)} for {problem_type.capitalize()} Problems")
    plt.xlabel('Problem Size (m×n)')
    plt.ylabel(metric_labels.get(metric, metric))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set logarithmic scale for iterations and time
    if metric in ['Avg_Iterations', 'Avg_Time', 'Time_Per_Iter']:
        plt.yscale('log')
    
    # Set y-axis limit for success rate
    if metric == 'Success_Rate':
        plt.ylim(0, 1.05)
        # Convert to percentages for display
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0%', '20%', '40%', '60%', '80%', '100%'])
    
    plt.tight_layout()
    
    # Save the plot with high DPI from config
    if save_file is None:
        save_file = os.path.join(OUTPUT_DIR, f"{problem_type}_{metric.lower()}.png")
    
    plt.savefig(save_file, dpi=PLOT_DPI)
    print(f"Plot saved to {save_file}")
    
    # Show the plot if configured to do so
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_comparison_with_benchmarks(results_df, metric='Avg_Iterations', problem_type='feasible', save_file=None):
    """
    Plot comparison of IAM variants against Simplex and HiGHS benchmarks.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results dataframe from run_comparison
    metric : str
        Metric to plot: 'Avg_Iterations', 'Avg_Time', or 'Time_Per_Iter'
    problem_type : str
        'feasible' or 'infeasible'
    save_file : str
        Filename to save the plot, or None to display
    """
    # Filter by problem type
    df = results_df[results_df['Problem_Type'] == problem_type]
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Benchmark columns
    simplex_col = f'Simplex_{metric}'
    highs_col = f'HiGHS_{metric}'
    
    # Plot benchmark methods
    plt.plot(df['Size'], df[simplex_col], marker='s', linewidth=3, linestyle='--', 
             color='black', label='Simplex (Benchmark)')
    plt.plot(df['Size'], df[highs_col], marker='d', linewidth=3, linestyle='--', 
             color='darkred', label='HiGHS (Benchmark)')
    
    # Get IAM variant columns for the specified metric
    if metric == 'Avg_Iterations':
        # For iterations, get both regular and _All columns
        iam_cols = [col for col in df.columns if col.endswith(f'_{metric}') 
                   and not col.startswith('Simplex_') 
                   and not col.startswith('HiGHS_')
                   and not col.endswith('_All')]  # Exclude _All columns
        
        # Additionally plot the _All columns with dotted lines
        iam_all_cols = [col for col in df.columns if col.endswith(f'_{metric}_All') 
                       and not col.startswith('Simplex_') 
                       and not col.startswith('HiGHS_')]
    else:
        iam_cols = [col for col in df.columns if col.endswith(f'_{metric}') 
                   and not col.startswith('Simplex_') 
                   and not col.startswith('HiGHS_')]
        iam_all_cols = []
    
    # Plot each IAM method
    for col in iam_cols:
        # Extract method name by removing the metric suffix
        method = col.replace(f'_{metric}', '')
        plt.plot(df['Size'], df[col], marker='o', linewidth=2, label=f'{method} (excl. max iter)')
    
    # Plot _All versions if metric is Avg_Iterations
    for col in iam_all_cols:
        # Extract method name by removing the metric suffix
        method = col.replace(f'_{metric}_All', '')
        plt.plot(df['Size'], df[col], marker='x', linewidth=1, linestyle=':', 
                label=f'{method} (all cases)', alpha=0.7)
    
    # Customize the plot
    plt.title(f'{metric} Comparison Against Benchmarks for {problem_type.capitalize()} Problems', fontsize=16)
    plt.xlabel('Problem Size (m×n)', fontsize=14)
    
    if metric == 'Avg_Time':
        plt.ylabel('Average Time (seconds)', fontsize=14)
    elif metric == 'Avg_Iterations':
        plt.ylabel('Average Iterations', fontsize=14)
    elif metric == 'Time_Per_Iter':
        plt.ylabel('Time per Iteration (seconds)', fontsize=14)
        
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file)
    else:
        plt.show()


def main():
    """Main function to run the comparison tests."""
    # Print global configuration info
    print(f"{Colors.BOLD}{Colors.GREEN}LP Feasibility Methods Comparison{Colors.END}")
    print("This tool compares different methods for solving LP feasibility problems.")
    
    print("\n" + "="*60)
    print("GLOBAL CONFIGURATION".center(60))
    print("="*60)
    print(get_random_config_info())
    print("="*60 + "\n")
    
    print("Methods compared:")
    print("1. IAM-1 Standard (with exact L2 norm)")
    print("2. Simplex Phase 1 (Two-phase simplex method implementation for LP feasibility problems - BENCHMARK)")
    print("3. HiGHS (high-performance solver using SciPy's linprog with zero objective, presolve disabled)")
    # print("4. AVF Simplex Phase 1 (Artificial Variable Free simplex method that directly constructs an auxiliary objective from infeasible constraints)")
    print("4. Pan's Method with Row Scaling (Implementation of Pan's Most-Obtuse-Angle Column Rule algorithm with initial row scaling)")
    print(f"\nAll output files will be saved to the '{OUTPUT_DIR}' folder")
    
    # Ask for random seed
    seed_input = input("\nEnter random seed for reproducibility (press Enter to use random seed): ").strip()
    if seed_input:
        try:
            seed = int(seed_input)
        except ValueError:
            # Convert string to integer seed using hash
            seed = hash(seed_input) 
            # Ensure seed is within valid range for numpy (0 to 2^32-1)
            seed = seed % (2**32 - 1)
    else:
        # Generate a random seed if user skipped
        seed = random.randint(1, 2**32 - 1)
    
    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    print(f"{Colors.GREEN}Using random seed: {seed}{Colors.END}")
    
    # Ask user for test parameters
    print("\nSelect problem type:")
    print("1. Feasible problems")
    print("2. Infeasible problems")
    problem_type_choice = input("Enter choice (1-2): ").strip()
    problem_type = 'feasible' if problem_type_choice == '1' else 'infeasible'
    
    print("\nSelect test size:")
    print("1. Quick test (small problems, few samples)")
    print("2. Medium test (mixed sizes, more samples)")
    print("3. Full test (all sizes, many samples) - can take a long time")
    print("4. Custom (specify your own problem sizes and sample count)")
    test_size_choice = input("Enter choice (1-4): ").strip()
    
    if test_size_choice == '1':
        problem_sizes = [(5, 5), (10, 10), (15, 15), (20, 20), (25, 25), (30, 30), (40, 40), (50, 50), (50, 60), (60, 60)]
        num_problems = 5
    elif test_size_choice == '2':
        problem_sizes = [(5, 5), (10, 10), (15, 15), (20, 20), (25, 25), (30, 30), (40, 40), (50,50), (50, 60), (60, 60)]
        num_problems = 30
    elif test_size_choice == '3':
        problem_sizes = [(5, 5), (10, 10), (15, 15), (20, 20), (25, 25), (30, 30), (40, 40), (50,50), (50, 60), (60, 60)]
        num_problems = 100
    else:  # Custom option
        try:
            custom_input = input("Enter problem sizes as comma-separated m,n pairs (e.g., '5,5 10,10 20,20'): ").strip()
            problem_sizes = []
            for pair in custom_input.split():
                m, n = map(int, pair.split(','))
                problem_sizes.append((m, n))
            
            if not problem_sizes:
                print(f"{Colors.YELLOW}No valid problem sizes, using default set{Colors.END}")
                problem_sizes = [(5, 5), (10, 10), (15, 15), (20, 20)]
                
            num_problems_input = input("Enter number of random problems to generate for each size: ").strip()
            num_problems = int(num_problems_input) if num_problems_input else 5
        except Exception as e:
            print(f"{Colors.RED}Error parsing custom settings: {e}. Using default values.{Colors.END}")
            problem_sizes = [(5, 5), (10, 10), (15, 15), (20, 20)]
            num_problems = 5
    
    print("\nSelect sparsity level:")
    print("1. Dense (0.0)")
    print("2. Medium sparse (0.2)")
    print("3. Very sparse (0.5)")
    print("4. Custom sparsity")
    sparsity_choice = input("Enter choice (1-4): ").strip()
    
    if sparsity_choice == '1':
        sparsity = 0.0
    elif sparsity_choice == '2':
        sparsity = 0.2
    elif sparsity_choice == '3':
        sparsity = 0.5
    else:
        try:
            sparsity_input = input("Enter custom sparsity (0.0 to 0.9): ").strip()
            sparsity = float(sparsity_input)
            # Constrain to reasonable range
            sparsity = min(max(sparsity, 0.0), 0.9)
        except:
            print(f"{Colors.YELLOW}Invalid sparsity. Using default value 0.0{Colors.END}")
            sparsity = 0.0
    
    # Option to enable/disable plot generation
    generate_plots = False  # Disabled by default
    plots_choice = input("\nGenerate plots? (y/N): ").strip().lower()
    if plots_choice == 'y' or plots_choice == 'yes':
        generate_plots = True
    
    # Run comparison tests
    print(f"\n{Colors.BOLD}Running comparison tests...{Colors.END}")
    print(f"- Using seed: {seed}")
    print(f"- Number of problems per size: {num_problems}")
    print(f"- Problem sizes: {', '.join([f'{m}x{n}' for m, n in problem_sizes])}")
    print(f"- Sparsity: {sparsity}")
    print(f"- Tolerance: {ZERO_TOLERANCE}")
    print(f"- Max iterations: {MAX_ITERATIONS}")
    print(f"- Generate plots: {'Yes' if generate_plots else 'No'}")
    print()
    
    results = run_comparison(
        problem_sizes=problem_sizes,
        num_problems=num_problems,
        problem_type=problem_type,
        sparsity=sparsity,
        max_iter=MAX_ITERATIONS,
        tol=ZERO_TOLERANCE
    )
    
    # Save results to CSV
    filename = os.path.join(OUTPUT_DIR, f"lp_feasibility_comparison_{problem_type}_sparsity{sparsity}_tol{ZERO_TOLERANCE}_maxiter{MAX_ITERATIONS}_seed{seed}_problems{num_problems}_results.csv")
    results.to_csv(filename, index=False)
    print(f"\n{Colors.GREEN}Results saved to {filename}{Colors.END}")
    
    # Add seed and tolerance to the DataFrame for reference
    results['Random_Seed'] = seed
    results['Tolerance'] = ZERO_TOLERANCE
    
    # Generate plots only if requested
    if generate_plots:
        print(f"\n{Colors.BOLD}Generating comparison plots...{Colors.END}")
        
        # Include tolerance in file names
        plot_comparison(results, 'Success_Rate', problem_type, 
                       os.path.join(OUTPUT_DIR, f"lp_success_rate_{problem_type}_tol{ZERO_TOLERANCE}_maxiter{MAX_ITERATIONS}_seed{seed}_problems{num_problems}.png"))
        plot_comparison(results, 'Agreement_with_Simplex', problem_type, 
                       os.path.join(OUTPUT_DIR, f"lp_agreement_simplex_{problem_type}_tol{ZERO_TOLERANCE}_maxiter{MAX_ITERATIONS}_seed{seed}_problems{num_problems}.png"))
        plot_comparison(results, 'Agreement_with_HiGHS', problem_type, 
                       os.path.join(OUTPUT_DIR, f"lp_agreement_highs_{problem_type}_tol{ZERO_TOLERANCE}_maxiter{MAX_ITERATIONS}_seed{seed}_problems{num_problems}.png"))
        plot_comparison(results, 'Max_Iter_Reached', problem_type, 
                       os.path.join(OUTPUT_DIR, f"lp_max_iter_{problem_type}_tol{ZERO_TOLERANCE}_maxiter{MAX_ITERATIONS}_seed{seed}_problems{num_problems}.png"))
        
        # Generate benchmark comparison plots
        plot_comparison_with_benchmarks(results, 'Avg_Iterations', problem_type, 
                       os.path.join(OUTPUT_DIR, f"lp_iterations_vs_benchmarks_{problem_type}_tol{ZERO_TOLERANCE}_maxiter{MAX_ITERATIONS}_seed{seed}_problems{num_problems}.png"))
        plot_comparison_with_benchmarks(results, 'Avg_Time', problem_type, 
                       os.path.join(OUTPUT_DIR, f"lp_time_vs_benchmarks_{problem_type}_tol{ZERO_TOLERANCE}_maxiter{MAX_ITERATIONS}_seed{seed}_problems{num_problems}.png"))
        plot_comparison_with_benchmarks(results, 'Time_Per_Iter', problem_type, 
                       os.path.join(OUTPUT_DIR, f"lp_time_per_iter_vs_benchmarks_{problem_type}_tol{ZERO_TOLERANCE}_maxiter{MAX_ITERATIONS}_seed{seed}_problems{num_problems}.png"))
        
        print(f"\n{Colors.GREEN}Plots saved to {OUTPUT_DIR} directory.{Colors.END}")
    else:
        print(f"\n{Colors.YELLOW}Plot generation is disabled. Use the CSV file for custom analysis.{Colors.END}")
    
    # Include tolerance in the summary
    print(f"\n{Colors.BOLD}Summary of Best Methods (Random Seed: {seed}, Tolerance: {ZERO_TOLERANCE}):{Colors.END}")
    
    # For larger problems (20x20 and up)
    large_problems = results[results['Size'].isin([s for s in results['Size'] if int(s.split('x')[0]) >= 20])]
    
    if not large_problems.empty:
        print("\nFor larger problems (20x20 and up):")
        
        # Flag methods with poor agreement or max iteration issues
        has_issues = False
        
        # Check agreement with simplex and HiGHS
        for benchmark in ['Simplex', 'HiGHS']:
            agreement_cols = [col for col in large_problems.columns if f'Agreement_with_{benchmark}' in col]
            if agreement_cols:
                # Flag methods with poor agreement
                for col in agreement_cols:
                    method = col.replace(f'_Agreement_with_{benchmark}', '')
                    agreement_rate = large_problems[col].mean()
                    if agreement_rate < 100:  # Any disagreement is flagged
                        if not has_issues:
                            print(f"\n  {Colors.YELLOW}Methods with issues:{Colors.END}")
                            has_issues = True
                        print(f"  {Colors.RED}WARNING: {method} disagrees with {benchmark} in {100-agreement_rate:.1f}% of cases{Colors.END}")
        
        # Check for max iteration issues
        max_iter_cols = [col for col in large_problems.columns if 'Max_Iter_Reached' in col]
        if max_iter_cols:
            for col in max_iter_cols:
                method = col.replace('_Max_Iter_Reached', '')
                max_iter_rate = large_problems[col].mean()
                if max_iter_rate > 0:  # Any max iterations is flagged
                    if not has_issues:
                        print(f"\n  {Colors.YELLOW}Methods with issues:{Colors.END}")
                        has_issues = True
                    print(f"  {Colors.YELLOW}WARNING: {method} reached max iterations in {max_iter_rate:.1f}% of cases{Colors.END}")
        
        if not has_issues:
            print(f"\n  {Colors.GREEN}All methods agree with benchmarks and no max iterations were reached.{Colors.END}")
        
        # Best for agreement with simplex
        simplex_agreement_cols = [col for col in large_problems.columns if 'Agreement_with_Simplex' in col]
        if simplex_agreement_cols:
            avg_agreement = large_problems[simplex_agreement_cols].mean()
            best_agreement = avg_agreement.idxmax()
            method_name = best_agreement.replace('_Agreement_with_Simplex', '')
            print(f"\n  Best for agreement with Simplex: {method_name} ({avg_agreement.max():.1f}%)")
        
        # Best for agreement with HiGHS
        highs_agreement_cols = [col for col in large_problems.columns if 'Agreement_with_HiGHS' in col]
        if highs_agreement_cols:
            avg_agreement = large_problems[highs_agreement_cols].mean()
            best_agreement = avg_agreement.idxmax()
            method_name = best_agreement.replace('_Agreement_with_HiGHS', '')
            print(f"  Best for agreement with HiGHS: {method_name} ({avg_agreement.max():.1f}%)")
        
        # Best for success rate
        success_cols = [col for col in large_problems.columns if 'Success_Rate' in col]
        if success_cols:
            avg_success = large_problems[success_cols].mean()
            best_success = avg_success.idxmax()
            method_name = best_success.replace('_Success_Rate', '')
            print(f"  Best for success rate: {method_name} ({avg_success.max():.1f}%)")
        
        # Show simplex performance
        if 'Simplex_Avg_Iterations' in large_problems.columns:
            avg_simplex_iter = large_problems['Simplex_Avg_Iterations'].mean()
            avg_simplex_time = large_problems['Simplex_Avg_Time'].mean()
            avg_simplex_time_per_iter = large_problems['Simplex_Time_Per_Iter'].mean()
            print(f"  Simplex benchmark: {avg_simplex_iter:.2f} iterations, {avg_simplex_time:.4f}s, {avg_simplex_time_per_iter:.6f}s/iter")
        
        # Show HiGHS performance
        if 'HiGHS_Avg_Iterations' in large_problems.columns:
            avg_highs_iter = large_problems['HiGHS_Avg_Iterations'].mean()
            avg_highs_time = large_problems['HiGHS_Avg_Time'].mean()
            avg_highs_time_per_iter = large_problems['HiGHS_Time_Per_Iter'].mean()
            print(f"  HiGHS benchmark: {avg_highs_iter:.2f} iterations, {avg_highs_time:.4f}s, {avg_highs_time_per_iter:.6f}s/iter")
        
        # Best for avoiding max iterations only if any method reached max
        if any(large_problems[col].mean() > 0 for col in max_iter_cols):
            avg_max_iter = large_problems[max_iter_cols].mean()
            best_max_iter = avg_max_iter.idxmin()
            method_name = best_max_iter.replace('_Max_Iter_Reached', '')
            print(f"  Best for avoiding max iterations: {method_name} (reached max in {avg_max_iter.min():.1f}% of cases)")
        
        # Best for iterations
        iter_cols = [col for col in large_problems.columns if 'Avg_Iterations' in col 
                     and col != 'Simplex_Avg_Iterations' and col != 'HiGHS_Avg_Iterations'
                     and not col.endswith('_All')]  # Exclude the _All columns
        if iter_cols:
            avg_iters = large_problems[iter_cols].mean()
            best_iters = avg_iters.idxmin()
            method_name = best_iters.replace('_Avg_Iterations', '')
            print(f"  Best for low iterations (excluding max iter cases): {method_name} ({avg_iters.min():.2f} iterations)")
            
        # Also show best including all iterations
        iter_all_cols = [col for col in large_problems.columns if col.endswith('_Avg_Iterations_All') 
                         and not col.startswith('Simplex_') and not col.startswith('HiGHS_')]
        if iter_all_cols:
            avg_iters_all = large_problems[iter_all_cols].mean()
            best_iters_all = avg_iters_all.idxmin()
            method_name_all = best_iters_all.replace('_Avg_Iterations_All', '')
            print(f"  Best for low iterations (all cases): {method_name_all} ({avg_iters_all.min():.2f} iterations)")
        
        # Best for speed
        time_cols = [col for col in large_problems.columns if 'Avg_Time' in col 
                    and col != 'Simplex_Avg_Time' and col != 'HiGHS_Avg_Time']
        if time_cols:
            avg_times = large_problems[time_cols].mean()
            best_time = avg_times.idxmin()
            method_name = best_time.replace('_Avg_Time', '')
            print(f"  Best for speed: {method_name} ({avg_times.min():.4f}s)")
        
        # Best for time per iteration
        time_per_iter_cols = [col for col in large_problems.columns if 'Time_Per_Iter' in col 
                             and col != 'Simplex_Time_Per_Iter' and col != 'HiGHS_Time_Per_Iter']
        if time_per_iter_cols:
            avg_time_per_iter = large_problems[time_per_iter_cols].mean()
            best_time_per_iter = avg_time_per_iter.idxmin()
            method_name = best_time_per_iter.replace('_Time_Per_Iter', '')
            print(f"  Best for time per iteration: {method_name} ({avg_time_per_iter.min():.6f}s/iter)")
    
    # Save the complete summary to a text file
    summary_file = os.path.join(OUTPUT_DIR, f"lp_summary_{problem_type}_sparsity{sparsity}_tol{ZERO_TOLERANCE}_maxiter{MAX_ITERATIONS}_seed{seed}_problems{num_problems}.txt")
    with open(summary_file, "w") as f:
        f.write(f"LP Feasibility Methods Comparison Summary\n")
        f.write(f"=====================================================\n\n")
        f.write(f"Parameters:\n")
        f.write(f"- Problem type: {problem_type}\n")
        f.write(f"- Random seed: {seed}\n")
        f.write(f"- Number of problems per size: {num_problems}\n")
        f.write(f"- Problem sizes: {', '.join([f'{m}x{n}' for m, n in problem_sizes])}\n")
        f.write(f"- Sparsity: {sparsity}\n")
        f.write(f"- Tolerance: {ZERO_TOLERANCE}\n")
        f.write(f"- Max iterations: {MAX_ITERATIONS}\n\n")
        
        f.write(f"Methods compared:\n")
        f.write(f"1. IAM-1 Standard (with exact L2 norm)\n")
        f.write(f"2. Simplex Phase 1: Two-phase simplex method implementation for LP feasibility problems - BENCHMARK\n")
        f.write(f"3. HiGHS (high-performance solver using SciPy's linprog with zero objective, presolve disabled)\n")
        f.write(f"4. Pan's Method with Row Scaling: Implementation of Pan's Most-Obtuse-Angle Column Rule algorithm with initial row scaling\n\n")
        
        f.write(f"Comparison with benchmark solvers:\n")
        f.write(f"- Methods that consistently agree with Simplex Phase 1 and HiGHS are more reliable for real-world applications\n")
        f.write(f"- High disagreement rates may indicate numerical instability or issues with max iteration limits\n")
        f.write(f"- Lower iteration counts and faster runtime than benchmarks are advantages of IAM methods\n\n")
        
        f.write(f"Numerical tolerance impact:\n")
        f.write(f"- Higher tolerance (e.g. 1e-5) may lead to faster convergence but possibly less accurate results\n")
        f.write(f"- Lower tolerance (e.g. 1e-10) may improve accuracy but may require more iterations to converge\n")
        f.write(f"- Current tolerance: {ZERO_TOLERANCE}\n\n")
        
        f.write(f"Note on iteration counts:\n")
        f.write(f"- 'Avg_Iterations' excludes cases that reached maximum iterations\n")
        f.write(f"- 'Avg_Iterations_All' includes all cases, including those that reached maximum iterations\n")
        f.write(f"- Max_Iter_Reached shows the percentage of problems where the method hit the iteration limit\n\n")
        
        f.write(f"Key differences between methods:\n")
        f.write(f"1. Simplex Phase 1: Standard simplex method implementation with numerical stability enhancements - BENCHMARK\n")
        f.write(f"2. HiGHS: High-performance implementation of simplex method with zero objective vector\n")
        f.write(f"3. IAM-1 Standard: Uses exact L2 norm for calculating projections in every iteration\n")
        f.write(f"4. Pan's Method with Row Scaling: Implementation of Pan's Most-Obtuse-Angle Column Rule algorithm with initial row scaling\n\n")
        
    print(f"\n{Colors.GREEN}Complete summary saved to {summary_file}{Colors.END}")


def problem_benchmark(A, b, problem_index, num_problems=None, timeout=10.0, max_iter=MAX_ITERATIONS, tol=ZERO_TOLERANCE):
    """Benchmark all solvers on a single problem."""
    if num_problems:
        progress_str = f"Problem {problem_index}/{num_problems}..."
        print(f"  {progress_str}", end="", flush=True)
    
    results = {}
    methods = []
    
    # First, check if the problem is feasible using Simplex Phase 1 as the benchmark
    simplex_start_time = time.time()
    try:
        simplex_result = solve_with_simplex_phase1(A, b, max_iter=max_iter, tol=tol)
        simplex_time = simplex_result['time']
        is_simplex_feasible = simplex_result['status'] == 'feasible'
        simplex_iterations = simplex_result['iterations']
    except Exception as e:
        print(f"Simplex Phase 1 solver error: {e}")
        is_simplex_feasible = False
        simplex_time = 0
        simplex_iterations = 0
    
    # Check if the problem is feasible using HiGHS
    highs_start_time = time.time()
    try:
        # Use SciPy's linprog with HiGHS method
        highs_result = linprog(
            np.zeros(A.shape[1]), 
            A_ub=A, 
            b_ub=b, 
            method='highs', 
            options={'presolve': False}  # No maxiter limit for HiGHS
        )
        highs_time = time.time() - highs_start_time
        is_highs_feasible = highs_result.success
        highs_iterations = highs_result.nit if hasattr(highs_result, 'nit') else 0
    except Exception as e:
        print(f"HiGHS solver error: {e}")
        is_highs_feasible = False
        highs_time = 0
        highs_iterations = 0
    
    # Run each solver method
    solver_methods = [
        ("simplex", lambda A, b: {
            'status': 'feasible' if is_simplex_feasible else 'infeasible',
            'iterations': simplex_iterations,
            'time': simplex_time
        }),
        ("highs", lambda A, b: {
            'status': 'feasible' if is_highs_feasible else 'infeasible',
            'iterations': highs_iterations,
            'time': highs_time
        }),
        # Removed IAM variants, keeping only IAM-1 Standard
        ("iam1", iam1_standard),
        # Using simplex_phase1 as benchmark instead
        # ("simplex_phase1", solve_with_simplex_phase1),
        # Temporarily disabled due to indentation errors in the file
        # ("avf_simplex_phase1", solve_with_avf_simplex_phase1),
        # Add Pan's method with row scaling
        ("pans_method", check_feasibility_pan_row_scaled)
    ]
    
    for name, solve_func in solver_methods:
        methods.append(name)
        
        if name not in ["simplex", "highs"]:  # We already ran these above
            try:
                # Get the result from the solver with max_iter and tol parameters
                result = solve_func(A, b, max_iter=max_iter, tol=tol)
                
                # Store both the status and number of iterations
                results[f"{name}_status"] = result['status']
                results[f"{name}_iter"] = result['iterations']
                
                # Store the internal computation time
                results[f"{name}_time"] = result['time']
                
                # Check if result agrees with simplex and HiGHS
                simplex_status = 'feasible' if is_simplex_feasible else 'infeasible'
                results[f"{name}_agrees_with_simplex"] = result['status'] == simplex_status
                
                highs_status = 'feasible' if is_highs_feasible else 'infeasible'
                results[f"{name}_agrees_with_highs"] = result['status'] == highs_status
                
            except Exception as e:
                print(f"Error with {name}: {e}")
                results[f"{name}_status"] = "error"
                results[f"{name}_iter"] = 0
                results[f"{name}_time"] = 0
                results[f"{name}_agrees_with_simplex"] = False
                results[f"{name}_agrees_with_highs"] = False
        elif name == "simplex":
            # Store simplex results separately since we already computed them
            results[f"{name}_status"] = 'feasible' if is_simplex_feasible else 'infeasible'
            results[f"{name}_iter"] = simplex_iterations
            results[f"{name}_time"] = simplex_time
            results[f"{name}_agrees_with_simplex"] = True  # Simplex always agrees with itself
            results[f"{name}_agrees_with_highs"] = is_simplex_feasible == is_highs_feasible
        elif name == "highs":
            # Store HiGHS results separately since we already computed them
            results[f"{name}_status"] = 'feasible' if is_highs_feasible else 'infeasible'
            results[f"{name}_iter"] = highs_iterations
            results[f"{name}_time"] = highs_time
            results[f"{name}_agrees_with_simplex"] = is_highs_feasible == is_simplex_feasible
            results[f"{name}_agrees_with_highs"] = True  # HiGHS always agrees with itself
    
    # Mark completion
    if num_problems:
        print(" Done")
    
    return results, methods


if __name__ == "__main__":
    main() 