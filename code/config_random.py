"""
Centralized configuration for random problems comparison.

This module imports and overrides parameters from each method's configuration
to provide centralized control for random problem benchmarks.
"""

# Import all parameters from each solver's config
from config_iam import *
from config_simplex import *
from config_pan import *

# Override parameters for random problem tests
RANDOM_MAX_ITERATIONS = 1000     # Maximum iterations for random problems

# Method-specific overrides for random problems
IAM1_MAX_ITERATIONS = RANDOM_MAX_ITERATIONS      # Override for IAM-1
SIMPLEX_MAX_ITERATIONS = RANDOM_MAX_ITERATIONS   # Override for Simplex Phase 1
PAN_MAX_ITERATIONS = RANDOM_MAX_ITERATIONS       # Override for Pan's method

# Override MAX_ITERATIONS from individual configs
MAX_ITERATIONS = RANDOM_MAX_ITERATIONS

# Global tolerances for all methods in random problems
ZERO_TOLERANCE = 1e-8
FEASIBILITY_TOLERANCE = 1e-7

# Random problem generation parameters
DEFAULT_SPARSITY = 0.0           # Default sparsity for random problems
DEFAULT_BOUND = 50               # Default bound for random coefficients
MAX_GENERATION_ATTEMPTS = 1500   # Maximum attempts for problem generation
PROBLEMS_PER_SIZE = 10           # Number of problems per size

# Benchmark parameters
TIMEOUT_SECONDS = 10             # Timeout for individual tests in seconds
DEFAULT_PROBLEM_SIZES = [
    (10, 20), (20, 40), (30, 60), (40, 80), 
    (50, 100), (100, 200), (200, 400)
]

# Visualization parameters
PLOT_DPI = 300                   # DPI for saved plots
PLOT_SIZE_INCHES = (12, 8)       # Size of plots in inches
SHOW_PLOTS = True                # Whether to display plots in addition to saving

def get_random_config_info():
    """Return a string representation of the random test configuration."""
    config_info = "Random Problem Test Configuration:\n"
    config_info += f"- MAX_ITERATIONS: {MAX_ITERATIONS}\n"
    config_info += f"- ZERO_TOLERANCE: {ZERO_TOLERANCE}\n"
    config_info += f"- FEASIBILITY_TOLERANCE: {FEASIBILITY_TOLERANCE}\n"
    config_info += f"- DEFAULT_SPARSITY: {DEFAULT_SPARSITY}\n"
    config_info += f"- PROBLEMS_PER_SIZE: {PROBLEMS_PER_SIZE}\n"
    config_info += f"- TIMEOUT_SECONDS: {TIMEOUT_SECONDS}\n"
    
    # Show method-specific overrides
    config_info += "\nMethod-specific parameters:\n"
    config_info += f"- IAM1_MAX_ITERATIONS: {IAM1_MAX_ITERATIONS}\n"
    config_info += f"- SIMPLEX_MAX_ITERATIONS: {SIMPLEX_MAX_ITERATIONS}\n"
    config_info += f"- PAN_MAX_ITERATIONS: {PAN_MAX_ITERATIONS}\n"
    
    return config_info 