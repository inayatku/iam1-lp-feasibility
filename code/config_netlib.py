"""
Centralized configuration for NETLIB problems comparison.

This module imports and overrides parameters from each method's configuration
to provide centralized control for NETLIB benchmarks.

IMPORTANT: This file explicitly overrides parameters from individual config files,
so any changes made here will take precedence over the original config files.

HOW THIS WORKS:
1. When scripts import directly from individual config files (config_iam.py, config_pan.py, etc.),
   they get the standard precision settings defined in those files.
2. When scripts import from THIS file (config_netlib.py), they get the high-precision
   overrides defined below, which are designed for the NETLIB benchmarks.
3. NETLIB override values are set to maximum float64 precision based on machine epsilon
   regardless of the standard values in the individual config files.
"""
import numpy as np

# Import all parameters from each solver's config
from config_iam import *
from config_simplex import *
from config_pan import *

# -----------------------------
# Machine precision parameters
# -----------------------------
# Machine epsilon for float64 (~2.22e-16)
MACHINE_EPSILON = np.finfo(np.float64).eps
# Factors for different precision needs - MODIFIED: use same factors as in config_iam.py
EPSILON_FACTOR_SMALL = 10        # For small value detection
EPSILON_FACTOR_MEDIUM = 100      # For medium threshold checks
EPSILON_FACTOR_LARGE = 1000      # For larger threshold checks
# Derived epsilon-based tolerances
EPSILON_SMALL = MACHINE_EPSILON * EPSILON_FACTOR_SMALL    # ~2.22e-15
EPSILON_MEDIUM = MACHINE_EPSILON * EPSILON_FACTOR_MEDIUM  # ~2.22e-14
EPSILON_LARGE = MACHINE_EPSILON * EPSILON_FACTOR_LARGE    # ~2.22e-13

# -----------------------------
# Iteration control overrides
# -----------------------------
NETLIB_MAX_ITERATIONS = 6000     # Maximum iterations for NETLIB problems

# Method-specific overrides for NETLIB problems
IAM1_MAX_ITERATIONS = NETLIB_MAX_ITERATIONS      # Override for IAM-1
SIMPLEX_MAX_ITERATIONS = NETLIB_MAX_ITERATIONS   # Override for Simplex Phase 1
PAN_MAX_ITERATIONS = NETLIB_MAX_ITERATIONS       # Override for Pan's method

# Override MAX_ITERATIONS from individual configs
MAX_ITERATIONS = NETLIB_MAX_ITERATIONS

# -----------------------------
# Numerical precision overrides - MODIFIED: match config_iam.py for better compatibility
# -----------------------------
# Global tolerances for all methods in NETLIB tests (based on machine epsilon)
ZERO_TOLERANCE = 1e-8                    # Match config_iam.py value
FEASIBILITY_TOLERANCE = 1e-7             # Match config_iam.py value
ADMISSIBLE_COL_TOLERANCE = 1e-8          # Match config_iam.py value
PIVOT_MIN_VALUE = 1e-10                  # Match config_iam.py value
PIVOT_TOLERANCE = 1e-10                  # Match config_iam.py value

# -----------------------------
# Condition number thresholds
# -----------------------------
# Thresholds for condition number warnings/handling
CONDITION_WARNING_THRESHOLD = 1.0 / (MACHINE_EPSILON * 100)  # Alert at 1% of inverse epsilon
CONDITION_CRITICAL_THRESHOLD = 1.0 / (MACHINE_EPSILON * 10)  # Critical at 10% of inverse epsilon
CONDITION_RECOVERY_THRESHOLD = 1e8  # Threshold for applying condition number recovery techniques

# -----------------------------
# Scaling parameter overrides - MODIFIED: use values from config_iam.py
# -----------------------------
PIVOT_SAFETY_FACTOR = 100       # Match config_iam.py value
MIN_ROW_SCALE_VALUE = np.finfo(np.float64).tiny * 10  # Match config_iam.py value
MAX_ROW_SCALE_VALUE = np.finfo(np.float64).max / 10   # Match config_iam.py value
DEFAULT_INFEASIBLE_PROJECTION_VALUE = float('-inf')   # Match config_iam.py value

# -----------------------------
# Performance parameter overrides
# -----------------------------
ENABLE_ROW_SCALING = True        # Enable row scaling for all methods
IAM1_USE_EXACT_NORM = True       # Use exact L2 norm for IAM1 (more accurate)
USE_SPARSE_MATRICES = True       # Use sparse matrices for large NETLIB problems

# -----------------------------
# Advanced numerical stability controls
# -----------------------------
ENABLE_ITERATIVE_REFINEMENT = True  # Enable iterative refinement for improved precision
PRECISION_RECOVERY_FREQUENCY = "adaptive"  # Use adaptive frequency for stability checks
QR_HOUSEHOLDER_THRESHOLD = CONDITION_RECOVERY_THRESHOLD  # When to use Householder QR
STABILITY_CHECK_BASE_FREQUENCY = 10  # Base frequency for stability checks
STABILITY_CHECK_SCALING_FACTOR = 5   # Scaling factor after instability detection
ADVANCED_PIVOTING_STRATEGY = "complete"  # Use complete pivoting strategy for maximum stability
GRADIENT_CANCELLATION_PREVENTION = True  # Enable special handling for gradient cancellation cases

# -----------------------------
# Display formatting overrides - MODIFIED: matching config_iam.py
# -----------------------------
COLUMN_WIDTH = 10                # Width for column display in dictionary output (match config_iam.py)
DECIMAL_PRECISION = 4            # Use standard precision from config_iam.py
ENABLE_COLOR_OUTPUT = True       # Whether to use ANSI color codes in output

# -----------------------------
# Test and visualization parameters
# -----------------------------
RESUME_INTERRUPTED_TESTS = True  # Resume from checkpoint when restart
GENERATE_VISUALIZATIONS = True   # Generate plots and visualizations
MAX_PROBLEMS_PER_BATCH = 10      # Maximum problems per batch

def get_netlib_config_info():
    """Return a string representation of the NETLIB test configuration."""
    config_info = "NETLIB Test Configuration (Compatible with config_iam.py):\n"
    config_info += f"- MAX_ITERATIONS: {MAX_ITERATIONS}\n"
    config_info += f"- MACHINE_EPSILON: {MACHINE_EPSILON:.1e}\n"
    config_info += f"- ZERO_TOLERANCE: {ZERO_TOLERANCE:.1e}\n"
    config_info += f"- FEASIBILITY_TOLERANCE: {FEASIBILITY_TOLERANCE:.1e}\n"
    config_info += f"- PIVOT_TOLERANCE: {PIVOT_TOLERANCE:.1e}\n"
    config_info += f"- PIVOT_MIN_VALUE: {PIVOT_MIN_VALUE:.1e}\n"
    config_info += f"- CONDITION_WARNING_THRESHOLD: {CONDITION_WARNING_THRESHOLD:.1e}\n"
    config_info += f"- CONDITION_CRITICAL_THRESHOLD: {CONDITION_CRITICAL_THRESHOLD:.1e}\n"
    
    # Advanced numerical stability settings
    config_info += "\nAdvanced Numerical Stability Controls:\n"
    config_info += f"- ENABLE_ITERATIVE_REFINEMENT: {ENABLE_ITERATIVE_REFINEMENT}\n"
    config_info += f"- PRECISION_RECOVERY_FREQUENCY: {PRECISION_RECOVERY_FREQUENCY}\n"
    config_info += f"- ADVANCED_PIVOTING_STRATEGY: {ADVANCED_PIVOTING_STRATEGY}\n"
    config_info += f"- GRADIENT_CANCELLATION_PREVENTION: {GRADIENT_CANCELLATION_PREVENTION}\n"
    
    # Scaling settings
    config_info += "\nScaling Settings:\n"
    config_info += f"- PIVOT_SAFETY_FACTOR: {PIVOT_SAFETY_FACTOR}\n"
    config_info += f"- MIN_ROW_SCALE_VALUE: {MIN_ROW_SCALE_VALUE:.1e}\n"
    config_info += f"- MAX_ROW_SCALE_VALUE: {MAX_ROW_SCALE_VALUE:.1e}\n"
    config_info += f"- DECIMAL_PRECISION: {DECIMAL_PRECISION}\n"
    config_info += f"- ROW_SCALING: {'Enabled' if ENABLE_ROW_SCALING else 'Disabled'}\n"
    config_info += f"- IAM1_USE_EXACT_NORM: {'Yes' if IAM1_USE_EXACT_NORM else 'No'}\n"
    config_info += f"- USE_SPARSE_MATRICES: {'Yes' if USE_SPARSE_MATRICES else 'No'}\n"
    
    # Show method-specific overrides
    config_info += "\nMethod-specific parameters:\n"
    config_info += f"- IAM1_MAX_ITERATIONS: {IAM1_MAX_ITERATIONS}\n"
    config_info += f"- SIMPLEX_MAX_ITERATIONS: {SIMPLEX_MAX_ITERATIONS}\n"
    config_info += f"- PAN_MAX_ITERATIONS: {PAN_MAX_ITERATIONS}\n"
    
    return config_info 