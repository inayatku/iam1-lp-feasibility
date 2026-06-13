"""
Configuration parameters specific to the Simplex Phase 1 method.

This module provides centralized configuration parameters for the Simplex Phase 1 solver.
"""
import numpy as np

# ------------------------------
# Basic numerical tolerance parameters
# ------------------------------
ZERO_TOLERANCE = 1e-8           # Default tolerance for treating values as zero
FEASIBILITY_TOLERANCE = 1e-7    # Values >= -FEASIBILITY_TOLERANCE are considered feasible
PIVOT_TOLERANCE = 1e-10         # Tolerance for pivot selection (tighter than zero tolerance)
PERTURB_TOLERANCE = 1e-14       # Tolerance for perturbation in degenerate cases

# ------------------------------
# Floating-point precision controls
# ------------------------------
# Factors relative to machine epsilon for precision-related checks
EPSILON_FACTOR_SMALL = 10       # For small value detection (epsilon * 10)
EPSILON_FACTOR_MEDIUM = 100     # For medium threshold checks (epsilon * 100)
EPSILON_FACTOR_LARGE = 1000     # For larger threshold checks (epsilon * 1000)
# Machine epsilon for float64 (~2.22e-16)
MACHINE_EPSILON = np.finfo(np.float64).eps
# Derived epsilon-based tolerances for different precision needs
EPSILON_SMALL = MACHINE_EPSILON * EPSILON_FACTOR_SMALL
EPSILON_MEDIUM = MACHINE_EPSILON * EPSILON_FACTOR_MEDIUM
EPSILON_LARGE = MACHINE_EPSILON * EPSILON_FACTOR_LARGE

# ------------------------------
# Iteration control parameters
# ------------------------------
MAX_ITERATIONS = 6000           # Default maximum number of iterations
CYCLING_CHECK_FREQUENCY = 10    # Check for cycling every N iterations
CYCLING_CHECK_START = 20        # Start checking for cycling after this many iterations
MAX_CYCLING_ITERATIONS = 100    # Skip to next method after this many iterations of cycling

# ------------------------------
# Numerical stability parameters
# ------------------------------
BLAND_RULE_AFTER_ITER = 100     # Apply Bland's rule after this many iterations
MIN_PIVOT_VALUE = 1e-10         # Minimum absolute value for pivot elements
PIVOT_SAFETY_FACTOR = 100       # Safety factor for extremely small pivots
BLAND_RULE_TRIGGER_COUNT = 3    # Number of consecutive iterations with small improvement to trigger Bland's rule
STALLING_TOLERANCE = 1e-10      # Tolerance for detecting stalling in simplex
MAX_VALUE = np.finfo(np.float64).max / 10  # Maximum value limit to prevent overflow
MIN_VALUE = np.finfo(np.float64).tiny * 10  # Minimum non-zero value to prevent underflow
NUMERICAL_HEALTH_CHECK_FREQUENCY = 10  # Check numerical health every N iterations

# ------------------------------
# Matrix handling parameters
# ------------------------------
SPARSE_MATRIX_SIZE_THRESHOLD = 1e9  # Use sparse matrices if problem size exceeds this (in bytes)

# ------------------------------
# Method specific parameters
# ------------------------------
USE_BLAND_RULE = False           # Whether to use Bland's rule for anti-cycling
USE_STEEPEST_EDGE = False       # Whether to use steepest edge pricing
TRACK_BASIS_HISTORY = True      # Whether to track basis history for cycle detection
MAX_ARTIFICIAL_VARS = 50000     # Maximum number of artificial variables to add

# ------------------------------
# Output control parameters
# ------------------------------
KEY_PRESS_CHECK_FREQUENCY = 10  # Check for key press every N iterations
PROGRESS_DOT_FREQUENCY = 1      # Print a progress dot every N iterations
PROGRESS_LINE_FREQUENCY = 50    # Print a newline in progress output every N iterations
DICTIONARY_OUTPUT_DETAIL = 1    # Level of detail in dictionary/tableau output (0=minimal, 1=normal, 2=detailed)

# ------------------------------
# Display formatting parameters
# ------------------------------
COLUMN_WIDTH = 10               # Width for column display in tableau output
DECIMAL_PRECISION = 4           # Decimal places in numeric output
ENABLE_COLOR_OUTPUT = True      # Whether to use ANSI color codes in output

def get_config_info():
    """Return a string representation of the current Simplex Phase 1 configuration."""
    config_info = "Current Simplex Phase 1 Configuration:\n"
    config_info += f"- ZERO_TOLERANCE: {ZERO_TOLERANCE}\n"
    config_info += f"- FEASIBILITY_TOLERANCE: {FEASIBILITY_TOLERANCE}\n"
    config_info += f"- MACHINE_EPSILON: {MACHINE_EPSILON:.1e}\n"
    config_info += f"- EPSILON_SMALL: {EPSILON_SMALL:.1e}\n"
    config_info += f"- EPSILON_MEDIUM: {EPSILON_MEDIUM:.1e}\n"
    config_info += f"- MAX_ITERATIONS: {MAX_ITERATIONS}\n"
    config_info += f"- MAX_CYCLING_ITERATIONS: {MAX_CYCLING_ITERATIONS}\n"
    config_info += f"- USE_BLAND_RULE: {'Yes' if USE_BLAND_RULE else 'No'}\n"
    config_info += f"- USE_STEEPEST_EDGE: {'Yes' if USE_STEEPEST_EDGE else 'No'}\n"
    config_info += f"- TRACK_BASIS_HISTORY: {'Yes' if TRACK_BASIS_HISTORY else 'No'}\n"
    config_info += f"- PIVOT_TOLERANCE: {PIVOT_TOLERANCE}\n"
    config_info += f"- MIN_PIVOT_VALUE: {MIN_PIVOT_VALUE}\n"
    config_info += f"- PIVOT_SAFETY_FACTOR: {PIVOT_SAFETY_FACTOR}\n"
    config_info += f"- MAX_VALUE: {MAX_VALUE:.1e}\n"
    config_info += f"- MIN_VALUE: {MIN_VALUE:.1e}\n"
    config_info += f"- COLUMN_WIDTH: {COLUMN_WIDTH}\n"
    config_info += f"- ENABLE_COLOR_OUTPUT: {'Yes' if ENABLE_COLOR_OUTPUT else 'No'}\n"
    return config_info 
# Fix (June 2026): this constant exists in config_iam.py / config_netlib.py but was
# missing here, breaking simplex_phase_1.py when called from the random-LP driver
# (its caller-dependent import falls back to config_simplex). Same value as config_iam.
MIN_ROW_SCALE_VALUE = np.finfo(np.float64).tiny * 10

# Fix (June 2026): constants used by simplex_phase_1.py but missing here;
# values taken verbatim from config_iam.py/config_netlib.py.
MAX_ROW_SCALE_VALUE = np.finfo(np.float64).max / 10   # Maximum safe value for row scaling  # from config_iam.py
