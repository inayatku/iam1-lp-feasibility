"""
Configuration parameters specific to IAM-1 (Inayatullah's Angle Method Type 1).

This module provides centralized configuration parameters for the IAM-1 solver.
"""
import numpy as np

# ------------------------------
# Basic numerical tolerance parameters
# ------------------------------
# Standard precision tolerances (same as config_pan.py and config_simplex.py)
ZERO_TOLERANCE = 1e-8           # Standard tolerance for treating values as zero
FEASIBILITY_TOLERANCE = 1e-7    # Values >= -FEASIBILITY_TOLERANCE are considered feasible
ADMISSIBLE_COL_TOLERANCE = 1e-8 # Tolerance for identifying admissible columns

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
PIVOT_MIN_VALUE = 1e-10         # Minimum absolute value for pivot elements (same as other configs)
PIVOT_TOLERANCE = 1e-10         # Tolerance for pivot selection (same as other configs)
PIVOT_SAFETY_FACTOR = 100       # Safety factor for extremely small pivots
DEFAULT_INFEASIBLE_PROJECTION_VALUE = float('-inf')  # Value returned for infeasible projections

# ------------------------------
# Scaling parameters
# ------------------------------
ENABLE_ROW_SCALING = True       # Whether to apply row scaling by default
ENABLE_DYNAMIC_ROW_SCALING = True  # Whether to apply row scaling during pivoting for numerical stability
MIN_ROW_SCALE_VALUE = np.finfo(np.float64).tiny * 10  # Minimum safe value for row scaling
MAX_ROW_SCALE_VALUE = np.finfo(np.float64).max / 10   # Maximum safe value for row scaling

# Column scaling parameters
ENABLE_COLUMN_SCALING = False   # Whether to apply column scaling by default (L2-norm scaling)
ENABLE_DYNAMIC_COLUMN_SCALING = False  # Whether to apply column scaling during pivoting
MIN_COL_SCALE_VALUE = np.finfo(np.float64).tiny * 10  # Minimum safe value for column scaling
MAX_COL_SCALE_VALUE = np.finfo(np.float64).max / 10   # Maximum safe value for column scaling

# ------------------------------
# Method specific parameters
# ------------------------------
IAM1_USE_EXACT_NORM = True      # Use exact L2 norm for IAM1 (more accurate but slower)

# ------------------------------
# Output control parameters
# ------------------------------
KEY_PRESS_CHECK_FREQUENCY = 10  # Check for key press every N iterations
PROGRESS_DOT_FREQUENCY = 1      # Print a progress dot every N iterations
PROGRESS_LINE_FREQUENCY = 50    # Print a newline in progress output every N iterations
DICTIONARY_OUTPUT_DETAIL = 1    # Level of detail in dictionary output (0=minimal, 1=normal, 2=detailed)

# ------------------------------
# Display formatting parameters
# ------------------------------
COLUMN_WIDTH = 10               # Width for column display in dictionary output
DECIMAL_PRECISION = 4           # Decimal places in numeric output (same as other configs)
ENABLE_COLOR_OUTPUT = True      # Whether to use ANSI color codes in output

def get_config_info():
    """Return a string representation of the current IAM-1 configuration."""
    config_info = "Current IAM-1 Configuration:\n"
    config_info += f"- ZERO_TOLERANCE: {ZERO_TOLERANCE:.1e}\n"
    config_info += f"- FEASIBILITY_TOLERANCE: {FEASIBILITY_TOLERANCE:.1e}\n"
    config_info += f"- MACHINE_EPSILON: {MACHINE_EPSILON:.1e}\n"
    config_info += f"- EPSILON_SMALL: {EPSILON_SMALL:.1e}\n"
    config_info += f"- EPSILON_MEDIUM: {EPSILON_MEDIUM:.1e}\n"
    config_info += f"- MAX_ITERATIONS: {MAX_ITERATIONS}\n"
    config_info += f"- MAX_CYCLING_ITERATIONS: {MAX_CYCLING_ITERATIONS}\n"
    config_info += f"- ROW_SCALING: {'Enabled' if ENABLE_ROW_SCALING else 'Disabled'}\n"
    config_info += f"- DYNAMIC_ROW_SCALING: {'Enabled' if ENABLE_DYNAMIC_ROW_SCALING else 'Disabled'}\n"
    config_info += f"- IAM1_USE_EXACT_NORM: {'Yes' if IAM1_USE_EXACT_NORM else 'No'}\n"
    config_info += f"- ADMISSIBLE_COL_TOLERANCE: {ADMISSIBLE_COL_TOLERANCE:.1e}\n"
    config_info += f"- PIVOT_MIN_VALUE: {PIVOT_MIN_VALUE:.1e}\n"
    config_info += f"- PIVOT_SAFETY_FACTOR: {PIVOT_SAFETY_FACTOR}\n"
    config_info += f"- COLUMN_WIDTH: {COLUMN_WIDTH}\n"
    config_info += f"- DECIMAL_PRECISION: {DECIMAL_PRECISION}\n"
    config_info += f"- ENABLE_COLOR_OUTPUT: {'Yes' if ENABLE_COLOR_OUTPUT else 'No'}\n"
    return config_info 