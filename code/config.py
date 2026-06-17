#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py

Compatibility layer for older files that import from 'config' instead of specific
config modules like 'config_iam', 'config_pan', etc.

This module imports and re-exports commonly used configuration parameters from the
specific configuration modules to maintain backward compatibility.
"""

# Import and re-export configuration from config_iam.py
from config_iam import (
    ZERO_TOLERANCE,
    FEASIBILITY_TOLERANCE,
    ADMISSIBLE_COL_TOLERANCE,
    MAX_ITERATIONS,
    CYCLING_CHECK_FREQUENCY,
    MAX_CYCLING_ITERATIONS,
    PIVOT_MIN_VALUE,
    PIVOT_SAFETY_FACTOR,
    ENABLE_ROW_SCALING,
    MIN_ROW_SCALE_VALUE,
    MAX_ROW_SCALE_VALUE,
    IAM1_USE_EXACT_NORM,
    KEY_PRESS_CHECK_FREQUENCY,
    PROGRESS_DOT_FREQUENCY,
    PROGRESS_LINE_FREQUENCY,
    DICTIONARY_OUTPUT_DETAIL,
    COLUMN_WIDTH,
    DECIMAL_PRECISION,
    ENABLE_COLOR_OUTPUT,
    get_config_info
)

# Additional parameters from config_pan.py
try:
    from config_pan import (
        PIVOT_TOLERANCE,
        PERTURB_TOLERANCE,
        PANS_APPLY_ROW_SCALING,
        USE_MOST_NEGATIVE_SELECTION,
        TRACK_BASIS_HISTORY,
        STORE_PIVOT_HISTORY,
        CYCLING_CHECK_START,
        STANDARD_PAN_MESSAGE
    )
except ImportError:
    # Default values if config_pan is not available
    PIVOT_TOLERANCE = ZERO_TOLERANCE
    PERTURB_TOLERANCE = 1e-10
    PANS_APPLY_ROW_SCALING = ENABLE_ROW_SCALING
    USE_MOST_NEGATIVE_SELECTION = True
    TRACK_BASIS_HISTORY = True
    STORE_PIVOT_HISTORY = False
    CYCLING_CHECK_START = 10
    STANDARD_PAN_MESSAGE = "Pan's Method Implementation"

# Print a warning message when this compatibility module is used
import warnings
warnings.warn(
    "Importing from 'config' is deprecated. Please import from specific config modules "
    "like 'config_iam', 'config_pan', etc. instead.",
    DeprecationWarning,
    stacklevel=2
) 