import numpy as np
from config_iam import (
    ZERO_TOLERANCE,
    MACHINE_EPSILON,
    EPSILON_SMALL,
    EPSILON_MEDIUM,
    EPSILON_LARGE
)

def apply_zero_tolerance(vector, tol=ZERO_TOLERANCE):
    """
    Apply zero adjustment to values below tolerance.
    
    Parameters:
    -----------
    vector : numpy.ndarray
        Input vector
    tol : float, optional
        Tolerance value (default from config_iam)
        
    Returns:
    --------
    numpy.ndarray
        Vector with values below tolerance set to zero
    """
    # Use machine epsilon as a reference for smallest distinguishable values
    effective_tol = max(tol, EPSILON_SMALL)
    
    result = vector.copy()
    result[np.abs(result) < effective_tol] = 0.0
    return result

def kahan_sum(values):
    """
    Compute sum using Kahan summation algorithm for maximum precision.
    
    Parameters:
    -----------
    values : numpy.ndarray
        Values to sum
        
    Returns:
    --------
    float
        Sum with minimized floating-point error
    """
    sum_value = 0.0
    comp = 0.0  # Compensation term for lost low-order bits
    
    for val in values:
        y = val - comp  # Compensated summand
        t = sum_value + y  # Next approximation
        comp = (t - sum_value) - y  # Update compensation term
        sum_value = t
        
    return sum_value

def compute_dot_product(v1, v2, tol=ZERO_TOLERANCE):
    """
    Compute dot product between two vectors with maximum precision.
    
    Parameters:
    -----------
    v1 : numpy.ndarray
        First vector
    v2 : numpy.ndarray
        Second vector
    tol : float, optional
        Tolerance for zero values (default from config_iam)
        
    Returns:
    --------
    float
        Dot product value
    """
    # Apply zero adjustment with maximum precision
    v1_adj = apply_zero_tolerance(v1, tol)
    v2_adj = apply_zero_tolerance(v2, tol)
    
    # Compute dot product using Kahan summation for maximum precision
    products = v1_adj * v2_adj
    return kahan_sum(products)

def compute_l1_norm(vector, tol=ZERO_TOLERANCE):
    """
    Compute L1 norm (sum of absolute values) with maximum precision.
    
    Parameters:
    -----------
    vector : numpy.ndarray
        Input vector
    tol : float, optional
        Tolerance for zero values (default from config_iam)
        
    Returns:
    --------
    float
        L1 norm value
    """
    # Apply zero adjustment with maximum precision
    v_adj = apply_zero_tolerance(vector, tol)
    
    # Compute L1 norm using Kahan summation for maximum precision
    abs_values = np.abs(v_adj)
    return kahan_sum(abs_values)

def compute_l2_norm(vector, tol=ZERO_TOLERANCE):
    """
    Compute L2 norm (Euclidean norm) with maximum precision.
    
    Parameters:
    -----------
    vector : numpy.ndarray
        Input vector
    tol : float, optional
        Tolerance for zero values (default from config_iam)
        
    Returns:
    --------
    float
        L2 norm value
    """
    # Apply zero adjustment with maximum precision
    v_adj = apply_zero_tolerance(vector, tol)
    
    # Compute L2 norm using a more numerically stable approach
    # Avoid potential overflow by finding the maximum absolute value
    max_abs = np.max(np.abs(v_adj))
    
    # If max_abs is very small, just compute directly to avoid division by near-zero
    if max_abs < EPSILON_MEDIUM:
        return np.sqrt(kahan_sum(np.square(v_adj)))
    
    # Scale the vector to avoid overflow in squares
    scaled = v_adj / max_abs
    
    # Use Kahan summation for maximum precision in the sum of squares
    sum_squares = kahan_sum(np.square(scaled))
    
    # Scale back the result
    return max_abs * np.sqrt(sum_squares)

def compute_l1_to_l2_approximation(vector, tol=ZERO_TOLERANCE):
    """
    Compute L1 approximation of L2 norm with maximum precision.
    
    Parameters:
    -----------
    vector : numpy.ndarray
        Input vector
    tol : float, optional
        Tolerance for zero values (default from config_iam)
        
    Returns:
    --------
    float
        L1 approximation of L2 norm
    """
    # Apply zero adjustment with maximum precision
    v_adj = apply_zero_tolerance(vector, tol)
    
    # Count non-zero elements - faster to use direct comparison with zero after tolerance adjustment
    non_zero_count = np.count_nonzero(v_adj)
    
    # Compute L1 norm using Kahan summation for maximum precision
    l1_norm = compute_l1_norm(v_adj, tol)
    
    # Check for division by zero using machine epsilon
    if non_zero_count == 0 or l1_norm < EPSILON_SMALL:
        return 0.0
    
    # L1 approximation of L2 norm: L1 / sqrt(number of non-zero elements)
    return l1_norm / np.sqrt(non_zero_count)

def compute_normalized_dot_product_l1(v1, v2, tol=ZERO_TOLERANCE):
    """
    Compute normalized dot product using L1 norm.
    
    Parameters:
    -----------
    v1 : numpy.ndarray
        First vector
    v2 : numpy.ndarray
        Second vector
    tol : float, optional
        Tolerance for zero values (default from config_iam)
        
    Returns:
    --------
    float
        Normalized dot product value using L1 norm
    """
    # Compute dot product with maximum precision
    dot_prod = compute_dot_product(v1, v2, tol)
    
    # Compute L1 norm with maximum precision
    l1_norm_v1 = compute_l1_norm(v1, tol)
    l1_norm_v2 = compute_l1_norm(v2, tol)
    
    # Avoid division by small values
    min_norm = max(l1_norm_v1, EPSILON_SMALL)
    
    # Return normalized dot product
    return dot_prod / min_norm

def compute_normalized_dot_product_l1_approx_l2(v1, v2, tol=ZERO_TOLERANCE):
    """
    Compute normalized dot product using L1 approximation of L2 norm.
    
    Parameters:
    -----------
    v1 : numpy.ndarray
        First vector
    v2 : numpy.ndarray
        Second vector
    tol : float, optional
        Tolerance for zero values (default from config_iam)
        
    Returns:
    --------
    float
        Normalized dot product value using L1 approximation of L2 norm
    """
    # Compute dot product with maximum precision
    dot_prod = compute_dot_product(v1, v2, tol)
    
    # Compute L1 approximation of L2 norm with maximum precision
    l1_approx_l2 = compute_l1_to_l2_approximation(v1, tol)
    
    # Avoid division by small values
    min_norm = max(l1_approx_l2, EPSILON_SMALL)
    
    # Return normalized dot product
    return dot_prod / min_norm

def compute_normalized_dot_product_l2(v1, v2, tol=ZERO_TOLERANCE):
    """
    Compute normalized dot product using L2 norm (cosine similarity).
    
    Parameters:
    -----------
    v1 : numpy.ndarray
        First vector
    v2 : numpy.ndarray
        Second vector
    tol : float, optional
        Tolerance for zero values (default from config_iam)
        
    Returns:
    --------
    float
        Normalized dot product value using L2 norm
    """
    # Compute dot product with maximum precision
    dot_prod = compute_dot_product(v1, v2, tol)
    
    # Compute L2 norms with maximum precision
    l2_norm_v1 = compute_l2_norm(v1, tol)
    l2_norm_v2 = compute_l2_norm(v2, tol)
    
    # Avoid division by small values
    safe_norm_product = max(l2_norm_v1 * l2_norm_v2, EPSILON_SMALL)
    
    # Return normalized dot product
    return dot_prod / safe_norm_product 