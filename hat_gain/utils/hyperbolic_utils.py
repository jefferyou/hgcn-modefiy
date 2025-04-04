import tensorflow as tf
import numpy as np

# Constants for numerical stability
PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0
CLIP_VALUE = 0.98

def project_hyperbolic_vector(x, c):
    """Project vector onto the Poincaré ball with radius 1/sqrt(c)."""
    # Projection operation to ensure vectors remain inside the hyperbolic space
    return tf.clip_by_norm(t=x, clip_norm=(1. - PROJ_EPS) / tf.sqrt(c), axes=[1])

def tf_atanh(x):
    """Stable implementation of arctanh."""
    # Clip values to avoid numerical instability at boundary
    x_clip = tf.clip_by_value(x, -CLIP_VALUE, CLIP_VALUE)
    return tf.atanh(x_clip)

def tf_tanh(x):
    """Stable implementation of tanh."""
    # Clip values to avoid excessively large arguments
    x_clip = tf.clip_by_value(x, -MAX_TANH_ARG, MAX_TANH_ARG)
    return tf.tanh(x_clip)

def mobius_addition(u, v, c):
    """
    Möbius addition in the Poincaré ball model.
    
    Args:
        u, v: Tensors of shape [batch_size, dim]
        c: Scalar curvature
        
    Returns:
        Tensor of shape [batch_size, dim]
    """
    # Add epsilon for numerical stability
    u = u + EPS
    v = v + EPS
    
    # Square norms
    u_norm_sq = tf.reduce_sum(tf.square(u), axis=1, keepdims=True)
    v_norm_sq = tf.reduce_sum(tf.square(v), axis=1, keepdims=True)
    
    # Compute the denominator term
    uv_dot = tf.reduce_sum(u * v, axis=1, keepdims=True)
    denominator = 1 + 2 * c * uv_dot + c * c * u_norm_sq * v_norm_sq
    
    # Compute numerator terms
    u_term = (1 + 2 * c * uv_dot + c * v_norm_sq) / denominator
    v_term = (1 - c * u_norm_sq) / denominator
    
    # Perform Möbius addition
    result = u_term * u + v_term * v
    return result

def mobius_scalar_multiplication(r, x, c):
    """
    Möbius scalar multiplication in the Poincaré ball model.
    
    Args:
        r: Scalar or tensor of scalars
        x: Tensor of shape [batch_size, dim]
        c: Scalar curvature
        
    Returns:
        Tensor of shape [batch_size, dim]
    """
    # Add epsilon for numerical stability
    x = x + EPS
    
    # Compute norm of x
    x_norm = tf.norm(x, axis=1, keepdims=True)
    
    # Return zero for zero inputs
    x_zero_mask = tf.cast(tf.equal(x_norm, 0), dtype=tf.float32)
    
    # Scale factor for non-zero inputs
    factor = tf.tanh(r * tf_atanh(tf.sqrt(c) * x_norm)) / (tf.sqrt(c) * x_norm + EPS)
    
    # Return scaled vector
    return factor * x * (1 - x_zero_mask) + x_zero_mask * x

def exponential_map(x, p, c):
    """
    Exponential map from tangent space at p to the Poincaré ball.
    
    Args:
        x: Tangent vector at p, shape [batch_size, dim]
        p: Base point in the Poincaré ball, shape [batch_size, dim]
        c: Scalar curvature
        
    Returns:
        Point on the manifold, shape [batch_size, dim]
    """
    # Add epsilon for numerical stability
    x = x + EPS
    
    # Norm of tangent vector
    x_norm = tf.norm(x, axis=1, keepdims=True)
    
    # Handle zero tangent vectors
    zero_mask = tf.cast(tf.less(x_norm, EPS), dtype=tf.float32)
    
    # Compute direction and scaling
    x_direction = x / (x_norm + EPS)
    scaling = tf.tanh(tf.sqrt(c) * x_norm / 2)
    
    # Compute exponential map
    exp_term = scaling * x_direction
    
    # Return p for zero tangent vectors, otherwise compute exp map
    return p * zero_mask + mobius_addition(p, exp_term, c) * (1 - zero_mask)

def exponential_map_zero(x, c):
    """
    Exponential map from tangent space at origin to the Poincaré ball.
    
    Args:
        x: Tangent vector at origin, shape [batch_size, dim]
        c: Scalar curvature
        
    Returns:
        Point on the manifold, shape [batch_size, dim]
    """
    # Add epsilon for numerical stability
    x = x + EPS
    
    # Compute norm of tangent vector
    x_norm = tf.norm(x, axis=1, keepdims=True)
    
    # Handle zero tangent vectors
    zero_mask = tf.cast(tf.less(x_norm, EPS), dtype=tf.float32)
    
    # Compute tanh scaling
    tanh_term = tf.tanh(tf.sqrt(c) * x_norm) / (tf.sqrt(c) * x_norm + EPS)
    
    # Scale the tangent vector
    result = tanh_term * x
    
    # Return zeros for zero inputs
    return result * (1 - zero_mask)

def logarithmic_map(x, p, c):
    """
    Logarithmic map from the Poincaré ball to tangent space at p.
    
    Args:
        x: Point on the manifold, shape [batch_size, dim]
        p: Base point in the Poincaré ball, shape [batch_size, dim]
        c: Scalar curvature
        
    Returns:
        Tangent vector at p, shape [batch_size, dim]
    """
    # Add epsilon for numerical stability
    x = x + EPS
    
    # Handle the case when x = p
    is_same = tf.reduce_all(tf.equal(x, p), axis=1, keepdims=True)
    
    # Möbius addition of -p and x
    neg_p = -p
    addition = mobius_addition(neg_p, x, c)
    
    # Compute norm of the addition result
    add_norm = tf.norm(addition, axis=1, keepdims=True)
    
    # Compute factor
    factor = tf_atanh(tf.sqrt(c) * add_norm) / (tf.sqrt(c) * add_norm + EPS)
    
    # Scale the addition result
    result = factor * addition
    
    # Return zero for same points
    return result * (1.0 - tf.cast(is_same, dtype=tf.float32))

def logarithmic_map_zero(x, c):
    """
    Logarithmic map from the Poincaré ball to tangent space at origin.
    
    Args:
        x: Point on the manifold, shape [batch_size, dim]
        c: Scalar curvature
        
    Returns:
        Tangent vector at origin, shape [batch_size, dim]
    """
    # Add epsilon for numerical stability
    x = x + EPS
    
    # Compute norm and ensure points are within the ball
    x_norm = tf.norm(x, axis=1, keepdims=True)
    x = tf.clip_by_norm(x, CLIP_VALUE / tf.sqrt(c), axes=1)
    x_norm = tf.minimum(x_norm, CLIP_VALUE / tf.sqrt(c))
    
    # Handle zero inputs
    zero_mask = tf.cast(tf.less(x_norm, EPS), dtype=tf.float32)
    
    # Compute scaling factor
    factor = tf_atanh(tf.sqrt(c) * x_norm) / (x_norm + EPS)
    
    # Scale input vector
    result = factor * x
    
    # Return zeros for zero inputs
    return result * (1.0 - zero_mask)

def hyperbolic_distance(x, y, c):
    """
    Compute hyperbolic distance between points in the Poincaré ball.
    
    Args:
        x, y: Tensors of shape [batch_size, dim]
        c: Scalar curvature
        
    Returns:
        Tensor of shape [batch_size, 1] with distances
    """
    # Add epsilon for numerical stability
    x = x + EPS
    y = y + EPS
    
    # Compute Möbius addition of -x and y
    neg_x = -x
    addition = mobius_addition(neg_x, y, c)
    
    # Compute norm of the result
    add_norm = tf.norm(addition, axis=1)
    
    # Compute hyperbolic distance
    dist = 2 / tf.sqrt(c) * tf_atanh(tf.sqrt(c) * add_norm)
    
    return dist

def matrix_exponential_map_zero(M, c):
    """
    Apply exponential map to each column of a matrix M from origin.
    
    Args:
        M: Matrix of shape [feature_dim, batch_size]
        c: Scalar curvature
        
    Returns:
        Matrix of shape [feature_dim, batch_size]
    """
    # Add epsilon for numerical stability
    M = M + EPS
    
    # Clip by norm for stability
    sqrt_c = tf.sqrt(c)
    M = tf.clip_by_norm(M, CLIP_VALUE / sqrt_c, axes=0)
    
    # Compute norms of each column
    norms = tf.norm(M, axis=0)
    
    # Compute tanh term
    tanh_term = tf.tanh(norms * sqrt_c)
    
    # Compute scaling coefficients
    coef = tanh_term / (norms + EPS) / sqrt_c
    
    # Apply scaling
    result = M * coef
    
    return result

def matrix_logarithmic_map_zero(M, c):
    """
    Apply logarithmic map to each column of a matrix M to origin.
    
    Args:
        M: Matrix of shape [feature_dim, batch_size]
        c: Scalar curvature
        
    Returns:
        Matrix of shape [feature_dim, batch_size]
    """
    # Add epsilon for numerical stability
    M = M + EPS
    
    # Clip by norm for stability
    sqrt_c = tf.sqrt(c)
    M = tf.clip_by_norm(M, CLIP_VALUE / sqrt_c, axes=0)
    
    # Compute norms of each column
    norms = tf.norm(M, axis=0)
    
    # Compute atanh term
    atanh_term = tf_atanh(sqrt_c * norms)
    
    # Compute scaling coefficients
    coef = atanh_term / (norms + EPS) / sqrt_c
    
    # Apply scaling
    result = M * coef
    
    return result
