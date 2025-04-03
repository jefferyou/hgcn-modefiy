import tensorflow as tf
import numpy as np

# 常量定义
PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0
CLIP_VALUE = 0.98


def tf_project_hyp_vecs(x, c):
    """
    投影操作。确保双曲嵌入在单位球内。

    Args:
        x: 输入向量
        c: 曲率参数

    Returns:
        投影后的向量
    """
    return tf.clip_by_norm(t=x, clip_norm=(1. - PROJ_EPS) / tf.sqrt(c), axes=[1])


def tf_atanh(x):
    """
    双曲反正切函数。

    Args:
        x: 输入值（标量，非向量）

    Returns:
        反双曲正切值
    """
    return tf.atanh(tf.minimum(x, 1. - EPS))  # 仅对正实数x有效


def tf_tanh(x):
    """
    双曲正切函数。

    Args:
        x: 输入值（标量，非向量）

    Returns:
        双曲正切值
    """
    return tf.tanh(tf.minimum(tf.maximum(x, -MAX_TANH_ARG), MAX_TANH_ARG))


def tf_my_prod_mob_addition(u, v, c):
    """
    双曲空间中的Möbius加法。

    Args:
        u: 第一个向量 [nodes, features]
        v: 第二个向量 [nodes, features]
        c: 曲率参数

    Returns:
        u ⊕ v
    """
    # 输入 [nodes, features]
    norm_u_sq = tf.reduce_sum(u * u, axis=1, keepdims=True)
    norm_v_sq = tf.reduce_sum(v * v, axis=1, keepdims=True)
    uv_dot_times = 4 * tf.reduce_sum(u * v, axis=1, keepdims=True) * c
    denominator = 1 + uv_dot_times + norm_u_sq * norm_v_sq * c * c
    coef_1 = (1 + uv_dot_times + c * norm_v_sq) / denominator
    coef_2 = (1 - c * norm_u_sq) / denominator
    return coef_1 * u + coef_2 * v


def tf_my_prod_mat_log_map_zero(M, c):
    """
    从原点开始的对数图。

    Args:
        M: 输入矩阵
        c: 曲率参数

    Returns:
        对数映射结果
    """
    sqrt_c = tf.sqrt(c)
    # M = tf.transpose(M)
    M = M + EPS
    M = tf.clip_by_norm(M, clip_norm=CLIP_VALUE / sqrt_c, axes=0)
    m_norm = tf.norm(M, axis=0)
    atan_norm = tf.atanh(tf.clip_by_value(m_norm * sqrt_c, clip_value_min=-0.9, clip_value_max=0.9))
    M_cof = atan_norm / (m_norm * sqrt_c + EPS)
    res = M * M_cof
    return res


def tf_my_prod_mat_exp_map_zero(vecs, c):
    """
    从原点开始的指数图。

    Args:
        vecs: 输入向量
        c: 曲率参数

    Returns:
        指数映射结果
    """
    sqrt_c = tf.sqrt(c)
    vecs = vecs + EPS
    vecs = tf.transpose(vecs)
    vecs = tf.clip_by_norm(vecs, clip_norm=CLIP_VALUE / sqrt_c, axes=0)
    norms = tf.norm(vecs, axis=0)
    c_tanh = tf.tanh(norms * sqrt_c)
    coef = c_tanh / (norms * sqrt_c + EPS)
    res = vecs * coef
    return tf.transpose(res)


def tf_my_mobius_list_distance(mat_x, mat_y, c):
    """
    计算两组向量之间的Möbius距离。

    Args:
        mat_x: 第一组向量 [nodes, features]
        mat_y: 第二组向量 [nodes, features]
        c: 曲率参数

    Returns:
        两组向量之间的Möbius距离
    """
    # 输入：[nodes features]
    mat_add = tf_my_prod_mob_addition(-mat_x, mat_y, c)
    sqrt_c = tf.sqrt(c)
    res_norm = tf.norm(mat_add, axis=1)
    res = tf.atanh(tf.clip_by_value(sqrt_c * res_norm, clip_value_min=1e-8, clip_value_max=CLIP_VALUE))
    return 2 / sqrt_c * res


def tf_mat_exp_map_zero(M, c=1.):
    """
    从原点开始的矩阵指数图（更简单的版本）。

    Args:
        M: 输入矩阵
        c: 曲率参数

    Returns:
        指数映射结果
    """
    M = M + EPS
    sqrt_c = tf.sqrt(c)
    M = tf.clip_by_norm(M, clip_norm=CLIP_VALUE / sqrt_c, axes=0)
    norms = tf.norm(M, axis=0)
    c_tanh = tf.tanh(norms * sqrt_c)
    coef = c_tanh / (norms * sqrt_c + EPS)
    res = M * coef
    return res


def tf_mat_log_map_zero(M, c=1):
    """
    从原点开始的矩阵对数图（更简单的版本）。

    Args:
        M: 输入矩阵
        c: 曲率参数

    Returns:
        对数映射结果
    """
    M = M + EPS
    M = tf.clip_by_norm(M, clip_norm=CLIP_VALUE, axes=0)
    m_norm = tf.norm(M, axis=0)
    atan_norm = tf_atanh(m_norm)
    M_cof = atan_norm / (m_norm + EPS)
    res = M * M_cof
    return res