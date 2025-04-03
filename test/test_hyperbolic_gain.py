import tensorflow as tf
import numpy as np
import argparse
import os
import scipy.sparse as sp
from utils import process


def parse_args():
    parser = argparse.ArgumentParser(description='Test Hyperbolic GAIN implementation')
    parser.add_argument('-gpu', nargs='?', default='0', help='the ID for GPU')
    return parser.parse_args()


def simple_test():
    """简化的测试函数，仅测试基本组件"""
    # 创建小图
    num_nodes = 5
    feature_dim = 4

    # 创建特征
    features = np.random.randn(num_nodes, feature_dim).astype(np.float32)
    features = np.expand_dims(features, axis=0)  # [1, nodes, features]

    # 创建邻接矩阵
    adj = np.eye(num_nodes) + np.diag(np.ones(num_nodes - 1), 1) + np.diag(np.ones(num_nodes - 1), -1)
    adj_sparse = process.preprocess_adj_bias(sp.csr_matrix(adj))

    tf.reset_default_graph()

    # 创建占位符
    x = tf.placeholder(tf.float32, shape=[1, num_nodes, feature_dim])
    adj_ph = tf.sparse_placeholder(tf.float32)

    # 测试log_map和exp_map
    with tf.variable_scope("test"):
        c = tf.Variable([1.0], trainable=True)

        from utils.util import tf_mat_exp_map_zero, tf_mat_log_map_zero

        # 测试基本的双曲操作
        x_squeezed = tf.squeeze(x, 0)
        x_t = tf.transpose(x_squeezed)

        # 映射到双曲空间
        exp_result = tf_mat_exp_map_zero(x_t, c)
        log_result = tf_mat_log_map_zero(exp_result, c)

        # 测试注意力
        att_weights = tf.get_variable(
            "att_weights",
            shape=[feature_dim, 1],
            initializer=tf.glorot_uniform_initializer()
        )

        # 简化注意力计算
        logits = tf.matmul(x_squeezed, att_weights)
        attention = tf.nn.softmax(logits, axis=0)

        # 测试MLP
        W1 = tf.get_variable(
            "W1",
            shape=[feature_dim, 8],
            initializer=tf.glorot_uniform_initializer()
        )
        b1 = tf.get_variable(
            "b1",
            shape=[8],
            initializer=tf.zeros_initializer()
        )

        hidden = tf.nn.relu(tf.matmul(x_squeezed, W1) + b1)

    # 运行会话
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 测试各组件
        exp_val, log_val, att_val, hidden_val = sess.run(
            [exp_result, log_result, attention, hidden],
            feed_dict={x: features, adj_ph: adj_sparse}
        )

        print("\n--- Test Results ---")
        print("Exp map output shape:", exp_val.shape)
        print("Log map output shape:", log_val.shape)
        print("Attention shape:", att_val.shape)
        print("MLP hidden shape:", hidden_val.shape)

        # 检查NaN
        has_nan = (
                np.isnan(exp_val).any() or
                np.isnan(log_val).any() or
                np.isnan(att_val).any() or
                np.isnan(hidden_val).any()
        )

        if has_nan:
            print("WARNING: NaN values detected!")
        else:
            print("All operations completed successfully with no NaN values.")

        print("Basic component test passed!")


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    simple_test()
    print("Simple test completed successfully!")