import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from utils import util, process
import os


def test_step_by_step():
    """逐步测试模型组件"""
    # 基本设置
    tf.reset_default_graph()

    num_nodes = 8
    feature_dim = 6
    hidden_dim = 12
    num_classes = 4

    # 准备输入数据
    features = np.random.randn(1, num_nodes, feature_dim).astype(np.float32)
    adj = np.eye(num_nodes) + np.diag(np.ones(num_nodes - 1), 1) + np.diag(np.ones(num_nodes - 1), -1)

    # 创建稀疏张量输入
    adj_sparse = process.preprocess_adj_bias(sp.csr_matrix(adj))
    # 解包adj_sparse元组为indices, values, shape
    indices, values, shape = adj_sparse

    # 创建占位符
    x = tf.placeholder(tf.float32, shape=[1, num_nodes, feature_dim])
    adj_indices = tf.placeholder(tf.int64, shape=[None, 2])
    adj_values = tf.placeholder(tf.float32, shape=[None])
    adj_shape = tf.placeholder(tf.int64, shape=[2])
    is_training = tf.placeholder(tf.bool)

    # 创建稀疏张量
    adj_sparse_tensor = tf.SparseTensor(
        indices=adj_indices,
        values=adj_values,
        dense_shape=adj_shape
    )

    # 创建模型组件
    with tf.variable_scope("test_model"):
        # 1. 输入处理
        c = tf.Variable([1.0], trainable=True)
        x_squeezed = tf.squeeze(x, 0)
        x_t = tf.transpose(x_squeezed)

        # 2. 映射到双曲空间
        print("Mapping to hyperbolic space...")
        x_exp = tf.transpose(util.tf_mat_exp_map_zero(x_t, c))

        # 3. 创建简化的注意力
        print("Creating attention mechanism...")
        with tf.variable_scope("attention"):
            # 转换矩阵
            W = tf.get_variable(
                "transform",
                shape=[feature_dim, hidden_dim],
                initializer=tf.glorot_uniform_initializer()
            )

            # 转换特征
            transformed = tf.matmul(x_squeezed, W)

            # 注意力权重
            a = tf.get_variable(
                "attention_weights",
                shape=[2 * hidden_dim, 1],
                initializer=tf.glorot_uniform_initializer()
            )

            # 获取邻接信息 - 使用传入的稀疏张量
            indices_tensor = adj_sparse_tensor.indices

            # 收集特征
            from_nodes = tf.gather(transformed, indices_tensor[:, 0])
            to_nodes = tf.gather(transformed, indices_tensor[:, 1])

            # 计算注意力分数
            concat_features = tf.concat([from_nodes, to_nodes], axis=1)
            attn_logits = tf.matmul(concat_features, a)
            attn_logits = tf.nn.leaky_relu(attn_logits)

            # 稀疏softmax
            attn_sparse = tf.SparseTensor(
                indices=indices_tensor,
                values=tf.squeeze(attn_logits),
                dense_shape=adj_sparse_tensor.dense_shape
            )
            attn_coefs = tf.sparse_softmax(attn_sparse)

            # 应用注意力
            attended = tf.sparse_tensor_dense_matmul(attn_coefs, transformed)

        # 4. 应用MLP
        print("Applying MLP...")
        with tf.variable_scope("mlp"):
            # GAIN样式的自注意力
            epsilon = tf.Variable(0.5, name="epsilon")

            self_weights = tf.get_variable(
                "self_weights",
                shape=[hidden_dim, hidden_dim],
                initializer=tf.glorot_uniform_initializer()
            )

            from_self = tf.matmul(attended, self_weights)
            output = (1 + epsilon) * from_self

            # MLP层
            W1 = tf.get_variable(
                "W1",
                shape=[hidden_dim, hidden_dim],
                initializer=tf.glorot_uniform_initializer()
            )
            b1 = tf.get_variable(
                "b1",
                shape=[hidden_dim],
                initializer=tf.zeros_initializer()
            )

            hidden = tf.nn.leaky_relu(tf.matmul(output, W1) + b1)

            # 批量归一化
            mean, variance = tf.nn.moments(hidden, axes=[0])
            hidden = tf.cond(
                is_training,
                lambda: tf.nn.batch_normalization(hidden, mean, variance, None, None, 1e-8),
                lambda: hidden
            )

            # 输出层
            W_out = tf.get_variable(
                "W_out",
                shape=[hidden_dim, num_classes],
                initializer=tf.glorot_uniform_initializer()
            )
            b_out = tf.get_variable(
                "b_out",
                shape=[num_classes],
                initializer=tf.zeros_initializer()
            )

            logits = tf.matmul(hidden, W_out) + b_out

        # 5. 映射回双曲空间
        final_output = util.tf_my_prod_mat_exp_map_zero(hidden, c)

    # 运行会话
    print("Running session...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 测试各组件
        attended_val, hidden_val, logits_val, final_val = sess.run(
            [attended, hidden, logits, final_output],
            feed_dict={
                x: features,
                adj_indices: indices,
                adj_values: values,
                adj_shape: shape,
                is_training: True
            }
        )

        print("\n--- Results ---")
        print("Attended features shape:", attended_val.shape)
        print("Hidden layer shape:", hidden_val.shape)
        print("Logits shape:", logits_val.shape)
        print("Final output shape:", final_val.shape)

        # 检查NaN
        has_nan = (
                np.isnan(attended_val).any() or
                np.isnan(hidden_val).any() or
                np.isnan(logits_val).any() or
                np.isnan(final_val).any()
        )

        if has_nan:
            print("WARNING: NaN values detected!")
        else:
            print("All operations completed successfully with no NaN values.")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_step_by_step()
    print("Step-by-step test completed successfully!")