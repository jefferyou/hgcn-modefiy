import tensorflow as tf
from utils import util

# 定义常量，避免使用魔法数字
PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0
CLIP_VALUE = 0.98


class HyperbolicGainAggregator:
    """
    Hyperbolic Graph Attention Isomorphism Network (GAIN) 聚合器。

    结合了：
    - 双曲注意力机制
    - GAIN的自环增强
    - 用于多集合函数的MLP
    """

    def __init__(self, input_dim, output_dim, dropout=0.0, bias=True,
                 act=tf.nn.relu, name=None, concat=False, model_size="small"):
        """
        初始化Hyperbolic GAIN聚合器。

        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            dropout: Dropout率
            bias: 是否使用偏置
            act: 激活函数
            name: 层名称
            concat: 是否连接输出
            model_size: MLP模型大小 ("small" 或 "big")
        """
        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.name = name if name is not None else "hyp_gain_agg"

        # 为此聚合器创建变量作用域
        with tf.variable_scope(self.name):
            # 曲率参数（可训练）
            self.c = tf.Variable([1.0], trainable=True, name="curvature")

            # 创建转换权重
            self.transform_weights = tf.get_variable(
                'transform_weights',
                shape=[input_dim, output_dim],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            # 创建注意力权重
            self.attention_weights = tf.get_variable(
                'attention_weights',
                shape=[2 * output_dim, 1],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            # 创建自身和邻居特征的权重
            self.self_weights = tf.get_variable(
                'self_weights',
                shape=[input_dim, output_dim],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            self.neigh_weights = tf.get_variable(
                'neigh_weights',
                shape=[output_dim, output_dim],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            # 控制自重要性的epsilon参数（来自GAIN）
            self.epsilon = tf.Variable(0.5, name="epsilon", trainable=True)

            # 如果需要添加偏置
            if bias:
                self.bias_var = tf.get_variable(
                    'bias',
                    shape=[output_dim],
                    initializer=tf.zeros_initializer()
                )

        # 根据模型大小设置MLP维度
        if model_size == "small":
            self.hidden_dim = 128
        else:
            self.hidden_dim = 256

        # 创建MLP层
        with tf.variable_scope(self.name + '_mlp'):
            self.mlp_weights1 = tf.get_variable(
                'mlp_weights1',
                shape=[output_dim, self.hidden_dim],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            self.mlp_bias1 = tf.get_variable(
                'mlp_bias1',
                shape=[self.hidden_dim],
                initializer=tf.zeros_initializer()
            )

            self.mlp_weights2 = tf.get_variable(
                'mlp_weights2',
                shape=[self.hidden_dim, output_dim],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            self.mlp_bias2 = tf.get_variable(
                'mlp_bias2',
                shape=[output_dim],
                initializer=tf.zeros_initializer()
            )

        # 存储维度
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 存储变量以便于访问
        self.vars = {
            'transform_weights': self.transform_weights,
            'attention_weights': self.attention_weights,
            'self_weights': self.self_weights,
            'neigh_weights': self.neigh_weights,
            'mlp_weights1': self.mlp_weights1,
            'mlp_bias1': self.mlp_bias1,
            'mlp_weights2': self.mlp_weights2,
            'mlp_bias2': self.mlp_bias2,
            'c': self.c,
            'epsilon': self.epsilon
        }

        if self.bias:
            self.vars['bias'] = self.bias_var

    def __call__(self, inputs):
        """
        应用Hyperbolic GAIN聚合器。

        Args:
            inputs: (self_vecs, neigh_vecs)元组
                - self_vecs: 节点特征 [batch_size, input_dim]
                - neigh_vecs: 邻居特征 [batch_size, num_neighbors, input_dim]

        Returns:
            更新的节点嵌入 [batch_size, output_dim]
        """
        self_vecs, neigh_vecs = inputs

        # 应用dropout（如果需要）
        dropout_rate = self.dropout
        if dropout_rate > 0.0:
            self_vecs = tf.nn.dropout(self_vecs, rate=dropout_rate)
            neigh_vecs = tf.nn.dropout(neigh_vecs, rate=dropout_rate)

        # 将自身向量转换为双曲空间
        self_transformed = tf.matmul(self_vecs, self.transform_weights)
        self_hyperbolic = util.tf_my_prod_mat_exp_map_zero(self_transformed, self.c)

        # 获取重塑的形状信息
        neigh_shape = tf.shape(neigh_vecs)
        batch_size = neigh_shape[0]
        num_neighbors = neigh_shape[1]

        # 展平邻居向量进行转换
        neigh_flat = tf.reshape(neigh_vecs, [-1, self.input_dim])
        neigh_transformed = tf.matmul(neigh_flat, self.transform_weights)

        # 映射到双曲空间
        neigh_hyperbolic = util.tf_my_prod_mat_exp_map_zero(neigh_transformed, self.c)

        # 重塑回 [batch_size, num_neighbors, output_dim]
        neigh_hyperbolic = tf.reshape(neigh_hyperbolic, [batch_size, num_neighbors, self.output_dim])

        # --- 注意力机制 ---
        # 复制自身向量用于注意力
        self_tiled = tf.tile(tf.expand_dims(self_hyperbolic, 1), [1, num_neighbors, 1])

        # 连接自身和邻居向量用于注意力
        concat_features = tf.concat([self_tiled, neigh_hyperbolic], axis=2)
        concat_flat = tf.reshape(concat_features, [-1, 2 * self.output_dim])

        # 计算注意力分数
        attention_scores = tf.matmul(concat_flat, self.attention_weights)
        attention_scores = tf.nn.leaky_relu(attention_scores)
        attention_scores = tf.reshape(attention_scores, [batch_size, num_neighbors])

        # 应用softmax获取注意力权重
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        attention_weights = tf.expand_dims(attention_weights, 2)

        # 将注意力权重应用于邻居特征
        weighted_neighbors = neigh_hyperbolic * attention_weights

        # 聚合（求和）加权邻居特征
        aggregated_neighbors = tf.reduce_sum(weighted_neighbors, axis=1)

        # --- GAIN自环增强 ---
        # 为组合投影自身特征
        from_self = tf.matmul(self_vecs, self.self_weights)
        from_self = util.tf_my_prod_mat_exp_map_zero(from_self, self.c)

        # 投影邻居特征
        from_neighbors = tf.matmul(aggregated_neighbors, self.neigh_weights)

        # 组合增强自环（GAIN机制）
        combined = (1 + self.epsilon) * from_self + from_neighbors

        # --- 用于多集合函数的MLP ---
        # 映射到切空间进行MLP
        combined_tangent = util.tf_my_prod_mat_log_map_zero(combined, self.c)

        # 第一层
        hidden = tf.matmul(combined_tangent, self.mlp_weights1) + self.mlp_bias1
        hidden = tf.nn.leaky_relu(hidden)

        # 应用批归一化
        mean, variance = tf.nn.moments(hidden, axes=[0])
        hidden = tf.nn.batch_normalization(hidden, mean, variance, None, None, 1e-8)

        # 第二层
        output = tf.matmul(hidden, self.mlp_weights2) + self.mlp_bias2

        # 应用激活函数
        output = self.act(output)

        # 映射回双曲空间
        final_output = util.tf_my_prod_mat_exp_map_zero(output, self.c)

        return final_output


def create_hyperbolic_gain_aggregator(input_dim, output_dim, dropout=0.0,
                                      bias=True, act=tf.nn.relu, name=None,
                                      concat=False, model_size="small"):
    """
    创建Hyperbolic GAIN聚合器的工厂函数。

    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        dropout: Dropout率
        bias: 是否使用偏置
        act: 激活函数
        name: 层名称
        concat: 是否连接输出
        model_size: MLP模型大小 ("small" 或 "big")

    Returns:
        HyperbolicGainAggregator实例
    """
    return HyperbolicGainAggregator(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout=dropout,
        bias=bias,
        act=act,
        name=name,
        concat=concat,
        model_size=model_size
    )