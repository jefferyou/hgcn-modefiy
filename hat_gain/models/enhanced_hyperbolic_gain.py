import tensorflow as tf
import numpy as np
from utils import util
from models.base_hgattn import BaseHGAttN


class EnhancedHyperbolicGAIN(BaseHGAttN):
    """
    Enhanced Hyperbolic Graph Attention Isomorphism Network with
    explicit local and global sampling.
    """

    def __init__(self, placeholders, features, adj, degrees,
                 layer_infos, context_pairs=None, concat=True,
                 aggregator_type="gain", identity_dim=0, neg_sample_size=20, **kwargs):
        """初始化模型"""
        super(EnhancedHyperbolicGAIN, self).__init__(**kwargs)

        # 存储基本属性
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]
        self.adj_info = adj
        self.degrees = degrees
        self.concat = concat
        self.neg_sample_size = neg_sample_size

        # 处理特征
        if identity_dim > 0:
            self.embeds = tf.get_variable("node_embeddings",
                                          [adj.get_shape().as_list()[0], identity_dim])
        else:
            self.embeds = None

        # 检查特征是否为None (避免使用张量布尔值)
        features_is_none = (features is None)
        if features_is_none:
            if identity_dim == 0:
                raise Exception("Must provide features or positive identity dimension")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if self.embeds is not None:
                self.features = tf.concat([self.embeds, self.features], axis=1)

        # 设置维度
        if features_is_none:
            first_dim = identity_dim
        else:
            first_dim = features.shape[1] + identity_dim

        self.dims = [first_dim]
        for i in range(len(layer_infos)):
            self.dims.append(layer_infos[i].output_dim)

        # 存储上下文对
        self.context_pairs = context_pairs

        # 设置其他参数
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        # 设置优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005)

        # 构建模型
        self.build()

    def sample(self, inputs, layer_infos, batch_size=None):
        """采样邻居节点"""
        if batch_size is None:
            batch_size = self.batch_size

        samples = [inputs]

        # 从单个节点支持开始
        support_size = 1
        support_sizes = [support_size]

        # 为每层采样（反向顺序）
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1  # 从深层到浅层工作
            support_size *= layer_infos[t].num_samples

            # 使用该层的配置采样器
            sampler = layer_infos[t].neigh_sampler
            node = sampler((samples[k], layer_infos[t].num_samples))

            # 重塑以存储采样的节点ID
            samples.append(tf.reshape(node, [support_size * batch_size, ]))
            support_sizes.append(support_size)

        return samples, support_sizes

    def aggregate_neighbors(self, samples, input_features, dims, num_samples,
                            support_sizes, batch_size=None, aggregators=None):
        """聚合邻居信息"""
        if batch_size is None:
            batch_size = self.batch_size

        # 获取初始隐藏状态
        hidden = []
        for node_samples in samples:
            # 使用embedding_lookup获取特征
            node_features = tf.nn.embedding_lookup(input_features[0], node_samples)
            hidden.append(node_features)

        # 检查是否创建新的聚合器 (避免使用张量布尔值)
        create_new_aggregators = (aggregators is None)
        if create_new_aggregators:
            aggregators = []

        # 处理每一层
        for layer in range(len(num_samples)):
            if create_new_aggregators:
                # 设置该层的聚合器
                # 计算维度乘数 (避免使用张量布尔值)
                if layer == 0:
                    dim_mult = 1
                else:
                    dim_mult = 2 if self.concat else 1

                # 创建聚合器
                aggregator = self.create_aggregator(layer, dims, dim_mult)
                aggregators.append(aggregator)
            else:
                # 使用现有聚合器
                aggregator = aggregators[layer]

            # 处理该层的隐藏状态
            next_hidden = []

            # 聚合每个跳跃距离
            for hop in range(len(num_samples) - layer):
                # 准备维度
                if layer == 0:
                    dim_mult = 1
                else:
                    dim_mult = 2 if self.concat else 1

                # 重塑维度进行批处理
                neigh_dims = [batch_size * support_sizes[hop],
                              num_samples[len(num_samples) - hop - 1],
                              dim_mult * dims[layer]]

                # 自身特征维度
                self_dims = [batch_size * support_sizes[hop],
                             dim_mult * dims[layer]]

                # 运行聚合
                h = aggregator((tf.reshape(hidden[hop], self_dims),
                                tf.reshape(hidden[hop + 1], neigh_dims)))

                next_hidden.append(h)

            # 更新下一层的隐藏状态
            hidden = next_hidden

        # 返回最终隐藏状态和聚合器
        return hidden[0], aggregators

    def create_aggregator(self, layer, dims, dim_mult):
        """创建适当的聚合器"""
        # 导入聚合器实现
        from utils.hyperbolic_gain_layer import create_hyperbolic_gain_aggregator

        # 创建聚合器
        if layer == 0:
            # 输入层可能需要特殊处理
            return create_hyperbolic_gain_aggregator(
                dim_mult * dims[layer],
                dims[layer + 1],
                dropout=self.placeholders['dropout'],
                concat=self.concat
            )
        elif layer == len(dims) - 2:
            # 输出层
            return create_hyperbolic_gain_aggregator(
                dim_mult * dims[layer],
                dims[layer + 1],
                act=lambda x: x,  # 线性激活输出
                dropout=self.placeholders['dropout'],
                concat=self.concat
            )
        else:
            # 隐藏层
            return create_hyperbolic_gain_aggregator(
                dim_mult * dims[layer],
                dims[layer + 1],
                dropout=self.placeholders['dropout'],
                concat=self.concat
            )

    def _build(self):
        """构建模型图"""
        # 定义标签
        labels = tf.reshape(
            tf.cast(self.placeholders['batch2'], dtype=tf.int64),
            [self.batch_size, 1]
        )

        # 创建负样本
        self.neg_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=self.neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()
        )

        # 为每个批次中的节点采样邻居
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        samples2, support_sizes2 = self.sample(self.inputs2, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]

        # 映射到双曲空间并聚合
        with tf.variable_scope("hyperbolic_gain"):
            # 生成第一组节点的双曲嵌入
            self.outputs1, self.aggregators = self.aggregate_neighbors(
                samples1, [self.features], self.dims, num_samples,
                support_sizes1
            )

            # 生成第二组节点的双曲嵌入（共现）
            self.outputs2, _ = self.aggregate_neighbors(
                samples2, [self.features], self.dims, num_samples,
                support_sizes2, aggregators=self.aggregators
            )

            # 处理负采样
            neg_samples, neg_support_sizes = self.sample(
                self.neg_samples, self.layer_infos, self.neg_sample_size
            )

            self.neg_outputs, _ = self.aggregate_neighbors(
                neg_samples, [self.features], self.dims, num_samples,
                neg_support_sizes, batch_size=self.neg_sample_size,
                aggregators=self.aggregators
            )

        # 设置预测层
        from utils.prediction import HyperbolicEdgePredLayer

        # 计算维度乘数 (避免使用张量布尔值)
        dim_mult = 2 if self.concat else 1

        self.link_pred_layer = HyperbolicEdgePredLayer(
            dim_mult * self.dims[-1], dim_mult * self.dims[-1],
            self.placeholders, act=tf.nn.sigmoid
        )

        # 归一化输出
        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        self.outputs2 = tf.nn.l2_normalize(self.outputs2, 1)
        self.neg_outputs = tf.nn.l2_normalize(self.neg_outputs, 1)

    def build(self):
        """构建完整模型"""
        self._build()

        # 计算损失
        self._loss()
        self._accuracy()

        # 按批次大小归一化损失
        self.loss = self.loss / tf.cast(self.batch_size, tf.float32)

        # 设置带梯度裁剪的优化器
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grad = tf.clip_by_value(grad, -5.0, 5.0)
                clipped_grads_and_vars.append((clipped_grad, var))
            else:
                clipped_grads_and_vars.append((None, var))

        if len(clipped_grads_and_vars) > 0:
            self.grad, _ = clipped_grads_and_vars[0]
            self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        else:
            self.grad = None
            self.opt_op = None

    def _loss(self):
        """定义模型损失函数"""
        # 初始化损失
        self.loss = 0.0

        # 应用L2正则化
        for aggregator in self.aggregators:
            for var_name, var in aggregator.vars.items():
                self.loss += tf.nn.l2_loss(var) * self.placeholders['weight_decay']

        # 链接预测损失
        self.loss += self.link_pred_layer.loss(self.outputs1, self.outputs2, self.neg_outputs)
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        """定义准确率指标"""
        # 计算正对之间的亲和力
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)

        # 计算与负样本的亲和力
        self.neg_aff = self.link_pred_layer.neg_cost(self.outputs1, self.neg_outputs)
        self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, self.neg_sample_size])

        # 合并进行排名计算
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])

        # 计算排名
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)

        # 计算平均倒数排名
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)