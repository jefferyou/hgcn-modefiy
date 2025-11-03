import argparse
import tensorflow as tf
import numpy as np
import json
import scipy.sparse as sp
import networkx as nx
from networkx.readwrite import json_graph
from sklearn.preprocessing import StandardScaler
from saved_models import SpHyperbolicGAIN  # 与训练时一致的模型定义
from utils import process  # 与训练时一致的工具函数
from sklearn.metrics import f1_score
import os




def test(city,save_value,path_suffix=''):

    def load_model_metadata(metadata_path):
        """读取模型元数据（节点数、特征维度等关键参数）"""
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata

    # 1. 配置路径（需替换为你的实际路径）
    model_dir = f"saved_models/{city}"  # 模型保存目录
    dataset='osm_inductive' if city=='italy' else 'osm_transductive'
    metadata_path = os.path.join(model_dir, f"hyperbolic_gain_{dataset}_{city}-osm_{path_suffix}_metadata.json")  # 元数据路径
    ckpt_path = os.path.join(model_dir, f"hyperbolic_gain_{dataset}_{city}-osm_{path_suffix}")  # .ckpt文件路径（不含后缀）
    data_dir = "graph_data/osm_inductive" if city=='italy' else "graph_data/osm_transductive"  # 测试数据目录
    prefix = f"{city}-osm"  # 测试数据前缀

    # 2. 读取元数据
    metadata = load_model_metadata(metadata_path)
    nb_nodes = metadata["nb_nodes"]
    ft_size = metadata["ft_size"]
    nb_classes = metadata["nb_classes"]
    c = metadata["curvature"]
    hid_units = [metadata["units"]]
    n_heads = [metadata["heads"], 1]
    model_size = metadata["model_size"]
    use_global_walks = metadata["use_global_walks"]
    fusion_type = metadata["fusion_type"]


    def load_test_data(data_dir, prefix, metadata):
        """加载测试数据，流程与训练时一致"""
        # 加载图、特征、ID映射、类别映射（同训练代码）
        G_data = json.load(open(os.path.join(data_dir, f"{prefix}-G.json")))
        G = json_graph.node_link_graph(G_data)
        features = np.load(os.path.join(data_dir, f"{prefix}-feats.npy"))
        id_map = json.load(open(os.path.join(data_dir, f"{prefix}-id_map.json")))
        class_map = json.load(open(os.path.join(data_dir, f"{prefix}-class_map.json")))

        # 处理ID映射和类别映射（同训练代码）
        id_map = {int(k): int(v) for k, v in id_map.items()} if isinstance(next(iter(id_map.keys())), str) else id_map
        is_multilabel = isinstance(next(iter(class_map.values())), list)
        class_map = {int(k) if isinstance(k, str) else k: (v if is_multilabel else int(v)) for k, v in class_map.items()}

        # 创建测试掩码（仅保留test=True的节点）
        test_mask = []
        for _, data in G.nodes(data=True):
            test_mask.append(data.get('test', False))
        test_mask = np.array(test_mask)

        # 处理标签（同训练代码）
        nodes = list(G.nodes())
        labels = np.zeros((len(nodes), nb_classes))
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        for node_id, class_id in class_map.items():
            if node_id in node_to_idx:
                idx = node_to_idx[node_id]
                labels[idx] = class_id if is_multilabel else [1 if i == class_id else 0 for i in range(nb_classes)]
        y_test = np.zeros(labels.shape)
        y_test[test_mask] = labels[test_mask]

        # 标准化特征（使用训练时的均值/方差，此处简化为重新拟合训练集，实际应保存训练时的scaler）
        train_mask = np.array([not data.get('val', False) and not data.get('test', False) for _, data in G.nodes(data=True)])
        train_features = features[train_mask]
        scaler = StandardScaler()
        scaler.fit(train_features)
        features = scaler.transform(features)

        # 处理邻接矩阵（同训练代码）
        adj = nx.adjacency_matrix(G)
        biases = process.preprocess_adj_bias(adj)  # 稀疏矩阵处理

        # 扩展维度以匹配模型输入（batch_size=1）
        features = features[np.newaxis]
        y_test = y_test[np.newaxis]
        test_mask = test_mask[np.newaxis]

        return biases, features, y_test, test_mask

    # 加载测试数据
    biases, features, y_test, test_mask = load_test_data(data_dir, prefix, metadata)

    def load_and_test_model(ckpt_path, biases, features, y_test, test_mask):
        """重建图结构、加载模型并执行测试"""
        # 1. 配置GPU（同训练代码）
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 替换为你的GPU ID
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # 2. 重建计算图（与训练时完全一致）
        with tf.Graph().as_default():
            # 输入占位符（需与训练时的名称、形状完全匹配）
            ftr_in = tf.placeholder(dtype=tf.float32, shape=(1, nb_nodes, ft_size))  # batch_size=1
            bias_in = tf.sparse_placeholder(dtype=tf.float32)  # 稀疏邻接矩阵
            lbl_in = tf.placeholder(dtype=tf.int32, shape=(1, nb_nodes, nb_classes))
            msk_in = tf.placeholder(dtype=tf.int32, shape=(1, nb_nodes))
            attn_drop = tf.placeholder(dtype=tf.float32, shape=())
            ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
            is_train = tf.placeholder(dtype=tf.bool, shape=())

            # 重建模型（与训练时的inference调用参数完全一致）
            logits, _, _ = SpHyperbolicGAIN.inference(
                ftr_in, nb_classes, nb_nodes, is_train,
                attn_drop, ffd_drop,
                bias_mat=bias_in,
                hid_units=hid_units, n_heads=n_heads,
                activation=tf.nn.elu, c=c,
                model_size=model_size,
                use_global_walks=(use_global_walks == 1),
                fusion_type=fusion_type
            )

            # 定义测试指标计算（同训练时的测试逻辑）
            log_resh = tf.reshape(logits, [-1, nb_classes])
            lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
            msk_resh = tf.reshape(msk_in, [-1])
            loss = SpHyperbolicGAIN.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
            accuracy = SpHyperbolicGAIN.masked_accuracy(log_resh, lab_resh, msk_resh)
            pred_all = tf.cast(tf.argmax(log_resh, 1), dtype=tf.int32)
            real_all = tf.cast(tf.argmax(lab_resh, 1), dtype=tf.int32)

            # 3. 恢复模型参数
            saver = tf.train.Saver()  # 需与训练时的变量集合一致（默认所有变量）
            with tf.Session(config=config) as sess:
                # 恢复参数（无需初始化，restore会覆盖初始化）
                saver.restore(sess, ckpt_path)
                print(f"模型已从 {ckpt_path} 加载成功")

                # 4. 执行测试（关闭dropout）
                test_loss, test_acc, real_y, pred_y = sess.run(
                    [loss, accuracy, real_all, pred_all],
                    feed_dict={
                        ftr_in: features,
                        bias_in: biases,  # 稀疏矩阵需传入tuple格式（indices, values, dense_shape）
                        lbl_in: y_test,
                        msk_in: test_mask,
                        is_train: False,  # 测试模式（禁用batch norm更新等）
                        attn_drop: 0.0,  # 关闭dropout
                        ffd_drop: 0.0
                    }
                )

                # 5. 计算F1分数（过滤非测试节点）
                mask = test_mask[0]  # 去除batch维度
                #print(np.where(mask))
                y_true = real_y[mask]
                y_pred = pred_y[mask]

                macro_f1 = f1_score(y_true, y_pred, average='macro')
                micro_f1 = f1_score(y_true, y_pred, average='micro')
                weight_f1=f1_score(y_true, y_pred, average='weighted')


                if save_value:
                    predict_result_dict = dict(zip(np.where(mask)[0].astype('int32'), y_pred.astype('int32')))
                    true_value_dict = dict(zip(np.where(mask)[0].astype('int32'), y_true.astype('int32')))
                    predict_result_dict = {str(k): int(v) for k, v in predict_result_dict.items()}
                    true_value_dict = {str(k): int(v) for k, v in true_value_dict.items()}
                    json.dump(predict_result_dict, open(f'saved_models/{city}/{city}_pred_result.json', "w", encoding="utf-8"))
                    json.dump(true_value_dict, open(f'saved_models/{city}/{city}_true_value.json', "w", encoding="utf-8"))



                # 输出测试结果
                print("=" * 50)
                print("测试结果：")
                print(f"测试损失: {test_loss:.5f}")
                print(f"测试准确率: {test_acc:.5f}")
                print(f"Macro F1: {macro_f1:.5f}")
                print(f"Micro F1: {micro_f1:.5f}")
                print(f"Weighted F1: {weight_f1:.5f}")
                print("=" * 50)

                return test_loss, test_acc, macro_f1, micro_f1

    # 执行测试
    test_loss, test_acc, macro_f1, micro_f1 = load_and_test_model(ckpt_path, biases, features, y_test, test_mask)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default='venice', help='the city for test')
    parser.add_argument('--save_value', default=False, help='whether save the value')
    parser.add_argument('--path_suffix', default='', help='the path suffix')
    args = parser.parse_args()
    args.path_suffix='20251101-101659'
    test(args.city,args.save_value,args.path_suffix)




