
from collections import namedtuple, Counter
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F

import scipy.io as sio
import dgl
import torch
import numpy as np

import dgl
from dgl.data import (
    load_data, 
    TUDataset, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader

from sklearn.preprocessing import StandardScaler

import scipy.io as sio
GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset
}


def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def load_dataset(dataset_name):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)


def load_inductive_dataset(dataset_name):
    if dataset_name == "ppi":
        batch_size = 2
        # define loss function
        # create the dataset
        train_dataset = PPIDataset(mode='train')
        valid_dataset = PPIDataset(mode='valid')
        test_dataset = PPIDataset(mode='test')
        train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        eval_train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        g = train_dataset[0]
        num_classes = train_dataset.num_labels
        num_features = g.ndata['feat'].shape[1]
    else:
        _args = namedtuple("dt", "dataset")
        dt = _args(dataset_name)
        batch_size = 1
        dataset = load_data(dt)
        num_classes = dataset.num_classes

        g = dataset[0]
        num_features = g.ndata["feat"].shape[1]

        train_mask = g.ndata['train_mask']
        feat = g.ndata["feat"]
        feat = scale_feats(feat)
        g.ndata["feat"] = feat

        g = g.remove_self_loop()
        g = g.add_self_loop()

        train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
        train_g = dgl.node_subgraph(g, train_nid)
        train_dataloader = [train_g]
        valid_dataloader = [g]
        test_dataloader = valid_dataloader
        eval_train_dataloader = [train_g]
        
    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes



# def load_graph_classification_dataset(dataset_name, deg4feat=False, PE=True):
#     dataset_name = dataset_name.upper()
#     dataset = TUDataset(dataset_name)
#     graph, _ = dataset[0]
#     if "attr" not in graph.ndata:
#         if "node_labels" in graph.ndata and not deg4feat:
#             print("Use node label as node features")
#             feature_dim = 0
#             for g, _ in dataset:
#                 feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())
#
#             feature_dim += 1
#             x_attr = []
#             for g, l in dataset:
#                 node_label = g.ndata["node_labels"].view(-1)
#                 feat = F.one_hot(node_label, num_classes=feature_dim).float()
#                 g.ndata["attr"] = feat
#                 x_attr.append(feat)
#             x_attr = torch.cat(x_attr, dim=0).numpy()
#
#             scaler = StandardScaler()
#             scaler.fit(x_attr)
#             for g, l in dataset:
#                 g.ndata['attr'] = torch.from_numpy(scaler.transform(g.ndata['attr'])).float()
#
#
#
#         else:
#             print("Using degree as node features")
#             feature_dim = 0
#             degrees = []
#             for g, _ in dataset:
#                 feature_dim = max(feature_dim, g.in_degrees().max().item())
#                 degrees.extend(g.in_degrees().tolist())
#             MAX_DEGREES = 400
#
#             oversize = 0
#             for d, n in Counter(degrees).items():
#                 if d > MAX_DEGREES:
#                     oversize += n
#             # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
#             feature_dim = min(feature_dim, MAX_DEGREES)
#
#             feature_dim += 1
#             x_attr = []
#             for g, l in dataset:
#                 degrees = g.in_degrees()
#                 degrees[degrees > MAX_DEGREES] = MAX_DEGREES
#
#                 feat = F.one_hot(degrees, num_classes=feature_dim).float()
#                 g.ndata["attr"] = feat
#                 x_attr.append(feat)
#             x_attr = torch.cat(x_attr, dim=0).numpy()
#             scaler = StandardScaler()
#             scaler.fit(x_attr)
#             for g, l in dataset:
#                 g.ndata['attr'] = torch.from_numpy(scaler.transform(g.ndata['attr'])).float()
#     else:
#         print("******** Use `attr` as node features ********")
#         feature_dim = graph.ndata["attr"].shape[1]
#
#     labels = torch.tensor([x[1] for x in dataset])
#
#     num_classes = torch.max(labels).item() + 1
#     dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]
#
#     print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")
#
#     return dataset, (feature_dim, num_classes)


def load_graph_classification_dataset(fmri_filepath, labels_filepath, deg4feat=False):
    graphs = []
    labels = []

    # 读取标签数据
    label_data = sio.loadmat(labels_filepath)
    label_list = label_data['labels'].flatten()  # 假设标签数据是一个列向量，转换为一维数组

    # 读取图数据
    data = sio.loadmat(fmri_filepath)
    all_matrices = data['all_matrices']  # 460x1 的单元格数组

    # 确保标签数量与图数据数量匹配
    assert len(label_list) == all_matrices.shape[0], "Number of labels must match the number of graphs."

    for i in range(all_matrices.shape[0]):
        # 提取每个图的数据
        matrix = all_matrices[i][0]  # 每个单元格中的 90x90 矩阵

        # 创建 DGL 图
        g = dgl.graph(([], []), num_nodes=matrix.shape[0])

        # 添加边
        src, dst = np.nonzero(matrix)
        g.add_edges(src, dst)

        # 添加节点特征
        g.ndata['attr'] = torch.tensor(matrix, dtype=torch.float32)

        # 这里我们假设没有边特征矩阵，你可以根据需要添加边特征

        graphs.append(g)
        labels.append(torch.tensor(label_list[i], dtype=torch.long))  # 确保标签是长整型张量

    dataset = list(zip(graphs, labels))

    # 处理节点特征
    if not deg4feat:
        print("Processing node features")
        # 直接获取特征维度
        feature_dim = graphs[0].ndata['attr'].shape[1]
        scaler = StandardScaler()
        x_attr = [g.ndata['attr'].numpy() for g, _ in dataset]
        x_attr = np.concatenate(x_attr, axis=0)
        scaler.fit(x_attr)
        for g, _ in dataset:
            g.ndata['attr'] = torch.from_numpy(scaler.transform(g.ndata['attr'].numpy())).float()
    else:
        print("Using degree as node features")
        feature_dim = 0
        degrees = []
        for g, _ in dataset:
            feature_dim = max(feature_dim, g.in_degrees().max().item())
            degrees.extend(g.in_degrees().tolist())
        MAX_DEGREES = 400
        feature_dim = min(feature_dim, MAX_DEGREES) + 1

        x_attr = []
        for g, _ in dataset:
            degrees = g.in_degrees()
            degrees[degrees > MAX_DEGREES] = MAX_DEGREES
            feat = torch.nn.functional.one_hot(degrees, num_classes=feature_dim).float()
            g.ndata["attr"] = feat
            x_attr.append(feat)

        x_attr = torch.cat(x_attr, dim=0).numpy()
        scaler = StandardScaler()
        scaler.fit(x_attr)

        for g, _ in dataset:
            g.ndata['attr'] = torch.from_numpy(scaler.transform(g.ndata['attr'].numpy())).float()

    # 提取标签
    labels = torch.tensor(labels)
    num_classes = torch.max(labels).item() + 1

    # 处理数据集中的每个图，添加自环
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)


def get_mat_data(filepath):
    """辅助函数：自动获取 .mat 文件中最大的那个变量"""
    data = sio.loadmat(filepath)
    keys = [k for k in data.keys() if not k.startswith('__')]
    target_key = max(keys, key=lambda k: data[k].size)
    print(f"  -> Loaded '{target_key}' from {filepath}")
    return data[target_key]


def pad_to_90x90(matrix):
    """
    辅助函数：将任意尺寸小于90x90的矩阵填充到90x90
    """
    target_shape = (90, 90)
    if matrix.shape == target_shape:
        return matrix

    # 创建全零底板
    padded = np.zeros(target_shape, dtype=matrix.dtype)

    # 计算有效区域
    r = min(matrix.shape[0], 90)
    c = min(matrix.shape[1], 90)

    # 复制数据到左上角
    padded[:r, :c] = matrix[:r, :c]

    print(f"    [Warning] Padding matrix from {matrix.shape} to {target_shape}")
    return padded


def load_multimodal_graph_classification_dataset(
        raw_fmri_path,
        raw_dti_path,
        struct_fmri_path,
        struct_dti_path,
        part_fmri_path,
        part_dti_path,
        labels_filepath,
        deg4feat=False
):
    graphs_fmri = []
    graphs_dti = []
    labels = []

    print("-" * 30)
    print("开始加载数据集...")

    # --- 1. 加载标签 ---
    label_raw = get_mat_data(labels_filepath)
    label_list = label_raw.flatten()
    num_samples = len(label_list)

    # --- 2. 加载原始矩阵 (Feature) ---
    raw_fmri_all = get_mat_data(raw_fmri_path)
    raw_dti_all = get_mat_data(raw_dti_path)

    # --- 3. 加载 Mask 结构矩阵 (Structure) ---
    struct_fmri_all = get_mat_data(struct_fmri_path)
    struct_dti_all = get_mat_data(struct_dti_path)

    # --- 4. 加载社区 Partition ---
    part_fmri_all = get_mat_data(part_fmri_path)
    part_dti_all = get_mat_data(part_dti_path)

    print(f"检测到 {num_samples} 个样本，开始构建图...")

    for i in range(num_samples):
        # === A. 提取并修复单样本数据 ===

        # 1. 特征 (Raw) - 这里可能存在尺寸问题
        mat_f_attr = raw_fmri_all[i, 0] if raw_fmri_all.ndim > 1 else raw_fmri_all[i]
        mat_d_attr = raw_dti_all[i, 0] if raw_dti_all.ndim > 1 else raw_dti_all[i]

        # *** 关键修复：强制填充到 90x90 ***
        mat_f_attr = pad_to_90x90(mat_f_attr)
        mat_d_attr = pad_to_90x90(mat_d_attr)

        # 2. 结构 (Masked) - 这些已经是处理好的 90x90
        mat_f_struct = struct_fmri_all[i, 0]
        mat_d_struct = struct_dti_all[i, 0]

        # 3. 社区 ID
        part_f = part_fmri_all[i, 0].flatten()
        part_d = part_dti_all[i, 0].flatten()

        # === B. 构建 fMRI 图 ===
        g_fmri = dgl.graph(([], []), num_nodes=90)
        src_f, dst_f = np.nonzero(mat_f_struct)
        weights_f = mat_f_struct[src_f, dst_f]

        # 添加边
        g_fmri.add_edges(src_f, dst_f, data={'weight': torch.tensor(weights_f, dtype=torch.float32)})

        # 添加特征 (现在肯定是 90x90 了)
        g_fmri.ndata['attr'] = torch.tensor(mat_f_attr, dtype=torch.float32)
        g_fmri.ndata['community'] = torch.tensor(part_f, dtype=torch.long)

        # === C. 构建 DTI 图 ===
        g_dti = dgl.graph(([], []), num_nodes=90)
        src_d, dst_d = np.nonzero(mat_d_struct)
        weights_d = mat_d_struct[src_d, dst_d]

        # 添加边
        if len(src_d) > 0:
            g_dti.add_edges(src_d, dst_d, data={'weight': torch.tensor(weights_d, dtype=torch.float32)})
        else:
            print(f"Warning: Subject {i} DTI edges are empty. Adding self-loops.")
            g_dti.add_edges(range(90), range(90), data={'weight': torch.ones(90)})

        # 添加特征
        g_dti.ndata['attr'] = torch.tensor(mat_d_attr, dtype=torch.float32)
        g_dti.ndata['community'] = torch.tensor(part_d, dtype=torch.long)

        # 放入列表
        graphs_fmri.append(g_fmri)
        graphs_dti.append(g_dti)
        labels.append(label_list[i])

    # 打包
    dataset = list(zip(graphs_fmri, graphs_dti, labels))

    # 获取特征维度
    feat_dim_f = graphs_fmri[0].ndata['attr'].shape[1]
    feat_dim_d = graphs_dti[0].ndata['attr'].shape[1]

    # 标签转 Tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    num_classes = torch.max(labels_tensor).item() + 1

    # 加上自环
    dataset = [(g_f.remove_self_loop().add_self_loop(),
                g_d.remove_self_loop().add_self_loop(),
                lbl) for g_f, g_d, lbl in dataset]

    print(f"数据集加载完毕: {len(dataset)} 个样本, {num_classes} 类.")
    return dataset, (feat_dim_f, feat_dim_d, num_classes)