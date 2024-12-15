import torch

def add_type_columns_to_edges(edges, nodes_df):
    unique_types = nodes_df["type"].unique()
    #import ipdb; ipdb.set_trace()
    type_to_int = {t: i for i, t in enumerate(unique_types)}

    # 将节点类型映射转换为 Tensor
    node_idx_to_type = torch.zeros(nodes_df["idx"].max() + 1, dtype=torch.long)
    for idx, t in zip(nodes_df["idx"], nodes_df["type"]):
        node_idx_to_type[idx] = type_to_int[t]
    # 更新nodes文件
    nodes_df["type"] = nodes_df["type"].map(type_to_int)
    # 提取边的类型
    batch_types = node_idx_to_type[edges[:, 0]]
    feature_types = node_idx_to_type[edges[:, 1]]

    # 添加类型列到边张量
    edges = torch.cat(
        [edges, batch_types.unsqueeze(1), feature_types.unsqueeze(1)], dim=1
    )

    return edges, nodes_df

# def get_intersecting_edges(edges):
#     """
#     获取 batch_types 和 feature_types 的所有不重复组合，
#     并构建起始点和目标点的映射。
#     """
#     # 提取 batch_types 和 feature_types
#     batch_types = edges[:, 2]
#     feature_types = edges[:, 3]

#     # 构建 batch_type 和节点的映射
#     batch_to_nodes = {}
#     feature_to_nodes = {}

#     # 使用 Tensor 实现映射
#     for batch_type in torch.unique(batch_types):
#         mask = batch_types == batch_type
#         batch_to_nodes[batch_type.item()] = torch.unique(edges[mask, 0])

#     for feature_type in torch.unique(feature_types):
#         mask = feature_types == feature_type
#         feature_to_nodes[feature_type.item()] = torch.unique(edges[mask, 1])

#     return batch_to_nodes, feature_to_nodes