from dataclasses import dataclass
from typing import Literal

import mudata as mu
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from .graph import MosaicDataGraph
from .model import FullModel
from .trainer import Trainer


def log_transform_and_normalize(matrix: csr_matrix | np.ndarray) -> csr_matrix:
    """对表达矩阵进行 log(X+1) 变换和 max-min 归一化"""
    if isinstance(matrix, csr_matrix):
        matrix = matrix.toarray()  # 转换为稠密矩阵以便进行操作
    elif not isinstance(matrix, np.ndarray):
        raise ValueError("matrix must be one of csr_matrix or ndarray!")
    matrix = np.log1p(matrix)  # log(X + 1) 变换
    # max-min 归一化
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    matrix = (matrix - min_val) / (
        max_val - min_val
    )  # x' = (x - min(x)) / (max(x) - min(x))
    return csr_matrix(matrix)


@dataclass
class GraphMosaicIntegration:
    embedding_dim: int = 50
    disc_hiddens: tuple[int] = (64,)
    disc_bn: bool = True
    add_batch_embedding: bool = False
    learning_rate: float = 0.01
    num_neg_per_pos: int = 5
    num_epochs: int = 100
    batch_size: int = 131072
    val_split: float = 0.2  # 验证集比例
    # add_feature_net: bool = True
    neg_sample_in_batch: bool = False
    device: str = "cuda"
    optimizer: Literal["adam", "rmsprop"] = "adam"
    adversarial_training: bool = True
    adversarial_batching_method: Literal["unique", "divide", "random"] = (
        "divide"
    )
    disc_node_num_per_batch: int = 200
    label_smoothing: float = 0.0
    alpha: float = 0.2
    loss_alpha: float = 0.2
    neg_sampling_mode: Literal["full", "matched", "bipartitle"] = "matched"
    loss_type: Literal["margin_ranking, weighted_softmax"] = "weighted_softmax"
    late_join_loss_alpha: int = 5
    late_join_alpha: int = 5
    patience: int | float = 5  # inf or np.inf表示不使用早停

    def fit(
        self,
        mdata: mu.MuData,
        batch_key: str | None,
        log_norm: bool = True,
        feature_interaction_key: str | None = None,
        result_key: str | None = None,
    ):
        if result_key is not None:
            raise NotImplementedError("result_key is not supported yet!")

        graph = self.mdata2graph(
            mdata,
            batch_key=batch_key,
            log_norm=log_norm,
            feature_interaction_key=feature_interaction_key,
        )
        self.fit_graph(graph)

    @classmethod
    def mdata2graph(
        cls,
        mdata: mu.MuData,
        batch_key: str | None,
        log_norm: bool = True,
        feature_interaction_key: str | None = None,
    ) -> MosaicDataGraph:
        # mdata -> graph

        # get all nodes
        indice = mdata.obs.index.tolist() + mdata.var.index.tolist()
        group_cell = (
            mdata.obs[batch_key].values
            if batch_key is not None
            else ["__cell__"] * mdata.shape[0]
        )
        group_feat = np.empty(mdata.shape[1], dtype=np.dtypes.StringDType())
        for k in mdata.mod:
            group_feat[mdata.varm[k]] = k
        group = np.concatenate(
            [group_cell.astype(np.dtypes.StringDType()), group_feat], axis=0
        )
        group = pd.Categorical(group)
        group_categories, group_code = group.categories, group.codes
        # transfer group to int
        nodes_df = pd.DataFrame(
            {
                "group": group_code,
                "type": ["cell"] * mdata.shape[0] + ["feat"] * mdata.shape[1],
                "idx": range(len(group)),
            },
            index=indice,
        )

        # get all edges
        main_edges_df = []
        for mod_name, adata_mod in mdata.mod.items():
            # 获取细胞和特征的全局索引
            cell_indices = adata_mod.obs.index
            feature_indices = adata_mod.var.index

            # 获取表达矩阵
            expression_matrix = adata_mod.X
            if log_norm:
                expression_matrix = log_transform_and_normalize(
                    expression_matrix
                )
            expression_matrix = coo_matrix(expression_matrix)
            row_indices, col_indices, expression_values = (
                expression_matrix.row,
                expression_matrix.col,
                expression_matrix.data,
            )

            # if isinstance(expression_matrix, csr_matrix):  # 稀疏矩阵
            #     row_indices, col_indices = expression_matrix.nonzero()
            #     expression_values = expression_matrix.data
            # else:  # 稠密矩阵
            #     expression_matrix = csr_matrix(expression_matrix)
            #     row_indices, col_indices = expression_matrix.nonzero()
            #     expression_values = expression_matrix.data

            # row_indices = np.asarray(row_indices).flatten()
            # col_indices = np.asarray(col_indices).flatten()
            # expression_values = np.asarray(expression_values).flatten()

            # 将细胞和特征映射到全局索引表中的序列号
            mapped_batches = nodes_df.loc[
                cell_indices[row_indices], "idx"
            ].values
            mapped_features = nodes_df.loc[
                feature_indices[col_indices], "idx"
            ].values

            # 创建临时 DataFrame
            edge_df = pd.DataFrame(
                {
                    "src": mapped_batches,
                    "dst": mapped_features,
                    "weight": expression_values,
                }
            )
            main_edges_df.append(edge_df)

        main_edges_df = pd.concat(main_edges_df, ignore_index=True)
        main_edges = main_edges_df[["src", "dst"]].values
        main_edges_group = np.unique(
            nodes_df["group"].values[main_edges], axis=0
        )

        if feature_interaction_key is not None:
            net = mdata.varp[feature_interaction_key]
            net = coo_matrix(net)
            row_indices, col_indices, expression_values = (
                net.row,
                net.col,
                net.data,
            )
            row_indices += mdata.shape[0]
            col_indices += mdata.shape[0]

            feat_edges_df = pd.DataFrame(
                {
                    "src": row_indices,
                    "dst": col_indices,
                    "weight": expression_values,
                }
            )
            feat_edges = feat_edges_df[["src", "dst"]].values
            feat_edges_group = np.unique(
                nodes_df["group"].values[feat_edges], axis=0
            )
        else:
            feat_edges_df, feat_edges_group = None, None

        return MosaicDataGraph(
            n_nodes=nodes_df.shape[0],
            n_edges=nodes_df.shape[0],
            n_cells=mdata.shape[0],
            n_feats=mdata.shape[1],
            n_batch=mdata.obs[batch_key].unique().shape[0],
            group_categories=group_categories.tolist(),
            nodes_df=nodes_df,
            main_edges_df=main_edges_df,
            feat_edges_df=feat_edges_df,
            edge_groups=main_edges_group,
            feat_edge_groups=feat_edges_group,
        )

    def fit_graph(self, graph: MosaicDataGraph):
        self.model = FullModel(
            num_nodes=graph.n_nodes,
            embedding_dim=self.embedding_dim,
            num_batch=graph.n_batch,
            hidden_dims=self.disc_hiddens,
            bn=self.disc_bn,
            add_batch_embedding=self.add_batch_embedding,
        )

        # 初始化训练器
        self.trainer = Trainer(
            model=self.model,
            device=self.device,
            optimizer=self.optimizer,
            lr=self.learning_rate,
            neg_sample_in_batch=self.neg_sample_in_batch,
            adversarial_training=self.adversarial_training,
            adversarial_batching_method=self.adversarial_batching_method,
            adversarial_with_feature_nodes=False,
            batch_size=self.batch_size,
            disc_node_num_per_batch=self.disc_node_num_per_batch,
            label_smoothing=self.label_smoothing,
            alpha=self.alpha,
            loss_alpha=self.loss_alpha,
            neg_sampling_mode=self.neg_sampling_mode,
            loss_type=self.loss_type,
            late_join_alpha=self.late_join_alpha,
            late_join_loss_alpha=self.late_join_loss_alpha,
            patience=self.patience,
        )

        self.trainer.train(
            graph=graph,
            num_neg_per_pos=self.num_neg_per_pos,
            num_epochs=self.num_epochs,
            val_split=self.val_split,
        )

    def save(self, path: str):
        self.trainer.save(path)

    def plot_losses(self, fn: str):
        self.trainer.plot_losses(fn)
