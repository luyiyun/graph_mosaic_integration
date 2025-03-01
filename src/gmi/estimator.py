from dataclasses import dataclass, asdict
from typing import Literal
import os
import os.path as osp
import json

import torch
import mudata as mu
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from .graph import MosaicDataGraph
from .model import GMIModel
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
    num_neg_per_pos: int = 4
    num_epochs: int = 100
    num_epochs_with_balanced_weights: int = 0
    batch_size: int = 131072
    val_split: float = 0.2  # 验证集比例
    # add_feature_net: bool = True
    neg_sample_in_batch: bool = False
    device: str = "cuda"
    optimizer: Literal["adam", "rmsprop"] = "adam"
    adversarial_batching_method: Literal["unique", "divide", "random"] = "divide"
    adversartial_balance_weights: bool = False
    disc_node_num_per_batch: int = 200
    neg_sampling_mode: Literal["full", "matched", "bipartitle"] = "matched"
    loss_type: Literal["margin_ranking", "weighted_softmax"] = "weighted_softmax"
    patience: int | float = 5  # inf or np.inf表示不使用早停
    random_seed: int = 0
    bilinear: bool = False
    num_cluster: int | None = None
    label_smoothing: float = 0.1
    w_grad_rev: float = 0.1
    w_loss_cls: float = 0.2
    w_loss_clu: float = 0.1
    w_infonce_temp: float = 1.0
    w_clu_temp: float = 1.0
    w_cov: float = 0.0
    w_dist: float = 0.0
    late_join_weights: dict[str, int] | None = None

    def __post_init__(self):
        self.late_join_weights = self.late_join_weights or {}

        weight_names = [
            "label_smoothing",
            "w_grad_rev",
            "w_loss_cls",
            "w_loss_clu",
            "w_infonce_temp",
            "w_clu_temp",
            "w_cov",
            "w_dist",
        ]
        assert all(k in weight_names for k in self.late_join_weights)
        self.weights = {}
        for k in weight_names:
            if k in self.late_join_weights:
                n_epochs_zero = self.late_join_weights[k]
                self.weights[k] = [0] * n_epochs_zero + [getattr(self, k)] * (
                    self.num_epochs - n_epochs_zero
                )
            else:
                self.weights[k] = getattr(self, k)

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

        self.graph = self.mdata2graph(
            mdata,
            batch_key=batch_key,
            log_norm=log_norm,
            feature_interaction_key=feature_interaction_key,
        )
        self.fit_graph(self.graph)

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
                expression_matrix = log_transform_and_normalize(expression_matrix)
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
            mapped_batches = nodes_df.loc[cell_indices[row_indices], "idx"].values
            mapped_features = nodes_df.loc[feature_indices[col_indices], "idx"].values

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
        main_edges_group = np.unique(nodes_df["group"].values[main_edges], axis=0)

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
            feat_edges_group = np.unique(nodes_df["group"].values[feat_edges], axis=0)
        else:
            feat_edges_df, feat_edges_group = None, None
        return MosaicDataGraph(
            # n_nodes=nodes_df.shape[0],
            # n_edges=nodes_df.shape[0],
            n_cells=mdata.shape[0],
            n_feats=mdata.shape[1],
            n_batch=mdata.obs[batch_key].unique().shape[0],
            group_categories=group_categories.tolist(),
            nodes_df=nodes_df,
            main_edges_df=main_edges_df,
            feat_edges_df=feat_edges_df,
            main_edge_groups=main_edges_group,
            feat_edge_groups=feat_edges_group,
        )

    def fit_graph(self, graph: MosaicDataGraph):
        self.model = GMIModel(
            num_nodes=graph.n_nodes,
            embedding_dim=self.embedding_dim,
            disc_hidden_dims=self.disc_hiddens,
            # num_batch=graph.n_batch,
            bn=self.disc_bn,
            bilinear=self.bilinear,
            num_cells=graph.n_cells,
            num_cluster=self.num_cluster,
            cell_batch_ids=torch.tensor(
                graph.nodes_df["group"].values[: graph.n_cells],
                device=self.device,
                dtype=torch.long,
            ),
            add_batch_embedding=self.add_batch_embedding,
            loss_type=self.loss_type,
        )

        # 初始化训练器
        self.trainer = Trainer(
            model=self.model,
            device=self.device,
            optimizer=self.optimizer,
            lr=self.learning_rate,
            neg_sample_in_batch=self.neg_sample_in_batch,
            adversarial_batching_method=self.adversarial_batching_method,
            batch_size=self.batch_size,
            disc_node_num_per_batch=self.disc_node_num_per_batch,
            neg_sampling_mode=self.neg_sampling_mode,
            loss_type=self.loss_type,
            patience=self.patience,
            random_seed=self.random_seed,
            # label_smoothing=self.label_smoothing,
            # grad_reverse_weight=alpha,
            # cls_loss_weight=loss_alpha,
            # clu_loss_weight=loss_clu_weight,
            # clu_loss_temp=loss_clu_temp,
        )

        self.trainer.train(
            graph=graph,
            num_neg_per_pos=self.num_neg_per_pos,
            num_epochs=self.num_epochs,
            val_split=self.val_split,
            disc_with_feature_nodes=False,
            **self.weights,
        )

        if not self.adversartial_balance_weights:
            return

        # # 训练 adversarial_training 后，再训练一次，使用平衡的权重
        # print("Estimate balance weights...")
        # balanced_weights = self.trainer.estimate_balance_weights()
        # graph.nodes_adversarial_weights = balanced_weights
        # # 重新构建新的训练流程
        # print("Retrain with balanced weights...")
        # self.trainer_balanced = Trainer(
        #     model=self.model,
        #     device=self.device,
        #     optimizer=self.optimizer,
        #     lr=self.learning_rate * 0.1,
        #     neg_sample_in_batch=self.neg_sample_in_batch,
        #     adversarial_training=self.adversarial_training,
        #     adversarial_batching_method=self.adversarial_batching_method,
        #     adversarial_with_feature_nodes=False,
        #     batch_size=self.batch_size,
        #     disc_node_num_per_batch=self.disc_node_num_per_batch,
        #     label_smoothing=self.label_smoothing,
        #     alpha=self.w_grad_rev,
        #     loss_alpha=self.w_loss_cls,
        #     neg_sampling_mode=self.neg_sampling_mode,
        #     loss_type=self.loss_type,
        #     late_join_alpha=0,
        #     late_join_loss_alpha=0,
        #     patience=self.patience,
        #     random_seed=self.random_seed,
        #     std_loss_alpha=self.std_loss_alpha,
        # )
        # self.trainer_balanced.train(
        #     graph=graph,
        #     num_neg_per_pos=self.num_neg_per_pos,
        #     num_epochs=self.num_epochs_with_balanced_weights,
        #     val_split=self.val_split,
        # )

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        model_path = os.path.join(path, "model.pth")
        torch.save(self.model.state_dict(), model_path)

        embed_df = pd.DataFrame(
            self.model.node_embedding.weight.detach().cpu().numpy(),
            index=self.graph.nodes_df.index,
        )
        embed_df.to_csv(osp.join(path, "final_embeddings.csv"))

        self.trainer.all_losses.to_csv(os.path.join(path, "all_losses.csv"))

        args = asdict(self)
        with open(osp.join(path, "args.json"), "w") as f:
            json.dump(args, f)

    @classmethod
    def load(cls, path: str):
        with open(osp.join(path, "args.json"), "r") as f:
            args = json.load(f)
        estimator = cls(**args)
        # TODO: model无法恢复
        # TODO: Trainer中的optimize, lr_scheduler(虽然现在还没有实现)等还需要恢复？
        # estimator.model.load_state_dict(
        #     torch.load(
        #         osp.join(path, "model.pth"), map_location=estimator.device
        #     )
        # )
        return estimator

    def plot_losses(self, fn: str):
        self.trainer.plot_losses(fn)

    @property
    def embeddings(self):
        if hasattr(self, "model"):
            return self.model.node_embedding.weight.detach()

        raise ValueError("Model is not fitted yet!")
