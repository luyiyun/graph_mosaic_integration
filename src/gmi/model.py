from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Function


def save_embeddings(node_embedding, embedding_path="final_embeddings.csv"):
    """
    保存最终的嵌入表示，使用 edge 文件中的索引
    """
    # 提取嵌入层的权重
    embeddings = node_embedding.weight.data.cpu().numpy()  # 这里暂时用了.cpu
    # 将嵌入表示转换为 DataFrame
    df = pd.DataFrame(
        embeddings,
        index=range(embeddings.shape[0]),  # 使用整数索引
        columns=[f"dim_{i}" for i in range(embeddings.shape[1])],
    )

    # 保存为 CSV 文件
    df.index.name = "node_index"
    df.to_csv(embedding_path)

    print(f"Final embeddings saved to {embedding_path}")


class GradientReversalFunc(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        alpha = torch.as_tensor(alpha).to(x.device)
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = -alpha * grad_output
        return grad_input, None


class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversal, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunc.apply(x, self.alpha)


class DomainClassifier(nn.Module):
    def __init__(
        self,
        n_inpt: int,
        n_out: int,
        hiddens: tuple[int],
        bn: bool = False,
        act: Literal["relu", "lrelu"] = "lrelu",
        dropout: float = 0.0,
    ):
        # TODO: 激活函数可改，dropout可以试试
        super().__init__()

        layers = []
        for i, o in zip([n_inpt] + list(hiddens[:-1]), hiddens):
            layers.append(nn.Linear(i, o))
            if bn:
                layers.append(nn.BatchNorm1d(o))
            layers.append(self.get_act(act))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hiddens[-1], n_out))
        self.net = nn.Sequential(*layers)  # TODO: 看文档学习

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_act(self, name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        elif name == "lrelu":
            return nn.LeakyReLU(0.2)
        else:
            raise NotImplementedError


class FullModel(nn.Module):
    def __init__(
        self,
        num_nodes,
        embedding_dim,
        num_batch: int | None = None,
        hidden_dims: tuple[int] = (64,),
        bn: bool = False,
        add_batch_embedding: bool = False,
        n_cells: int | None = None,
        bilinear: bool = False,
        num_cluster: int | None = None,
        loss_type: Literal["weighted_softmax", "margin_ranking"] = "weighted_softmax",
    ):
        if add_batch_embedding and num_batch is None:
            raise ValueError(
                "num_batch must be provided " "when add_batch_embedding is True"
            )
        assert loss_type in [
            "weighted_softmax",
            "margin_ranking",
        ], f"loss_type {loss_type} not supported"

        super(FullModel, self).__init__()
        self.add_batch_embedding = add_batch_embedding
        self.n_cells = n_cells
        self.bilinear = bilinear
        self.n_cluster = num_cluster
        self.loss_type = loss_type
        self.num_batch = num_batch

        # 嵌入层
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        if num_batch is not None:
            # 梯度反转层
            # self.grl = GradientReversal(alpha=alpha)
            # 领域分类器
            self.domain_classifier = DomainClassifier(
                embedding_dim, num_batch, hiddens=hidden_dims, bn=bn
            )
        else:
            print("initialized edge model")
        if add_batch_embedding:
            self.batch_embedding = nn.Embedding(num_batch, embedding_dim)
        if bilinear:
            self.relation_matrix = nn.Parameter(
                torch.randn(embedding_dim, embedding_dim) * 0.1
            )
        if self.n_cluster is not None:
            self.cluster_embedding = nn.Parameter(
                torch.randn(self.n_cluster, embedding_dim) * 0.1
            )

    def forward(
        self,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
        domain_input: torch.Tensor | None = None,
        domain_label: torch.Tensor | None = None,
        node_batch_indice: torch.Tensor | None = None,
        grad_reverse_weight: float = 1.0,
        label_smoothing: float = 0.0,
        pos_edges_weights: torch.Tensor | None = None,
        pred_sample_weights: torch.Tensor | None = None,
        info_nce_temp: float = 1.0,
        clu_loss_temp: float = 1.0,
        cls_loss_weight: float = 1.0,
        clu_loss_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        flag_calc_cls_loss = self.num_batch is not None and cls_loss_weight > 0.0
        flag_calc_clu_loss = self.n_cluster is not None and clu_loss_weight > 0.0
        if domain_input is None and (flag_calc_cls_loss or flag_calc_clu_loss):
            assert (
                domain_input is not None
            ), "domain_input must be provided when num_batch or n_cluster is not None"
        if domain_label is None and flag_calc_cls_loss:
            assert (
                domain_label is not None
            ), "domain_label must be provided when num_batch is not None"

        # 正样本和负样本嵌入
        # [n, 2, embedding_dim]
        pos_edges_emb = self.node_embedding(pos_edges)
        # [n, 4, 2, embedding_dim]
        neg_edges_emb = self.node_embedding(neg_edges)

        if self.add_batch_embedding:
            # 只给cell nodes加batch embedding
            mask_pos = pos_edges < self.n_cells
            cell_nodes = pos_edges[mask_pos]
            batch_embeddings = self.batch_embedding(
                node_batch_indice[cell_nodes]
            )  # n x emb_dim
            pos_edges_emb[mask_pos] = pos_edges_emb[mask_pos] + batch_embeddings

            mask_neg = neg_edges < self.n_cells
            cell_nodes = neg_edges[mask_neg]
            batch_embeddings = self.batch_embedding(
                node_batch_indice[cell_nodes]
            )  # n x emb_dim
            neg_edges_emb[mask_neg] = neg_edges_emb[mask_neg] + batch_embeddings

        # 计算边得分
        if self.bilinear:
            pos_scores = torch.einsum(
                "ij,ik,jk->i",
                pos_edges_emb[:, 0],
                pos_edges_emb[:, 1],
                self.relation_matrix,
            )
            neg_scores = torch.einsum(
                "ijk,ijl,kl->ij",
                neg_edges_emb[:, :, 0],
                neg_edges_emb[:, :, 1],
                self.relation_matrix,
            )
        else:
            pos_scores = torch.einsum(
                "ij,ij->i", pos_edges_emb[:, 0], pos_edges_emb[:, 1]
            )
            neg_scores = torch.einsum(
                "ijk,ijk->ij", neg_edges_emb[:, :, 0], neg_edges_emb[:, :, 1]
            )

        loss, loss_dict, others = 0.0, {}, {}
        # 计算正样本和负样本的损失
        if self.loss_type == "weighted_softmax":
            all_scores = (
                torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=1) / info_nce_temp
            )
            denominator = torch.logsumexp(all_scores, dim=1)
            softmax_loss = denominator - pos_scores
            if pos_edges_weights is not None:
                softmax_loss = softmax_loss * pos_edges_weights
            loss_edge = softmax_loss.mean()
        elif self.loss_type == "margin_ranking":
            pos_scores_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores)
            loss_edge = F.margin_ranking_loss(
                pos_scores_expanded, neg_scores, torch.ones_like(pos_scores)
            )
        loss += loss_edge
        loss_dict["edge"] = loss_edge

        if flag_calc_cls_loss or flag_calc_clu_loss:
            domain_embed_ori = self.node_embedding(domain_input)

        # 领域分类损失函数
        if flag_calc_cls_loss:
            domain_embed = GradientReversalFunc.apply(
                domain_embed_ori, grad_reverse_weight
            )
            domain_pred = self.domain_classifier(domain_embed)
            others["pred"] = domain_pred
            if pred_sample_weights is not None:
                loss_domain = F.cross_entropy(
                    domain_pred,
                    domain_label,
                    label_smoothing=label_smoothing,
                    reduction="none",
                )
                loss_domain = (
                    loss_domain * pred_sample_weights
                ).sum() / pred_sample_weights.sum()
            else:
                loss_domain = F.cross_entropy(
                    domain_pred, domain_label, label_smoothing=label_smoothing
                )
            loss += loss_domain * cls_loss_weight
            loss_dict["domain"] = loss_domain

        # 聚类损失函数
        if flag_calc_clu_loss:
            z_expand = domain_embed_ori.unsqueeze(1).expand(-1, self.n_cluster, -1)
            dist = (z_expand - self.cluster_embedding).pow(2).sum(dim=-1)
            # =================== DKM loss ===================
            soft_assign = torch.softmax(dist * -clu_loss_temp, dim=1)
            loss_cluster = (soft_assign * dist).sum(1).mean(0)
            # =================== DEC loss ===================
            # q = (1 + dist / clu_loss_temp).pow(-(clu_loss_temp + 1) / 2)
            # q = q / q.sum(dim=1, keepdim=True)
            # p = q.pow(2) / q.sum(dim=0, keepdim=True)  # NOTE: 样本量太大，导致p接近0？？
            # p = p / p.sum(dim=1, keepdim=True)
            # loss_cluster = F.kl_div(p.log(), q)
            # =================================================
            loss += loss_cluster * clu_loss_weight
            loss_dict["cluster"] = loss_cluster

        return loss, loss_dict, others
