from typing import Literal

import torch
import torch.nn as nn
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
        num_batch=None,
        hidden_dims: tuple[int] = (64,),
        bn: bool = False,
        add_batch_embedding: bool = False,
        n_cells: int | None = None,
    ):
        if add_batch_embedding and num_batch is None:
            raise ValueError(
                "num_batch must be provided "
                "when add_batch_embedding is True"
            )

        super(FullModel, self).__init__()
        self.add_batch_embedding = add_batch_embedding
        self.n_cells = n_cells

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

    def forward(
        self,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
        pos_domain_input: torch.Tensor | None = None,
        node_batch_indice: torch.Tensor | None = None,
        alpha: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
            pos_edges_emb[mask_pos] = (
                pos_edges_emb[mask_pos] + batch_embeddings
            )

            mask_neg = neg_edges < self.n_cells
            cell_nodes = neg_edges[mask_neg]
            batch_embeddings = self.batch_embedding(
                node_batch_indice[cell_nodes]
            )  # n x emb_dim
            neg_edges_emb[mask_neg] = (
                neg_edges_emb[mask_neg] + batch_embeddings
            )

        # 计算边得分
        pos_scores = torch.einsum(
            "ij,ij->i", pos_edges_emb[:, 0], pos_edges_emb[:, 1]
        )
        neg_scores = torch.einsum(
            "ijk,ijk->ij", neg_edges_emb[:, :, 0], neg_edges_emb[:, :, 1]
        )

        # 领域分类嵌入表示
        pos_domain_preds = None
        if pos_domain_input is not None:
            pos_domain_input = self.node_embedding(pos_domain_input)
            pos_domain_input = GradientReversalFunc.apply(
                pos_domain_input, alpha
            )
            pos_domain_preds = self.domain_classifier(pos_domain_input)
        return pos_scores, neg_scores, pos_domain_preds
