import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pandas as pd
import torch
from time import perf_counter
from collections import Counter

from .negative_sample import NegativeSampler
from .loss import LossType
from .comb_edge import add_type_columns_to_edges
from .model import save_embeddings

class GMI:
    def __init__(
        self,
        nodes,
        num_nodes,
        num_cell,  # cell_node的idx数量
        num_batch,  # batch的综述
        num_edges,
        embedding_dim,
        num_neg_samples,
        batch_size,
        lr,
        loss_type="margin_ranking",
        device="gpu",
        add_batch_embedding=False,
        add_feature_net=True,
        alpha=1.0,
    ):

        self.device = torch.device(device)
        self.nodes = nodes
        self.num_nodes = num_nodes
        self.num_cell = num_cell
        self.num_edges = num_edges
        self.embedding_dim = embedding_dim   
        self.num_neg_per_pos = num_neg_samples
        self.batch_size = batch_size

        #创建嵌入层
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim).to(
            self.device # 嵌入层
        )  # Class 新创建一个CLASS，
        param_groups = [
            {"params": self.node_embedding.parameters(), "lr": lr},
        ] 

        if add_batch_embedding:
            self.batch_embedding = nn.Embedding(num_batch, embedding_dim).to(
                self.device
            )
            param_groups.append({"params": self.batch_embedding.parameters(), "lr": lr})

        # 将两个嵌入层的参数都传递给优化器
        self.optimizer = optim.Adam(param_groups)
        
        #创建损失函数
        # self.optimizer = optim.Adam(self.node_embedding.parameters(), lr=lr) #放到adamn #放到nn.momodel形成一个新的class
        self.loss_type = loss_type
        self.loss_fn = LossType(loss_type=loss_type).to(self.device)


        self.train_losses = []
        self.eval_losses = []

        self.add_batch_embedding = add_batch_embedding
        self.add_feature_net = add_feature_net

        self.global_positive_edges = set()

    def compute_loss(
        self,
        pos_edges,
        neg_edges,
        edge_weights=None,
    ):
        if pos_edges.shape[0] != neg_edges.shape[0]:
            raise ValueError("pos and neg edges shape not matched")
        if edge_weights is not None:
            if pos_edges.shape[0] != edge_weights.shape[0]:
                raise ValueError("pos edges and edge weights shape not matched")

        # 获取正样本的嵌入
        pos_edges_emb = self.node_embedding(pos_edges)  # n*2*50
        # 获取负样本的嵌入
        neg_edges_emb = self.node_embedding(neg_edges)  # n*2*4*50

        if self.add_batch_embedding:
            node_type = torch.tensor(
                self.nodes["type"].values, dtype=torch.long, device=self.device
            )
            pos_batch = node_type[pos_edges[:, 0]]
            pos_batch_emb = self.batch_embedding(pos_batch)
            pos_edges_emb[:, 0, :] += pos_batch_emb
            neg_batch = node_type[neg_edges[:, :, 0]]
            neg_batch_emb = self.batch_embedding(neg_batch)
            neg_edges_emb[:, :, 0] += neg_batch_emb

        # 计算正样本对的点积得分
        pos_scores = (pos_edges_emb[:, 0, :] * pos_edges_emb[:, 1, :]).sum(dim=-1)
        # 计算负样本对的点积得分
        neg_scores = (neg_edges_emb[:, 0, :, :] * neg_edges_emb[:, 1, :, :]).sum(dim=-1)

        # 损失函数
        if self.loss_fn.loss_type == "weighted_softmax":
            return self.loss_fn(pos_scores, neg_scores, edge_weights)
        else:
            return self.loss_fn(pos_scores, neg_scores)

    def train_epoch(
        self,
        pos_edges,
        neg_edges,
        edge_weights,
    ):
        epoch_loss = 0.0

        # 处理每个 batch
        for i in tqdm(
            range(0, pos_edges.shape[0], self.batch_size), desc="Batch: ", leave=False
        ):
            # 获取当前批次的正样本
            batch_pos_edges = pos_edges[i : i + self.batch_size]
            batch_neg_edges = neg_edges[i : i + self.batch_size]
            batch_edge_weights = (
                edge_weights[i : i + self.batch_size]
                if self.loss_fn.loss_type == "weighted_softmax"
                else None
            )

            # 计算当前批次的损失
            loss = self.compute_loss(
                batch_pos_edges,
                batch_neg_edges,
                batch_edge_weights,
            )

            # 反向传播与优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 累计当前批次的损失
            epoch_loss += loss.item()

        return epoch_loss

    def evaluate(
        self,
        pos_edges,
        neg_edges,
        edge_weights,
    ):
        self.node_embedding.eval()

        eval_loss = 0.0

        # 开始评估
        with torch.no_grad():
            for i in range(0, pos_edges.shape[0], self.batch_size):
                # 获取当前批次的正样本
                batch_pos_edges = pos_edges[i : i + self.batch_size]
                batch_neg_edges = neg_edges[i : i + self.batch_size]
                batch_edge_weights = (
                    edge_weights[i : i + self.batch_size]
                    if self.loss_fn.loss_type == "weighted_softmax"
                    else None
                )
                # 前向计算损失
                loss = self.compute_loss(
                    batch_pos_edges,
                    batch_neg_edges,
                    batch_edge_weights,
                )

                eval_loss += loss.item()

        return eval_loss

    def train(
        self,
        nodes,
        edges,
        feat_net,
        feat_weight,
        num_weights,
        num_epochs,
        val_split=0.2,
        patience=5,
        edge_weights=None,
        result_dir="/home/yuyipei/graph_mosaic_integration/result",
        neg_sampling_mode=None,  # 负采样模式：可以是 "full", "matched" 或 "visible"
        neg_sampling_restrict=[
            (0, 4),
            (0, 6),
            (1, 4),
            (1, 6),
            (2, 5),
            (2, 6),
            (3, 5),
            (3, 6),
            (4, 5),
            (4, 6),
            (5, 6),
            (6, 5),
            (6, 4),
            (5, 4),
        ],  # 可选，限制采样的类型
    ):
        if self.add_feature_net:
            feat_weight= feat_weight* num_weights
            edges = torch.cat([edges, feat_net], dim=0)
            if edge_weights is not None:
                edge_weights = torch.cat([edge_weights, feat_weight], dim=0)
        # 添加类型信息到边集合
        edges_infor, nodes = add_type_columns_to_edges(edges, nodes)

        # 提取源节点和目标节点
        edges = edges.to(self.device)
        nodes = torch.tensor(nodes.values, dtype=torch.long, device=self.device)
        edges_infor = edges_infor.to(self.device)
        if edge_weights is not None:
            edge_weights = edge_weights.to(self.device)

        # 首先按第四列排序（次要排序关键字）
        sorted_indices_col4 = torch.argsort(edges_infor[:, 3], stable=True)
        edges_infor_sorted_col4 = edges_infor[
            sorted_indices_col4
        ]  # 应用第四列排序的结果，按第三列排序（主要排序关键字）
        sorted_indices_col3 = torch.argsort(edges_infor_sorted_col4[:, 2], stable=True)
        edges_infor_sorted = edges_infor_sorted_col4[sorted_indices_col3]

        best_eval_loss = float("inf")
        patience_counter = 0

        print("Before Epoch")
        print("Edge Weights (edge_weights):", edge_weights[:5])

        for epoch in tqdm(range(num_epochs), desc="Epoch: "):
            start_time = time.time()
            # 根据当前生成负样本
            neg_samples = NegativeSampler(
                nodes=nodes,  # 张量，节点信息
                num_cell=self.num_cell,  #
                num_neg_per_pos=self.num_neg_per_pos,  # 每条正边采样多少负边
                edges_infor=edges_infor_sorted,  # 排序好的边信息张量
                neg_sampling_mode=neg_sampling_mode,
                neg_sampling_restrict=neg_sampling_restrict,
            ).negative_sampling()

            # 划分训练和验证集
            num_samples = edges.shape[0]
            indices = torch.randperm(num_samples)
            split_idx = int(num_samples * (1 - val_split))
            train_indices, val_indices = indices[:split_idx], indices[split_idx:]
            train_edges, eval_edges = edges[train_indices], edges[val_indices]
            train_neg_samples, eval_neg_samples = (
                neg_samples[train_indices],
                neg_samples[val_indices],
            )
            train_edge_weights = (
                edge_weights[train_indices]
                if self.loss_fn.loss_type == "weighted_softmax"
                else None
            )
            eval_edge_weights = (
                edge_weights[val_indices]
                if self.loss_fn.loss_type == "weighted_softmax"
                else None
            )
            # 训练阶段
            epoch_loss = self.train_epoch(
                train_edges,
                train_neg_samples,
                train_edge_weights,
            )
            # 验证阶段
            eval_loss = self.evaluate(
                eval_edges,
                eval_neg_samples,
                eval_edge_weights,
            )
            self.train_losses.append(epoch_loss)
            self.eval_losses.append(eval_loss)
            tqdm.write(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Eval Loss: {eval_loss:.4f}"
            )
            # 早停检查
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                patience_counter = 0  # 重置计数器
            else:
                patience_counter += 1

            if patience_counter >= patience:
                tqdm.write(f"Early stopping triggered at epoch {epoch + 1}")
                break
            tqdm.write(f"Eval Time: {time.time() - start_time:.4f} seconds")

        if not os.path.exists(result_dir):  # 检查 result 文件夹是否存在，不存在则创建
            os.makedirs(result_dir)

        # 保存模型和嵌入
        model_path = os.path.join(result_dir, "model.pth")
        if self.add_feature_net:
            embedding_path = os.path.join(
                result_dir, f"final_embeddings_{neg_sampling_mode}_add_feat.csv"
            )
        else:
            embedding_path = os.path.join(
                result_dir, f"final_embeddings_{neg_sampling_mode}.csv"
            )
        torch.save(self.node_embedding.state_dict(), model_path)
        save_embeddings(self.node_embedding,embedding_path)

    def plot_losses(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.eval_losses, label="Eval Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Evaluation Loss")
        plt.legend()
        plt.grid()
        plt.savefig("./result/loss_plot.png")  # 保存图像
        print("Loss plot saved as 'loss_plot.png'")
