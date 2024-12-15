from typing import Literal
from math import ceil, inf
from copy import deepcopy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .graph import MosaicDataGraph
from .negative_sample import NegativeSampler
from .loss import LossType, compute_loss
from .model import FullModel


class Evaluator:
    def __init__(self):
        self.init()

    def init(self):
        self.total = 0.0
        self.count = 0

    def see(self, preds: torch.Tensor, labels: torch.Tensor):
        self.total += (preds.argmax(dim=1) == labels).sum().item()
        self.count += preds.shape[0]

    def cal(self) -> float:
        acc = self.total / self.count
        self.init()
        return acc


class LossAccumulator:
    def __init__(self):
        self.all: dict[str, list[float]] = {}
        self.init()

    def init(self):
        self.total = {}
        self.count = 0

    def see(self, loss_dict: dict[str, torch.Tensor]):
        self.count += 1
        for k, v in loss_dict.items():
            self.total[k] = self.total.get(k, 0.0) + v.item()

    def cal(self) -> float:
        res = {k: v / self.count for k, v in self.total.items()}
        for k, v in res.items():
            self.all.setdefault(k, []).append(v)
        self.init()
        return res


class EarlyStopper:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_eval_loss = inf
        self.cnt = 0
        self.best_model = None

    def see(self, eval_loss: float, model: nn.Module) -> bool:
        """
        if True, break
        if False, continue train
        """
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.cnt = 0
            self.best_model = deepcopy(model.state_dict())
            return False

        self.cnt += 1
        if self.cnt >= self.patience:
            if self.best_model is not None:
                model.load_state_dict(self.best_model)
            return True

    def load_best(self, model: nn.Module):
        model.load_state_dict(self.best_model)


class Trainer:
    def __init__(
        self,
        model: FullModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        optimizer: Literal["adam", "rmsprop"] = "adam",
        lr: float = 0.001,
        neg_sample_in_batch: bool = False,
        adversarial_training: bool = True,
        adversarial_batching_method: Literal[
            "unique", "divide", "random"
        ] = "divide",
        adversarial_with_feature_nodes: bool = False,
        batch_size: int = 128,
        disc_node_num_per_batch: int = 200,
        label_smoothing: float = 0.1,
        alpha: float = 0.2,
        loss_alpha: float = 0.2,
        neg_sampling_mode: Literal[
            "full", "matched", "bipartitle"
        ] = "matched",
        loss_type: Literal[
            "margin_ranking, weighted_softmax"
        ] = "margin_ranking",
        late_join_loss_alpha: int | None = None,
        late_join_alpha: int | None = None,
        patience: int | float = 5,  # inf or np.inf表示不使用早停
    ):
        self.device = torch.device(device)
        self.neg_sample_in_batch = neg_sample_in_batch
        self.adversarial_training = adversarial_training
        self.adversarial_batching_method = adversarial_batching_method
        self.adversarial_with_feature_nodes = adversarial_with_feature_nodes
        self.batch_size = batch_size
        self.disc_node_num_per_batch = disc_node_num_per_batch
        self.label_smoothing = label_smoothing
        self.alpha = alpha
        self.loss_alpha = loss_alpha
        self.neg_sampling_mode = neg_sampling_mode
        self.late_join_loss_alpha = late_join_loss_alpha
        self.late_join_alpha = late_join_alpha

        if adversarial_with_feature_nodes:
            raise NotImplementedError(
                "adversarial_with_feature_nodes not implemented"
            )

        # 初始化模型
        self.model = model.to(self.device)

        # 初始化优化器
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        else:
            raise ValueError("optimizer should be 'adam' or 'rmsprop'")

        # 初始化损失函数
        self.loss_type = loss_type
        self.use_weights = self.loss_type == "weighted_softmax"
        self.loss_fn = LossType(loss_type=loss_type).to(self.device)

        self._evaluator = Evaluator()
        self._loss_accumulator = LossAccumulator()
        self._flag_use_early_stop = patience < inf and patience < np.inf
        if self._flag_use_early_stop:
            self._early_stopper = EarlyStopper(patience=patience)

    def train_epoch(
        self,
        edges: torch.Tensor,  # n x 2
        edge_weights: torch.Tensor | None,
        alpha: float = 1.0,
        loss_alpha: float = 1.0,
    ):
        if not self.neg_sample_in_batch:
            neg_edges = self.neg_sampler.negative_sampling(edges)

        self.model.train()

        n_batches = (edges.shape[0] + self.batch_size - 1) // self.batch_size
        if (
            self.adversarial_training
            and self.adversarial_batching_method == "divide"
        ):
            # 这样只有cell nodes参与discriminator的训练
            n_nodes_per_domain_batch = ceil(self.graph.n_cells / n_batches)
            nodes_rand = torch.randperm(
                self.graph.n_cells, device=edges.device
            )
            nodes_group_rand = self.nodes_group[nodes_rand]

        for i in tqdm(range(n_batches), desc="Batch: ", leave=False):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            # 获取当前批次的正样本
            batch_pos_edges = edges[start:end, :2]
            # 获取当前批次的负样本
            if self.neg_sample_in_batch:
                batch_neg_edges = self.neg_sampler.negative_sampling(
                    batch_pos_edges
                )
            else:
                batch_neg_edges = neg_edges[start:end]

            batch_edge_weights = (
                edge_weights[start:end] if self.use_weights else None
            )

            if not self.adversarial_training:
                batch_domain_input, batch_domain_label = None, None
            elif self.adversarial_batching_method in ["unique", "random"]:
                src_nodes = batch_pos_edges[:, 0]
                # NOTE: cell node在nodes中排在最前面
                mask = src_nodes < self.graph.n_cells
                filtered_src_nodes = src_nodes[mask]
                if filtered_src_nodes.shape[0] == 0:
                    batch_domain_input, batch_domain_label = None, None
                else:
                    batch_domain_input = torch.unique(
                        filtered_src_nodes, sorted=False
                    )
                    if self.adversarial_batching_method == "random":
                        ind = torch.randperm(
                            batch_domain_input.shape[0],
                            device=batch_domain_input.device,
                        )[: self.disc_node_num_per_batch]
                        batch_domain_input = batch_domain_input[ind]
                    batch_domain_label = self.nodes_group[batch_domain_input]
            elif self.adversarial_batching_method == "divide":
                start_node = i * n_nodes_per_domain_batch
                end_node = (i + 1) * n_nodes_per_domain_batch
                batch_domain_input = nodes_rand[start_node:end_node]
                batch_domain_label = nodes_group_rand[start_node:end_node]
                batch_domain_label, batch_domain_input = (
                    batch_domain_label,
                    batch_domain_input,
                )
            else:
                raise ValueError(
                    "adversarial_batching_method should be 'unique', "
                    f"'divide', or 'random', "
                    f"but got {self.adversarial_batching_method}"
                )

            pos_scores, neg_scores, pos_domain_preds = self.model(
                batch_pos_edges,
                batch_neg_edges,
                pos_domain_input=batch_domain_input,
                node_batch_indice=self.nodes_group,
                alpha=alpha,
                num_cells=self.graph.n_cells,
            )
            # 计算当前批次的损失
            loss, loss_dict = compute_loss(
                pos_scores=pos_scores,
                neg_scores=neg_scores,
                pos_domain_preds=pos_domain_preds,
                domain_labels=batch_domain_label,
                edge_weights=batch_edge_weights,
                loss_fn=self.loss_fn,
                loss_alpha=loss_alpha,
                label_smoothing=self.label_smoothing,
            )
            # 反向传播与优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 累计loss和domain acc
            self._loss_accumulator.see(loss_dict)
            self._evaluator.see(pos_domain_preds, batch_domain_label)

    def evaluate(
        self,
        edges: torch.Tensor,  # n x 2
        edge_weights: torch.Tensor | None,
    ):
        if not self.neg_sample_in_batch:
            neg_edges = self.neg_sampler.negative_sampling(edges)

        self.model.eval()

        # 开始评估
        eval_loss, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, edges.shape[0], self.batch_size):
                # 获取当前批次的正样本
                batch_pos_edges = edges[i : i + self.batch_size]
                if self.neg_sample_in_batch:
                    batch_neg_edges = self.neg_sampler.negative_sampling(
                        batch_pos_edges
                    )
                else:
                    batch_neg_edges = neg_edges[i : i + self.batch_size]
                batch_edge_weights = (
                    edge_weights[i : i + self.batch_size]
                    if self.use_weights
                    else None
                )

                pos_scores, neg_scores, _ = self.model(
                    batch_pos_edges,
                    batch_neg_edges,
                    node_batch_indice=self.nodes_group,
                    num_cells=self.graph.n_cells,
                )
                # 前向传播
                # 前向计算损失
                # 没有adversarial training，所以不需要loss_alpha和alpha
                loss, _ = compute_loss(
                    pos_scores=pos_scores,
                    neg_scores=neg_scores,
                    edge_weights=batch_edge_weights,
                    loss_fn=self.loss_fn,
                )

                eval_loss += loss.item()
                cnt += 1

        eval_loss /= cnt
        return eval_loss

    def train(
        self,
        graph: MosaicDataGraph,
        num_neg_per_pos: int = 5,
        num_epochs: int = 100,
        val_split: float | None = 0.2,
    ):
        edges_df = graph.main_edges_df
        if graph.feat_edges_df is not None:
            edges_df = pd.concat([edges_df, graph.feat_edges_df], axis=0)
        edges = torch.tensor(
            edges_df[["src", "dst"]].values,
            dtype=torch.long,
            device=self.device,
        )
        edge_weights = (
            torch.tensor(
                edges_df["weight"].values,
                dtype=torch.float32,
                device=self.device,
            )
            if self.use_weights
            else None
        )
        node_group = torch.tensor(
            graph.nodes_df["group"].values,
            dtype=torch.long,
            device=self.device,
        )
        edge_group = (
            graph.edge_groups
            if graph.feat_edge_groups is None
            else np.concatenate(
                [graph.edge_groups, graph.feat_edge_groups], axis=0
            )
        )

        # 记录一下，方便后续每个epoch使用
        self.graph = graph
        self.nodes_group = node_group

        print("Before Epoch")
        print("Edge Weights (edge_weights):", edge_weights[:5])

        self.neg_sampler = NegativeSampler(
            n_nodes=graph.n_nodes,
            num_cell=graph.n_cells,
            num_neg_per_pos=num_neg_per_pos,
            neg_sampling_mode=self.neg_sampling_mode,
            node_group=node_group,
            neg_sampling_restrict=edge_group,
        )

        eval_losses = []
        for epoch in tqdm(range(num_epochs), desc="Epoch: "):
            # 划分训练和验证集
            num_samples = edges.shape[0]
            indices = torch.randperm(num_samples)
            split_idx = int(num_samples * (1 - val_split))
            train_indices, val_indices = (
                indices[:split_idx],
                indices[split_idx:],
            )
            train_edges, eval_edges = (
                edges[train_indices],
                edges[val_indices],
            )
            train_edge_weights = (
                edge_weights[train_indices] if self.use_weights else None
            )
            eval_edge_weights = (
                edge_weights[val_indices] if self.use_weights else None
            )

            # 训练阶段
            self.train_epoch(
                edges=train_edges,
                edge_weights=train_edge_weights,
                alpha=0.0 if epoch < self.late_join_alpha else self.alpha,
                loss_alpha=0.0
                if epoch < self.late_join_loss_alpha
                else self.loss_alpha,
            )
            # 验证阶段
            eval_loss = self.evaluate(
                eval_edges,
                eval_edge_weights,
            )
            # self.train_losses.append(epoch_loss)
            train_losses = self._loss_accumulator.cal()
            eval_losses.append(eval_loss)
            tqdm.write(
                f"Epoch {epoch + 1}/{num_epochs}, "
                + (
                    ", ".join(
                        f"{k} loss: {v:.4f}" for k, v in train_losses.items()
                    )
                )
                + f", Eval Loss: {eval_loss:.4f}, "
                + f"Domain classifer acc: {self._evaluator.cal():.4f}"
            )
            # 早停检查
            if self._flag_use_early_stop and self._early_stopper.see(
                eval_loss, self.model
            ):
                break
        else:
            self._early_stopper.load_best(self.model)

        self.all_losses: dict[str, list[float]] = self._loss_accumulator.all
        self.all_losses["eval_loss"] = eval_losses

    def save(self, result_dir: str):
        os.makedirs(result_dir, exist_ok=True)
        # 保存模型和嵌入
        model_path = os.path.join(result_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)

        if self.graph.feat_edges_df is not None:
            embedding_path = os.path.join(
                result_dir,
                f"final_embeddings_{self.neg_sampling_mode}_add_feat.csv",
            )
        else:
            embedding_path = os.path.join(
                result_dir, f"final_embeddings_{self.neg_sampling_mode}.csv"
            )
        embed_df = pd.DataFrame(
            self.model.node_embedding.weight.detach().cpu().numpy(),
            index=self.graph.nodes_df.index,
        )
        embed_df.to_csv(embedding_path)

    def plot_losses(self, fn: str):
        fig, ax = plt.subplots(figsize=(8, 6))
        for k, v in self.all_losses.items():
            ax.plot(v, label=f"{k} Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Evaluation Loss")
        ax.legend()
        ax.grid()
        fig.savefig(fn)  # 保存图像
