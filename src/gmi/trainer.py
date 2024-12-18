from typing import Literal
from math import inf
from copy import deepcopy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tqdm.contrib.logging import logging_redirect_tqdm

from .graph import MosaicDataGraph
from .dataloader import GraphDataset
from .loss import graph_mosaic_integration_loss
from .model import FullModel
from .balance_weights import estimate_balance_weights_glue


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

        self._evaluator = Evaluator()
        self._loss_accumulator = LossAccumulator()
        self._flag_use_early_stop = patience < inf and patience < np.inf
        if self._flag_use_early_stop:
            self._early_stopper = EarlyStopper(patience=patience)

    def train_epoch(
        self,
        train_dataset: GraphDataset,
        alpha: float = 1.0,
        loss_alpha: float = 1.0,
    ):
        self.model.train()

        for batch in tqdm(train_dataset, desc="Batch: ", leave=False):
            pos_scores, neg_scores, pos_domain_preds = self.model(
                batch["pos_edges"],
                batch["neg_edges"],
                pos_domain_input=batch["input"]
                if self.adversarial_training
                else None,
                node_batch_indice=batch["node_groups"],
                alpha=alpha,
                num_cells=self.graph.n_cells,
            )
            # 计算当前批次的损失
            loss, loss_dict = graph_mosaic_integration_loss(
                pos_scores=pos_scores,
                neg_scores=neg_scores,
                edge_weights=batch["edge_weights"]
                if self.loss_type == "weighted_softmax"
                else None,
                domain_preds=pos_domain_preds
                if self.adversarial_training
                else None,
                domain_labels=batch["label"]
                if self.adversarial_training
                else None,
                discriminate_weights=batch["weight"]
                if self.adversarial_training
                else None,
                edge_loss_type=self.loss_type,
                loss_alpha=loss_alpha,
                label_smoothing=self.label_smoothing,
            )

            # 反向传播与优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 累计loss和domain acc
            self._loss_accumulator.see(loss_dict)
            if pos_domain_preds is not None:
                self._evaluator.see(pos_domain_preds, batch["label"])

    def evaluate(self, eval_dataset: GraphDataset) -> torch.Tensor:
        self.model.eval()
        # 开始评估
        eval_loss, cnt = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(
                eval_dataset, desc="Batch(valid): ", leave=False
            ):
                # 获取当前批次的正样本
                pos_scores, neg_scores, _ = self.model(
                    batch["pos_edges"],
                    batch["neg_edges"],
                    node_batch_indice=batch["node_groups"],
                    num_cells=self.graph.n_cells,
                )
                # 前向传播
                # 前向计算损失
                # 没有adversarial training，所以不需要loss_alpha和alpha
                loss, _ = graph_mosaic_integration_loss(
                    pos_scores=pos_scores,
                    neg_scores=neg_scores,
                    edge_weights=batch["edge_weights"]
                    if self.loss_type == "weighted_softmax"
                    else None,
                    edge_loss_type=self.loss_type,
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
        # NOTE: 现在train test split是放在epoch循环的外面,
        #       这样能够保证valid时模型是无法看到test数据的
        self.graph = graph
        dataset_kwargs = dict(
            graph=graph,
            edge_batch_size=self.batch_size,
            device=self.device,
            use_edge_weights=self.loss_type == "weighted_softmax",
            num_neg_per_pos=num_neg_per_pos,
            neg_sampling_mode=self.neg_sampling_mode,
            neg_sample_in_batch=self.neg_sample_in_batch,
            node_batch_size=self.disc_node_num_per_batch,
            disc_add_features=self.adversarial_with_feature_nodes,
            node_batching_method=self.adversarial_batching_method,
            return_node_groups=True,
        )
        if val_split is None:
            train_dataset = GraphDataset(
                subset=None,
                shuffle=True,
                node_discriminate=self.adversarial_training,
                **dataset_kwargs,
            )
        else:
            train_indices, valid_indices = train_test_split(
                np.arange(graph.n_edges), test_size=val_split
            )
            train_dataset = GraphDataset(
                subset=train_indices,
                shuffle=True,
                node_discriminate=self.adversarial_training,
                **dataset_kwargs,
            )
            valid_dataset = GraphDataset(
                subset=valid_indices,
                shuffle=False,
                node_discriminate=False,
                **dataset_kwargs,
            )

        eval_losses = []
        with logging_redirect_tqdm():
            for epoch in tqdm(range(num_epochs), desc="Epoch: "):
                # 划分训练和验证集
                # num_samples = edges.shape[0]
                # indices = torch.randperm(num_samples)
                # split_idx = int(num_samples * (1 - val_split))
                # train_indices, val_indices = (
                #     indices[:split_idx],
                #     indices[split_idx:],
                # )
                # train_edges, eval_edges = (
                #     edges[train_indices],
                #     edges[val_indices],
                # )
                # train_edge_weights = (
                #     edge_weights[train_indices] if self.use_weights else None
                # )
                # eval_edge_weights = (
                #     edge_weights[val_indices] if self.use_weights else None
                # )

                # 训练阶段
                self.train_epoch(
                    train_dataset,
                    alpha=0.0 if epoch < self.late_join_alpha else self.alpha,
                    loss_alpha=0.0
                    if epoch < self.late_join_loss_alpha
                    else self.loss_alpha,
                )
                train_losses = self._loss_accumulator.cal()
                tqdm.write(
                    f"Epoch {epoch + 1}/{num_epochs}, Train, "
                    + (
                        ", ".join(
                            f"{k} loss: {v:.4f}"
                            for k, v in train_losses.items()
                        )
                    )
                    + (
                        f", Domain classifer acc: {self._evaluator.cal():.4f}"
                        if self.adversarial_training
                        else ""
                    )
                )
                # 验证阶段
                if val_split is not None:
                    eval_loss = self.evaluate(valid_dataset)
                    eval_losses.append(eval_loss)
                    tqdm.write(
                        f"Epoch {epoch + 1}/{num_epochs}, Valid, "
                        + f"Eval Loss: {eval_loss:.4f}"
                    )
                # 早停检查
                # 如果有valid，则使用valid的loss进行早停，否则使用train的loss进行早停
                if self._flag_use_early_stop and self._early_stopper.see(
                    eval_loss
                    if val_split is not None
                    else train_losses["edge"],
                    self.model,
                ):
                    self._early_stopper.load_best(self.model)
                    break

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

        pd.DataFrame(self.all_losses).to_csv(
            os.path.join(result_dir, "all_losses.csv")
        )

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

    def estimate_balance_weights(
        self, resolution: float = 1.0, cutoff: float = 0.5, power: float = 4.0
    ) -> np.ndarray:
        cell_embeddings = self.model.node_embedding.weight[
            : self.graph.n_cells
        ]
        cell_groups = (
            self.graph.nodes_df["group"].iloc[: self.graph.n_cells].values
        )
        weights = estimate_balance_weights_glue(
            cell_embeddings.detach().cpu().numpy(),
            cell_groups,
            resolution=resolution,
            cutoff=cutoff,
            power=power,
        )
        return weights
