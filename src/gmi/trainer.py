from typing import Literal
from math import inf
from copy import deepcopy
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# from sklearn.cluster import KMeans
from tqdm.contrib.logging import logging_redirect_tqdm

from .graph import MosaicDataGraph
from .dataloader import GraphDataset

# from .loss import graph_mosaic_integration_loss
from .model import GMIModel, WEIGHT, WEIGHTS
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
        if self.count > 0:
            acc = self.total / self.count
            self.init()
            return acc
        return None  # no data seen yet


class LossAccumulator:
    def __init__(self):
        self.all: list[dict[str, float]] = []
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
        self.all.append(res)
        self.init()
        return res


class EarlyStopper:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_eval_loss = inf
        self.best_epoch = 0
        self.cnt = 0
        self.best_model = None

    def see(self, epoch: int, eval_loss: float, model: nn.Module) -> bool:
        """
        if True, break
        if False, continue train
        """
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_epoch = epoch
            self.cnt = 0
            self.best_model = deepcopy(model.state_dict())
            return False

        self.cnt += 1
        if self.cnt >= self.patience:
            if self.best_model is not None:
                model.load_state_dict(self.best_model)
            return True

    def info(self) -> str:
        return f"early stop, best epoch: {self.best_epoch}, best eval loss: {self.best_eval_loss:.4f}"

    def load_best(self, model: nn.Module):
        model.load_state_dict(self.best_model)


@dataclass
class Trainer:
    model: GMIModel
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer: Literal["adam", "rmsprop"] = "adam"
    lr: float = 0.001
    neg_sample_in_batch: bool = False
    adversarial_batching_method: Literal["unique", "divide", "random"] = "divide"
    batch_size: int = 128
    disc_node_num_per_batch: int = 200
    neg_sampling_mode: Literal["full", "matched", "bipartitle"] = "matched"
    loss_type: Literal["margin_ranking, weighted_softmax"] = "margin_ranking"
    patience: int | float = 5  # inf or np.inf表示不使用早停
    random_seed: int | None = None

    def __post_init__(self):
        # if adversarial_with_feature_nodes:
        #     raise NotImplementedError("adversarial_with_feature_nodes not implemented")

        self.device = torch.device(self.device)
        self.model.to(self.device)
        self.optimizer = {"adam": optim.Adam, "rmsprop": optim.RMSprop}[self.optimizer](
            self.model.parameters(), lr=self.lr
        )
        self.use_weights = self.loss_type == "weighted_softmax"

        self._evaluator = Evaluator()
        self._loss_accumulator = LossAccumulator()
        self._flag_use_early_stop = self.patience < inf and self.patience < np.inf
        if self._flag_use_early_stop:
            self._early_stopper = EarlyStopper(patience=self.patience)

    def train(
        self,
        graph: MosaicDataGraph,
        num_neg_per_pos: int = 5,
        num_epochs: int = 100,
        val_split: float | None = 0.2,
        disc_with_feature_nodes: bool = False,
        **weights: WEIGHTS,
    ):
        assert (
            not disc_with_feature_nodes
        ), "disc_with_feature_nodes not implemented yet"

        # NOTE: 使用weights中的w_loss_cls的值来判断是否要包含分类器
        w_loss_cls = weights.get("w_loss_cls", 1.0)
        self._flag_disc = (
            any(wi > 0 for wi in w_loss_cls)
            if isinstance(w_loss_cls, Sequence)
            else w_loss_cls > 0
        )

        # NOTE: 现在train test split是放在epoch循环的外面,
        #       这样能够保证valid时模型是无法看到test数据的
        self.graph = graph
        dataset_kwargs = dict(
            graph=graph,
            edge_batch_size=self.batch_size,
            device=self.device,
            use_edge_weights=self.use_weights,
            num_neg_per_pos=num_neg_per_pos,
            neg_sampling_mode=self.neg_sampling_mode,
            neg_sample_in_batch=self.neg_sample_in_batch,
            node_batch_size=self.disc_node_num_per_batch,
            disc_add_features=disc_with_feature_nodes,
            node_batching_method=self.adversarial_batching_method,
            return_node_groups=True,
        )

        if val_split is None:
            train_dataset = GraphDataset(
                subset=None,
                shuffle=True,
                node_discriminate=self._flag_disc,
                **dataset_kwargs,
            )
        else:
            train_indices, valid_indices = train_test_split(
                np.arange(graph.n_edges),
                test_size=val_split,
                random_state=self.random_seed,
            )
            train_dataset = GraphDataset(
                subset=train_indices,
                shuffle=True,
                node_discriminate=self._flag_disc,
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
                weights_e = {
                    k: v[epoch] if isinstance(v, Sequence) else v
                    for k, v in weights.items()
                }

                # 训练阶段
                self.train_epoch(train_dataset, **weights_e)
                train_losses = self._loss_accumulator.cal()
                domain_cls_acc = self._evaluator.cal()
                tqdm.write(
                    f"Epoch {epoch + 1}/{num_epochs}, Train, "
                    + (", ".join(f"{k} loss: {v:.4f}" for k, v in train_losses.items()))
                    + (
                        f", Domain classifer acc: {domain_cls_acc:.4f}"
                        if self._flag_disc and domain_cls_acc is not None
                        else ""
                    )
                )
                # 验证阶段
                if val_split is not None:
                    eval_loss = self.eval_epoch(
                        valid_dataset,
                        w_infonce_temp=weights_e.get("w_infonce_temp", 1.0),
                        w_cov=0.0,  # TODO: weights_e.get("w_cov", 0.0),
                    )
                    eval_losses.append(eval_loss)
                    tqdm.write(
                        f"Epoch {epoch + 1}/{num_epochs}, Valid, "
                        + f"Eval Loss: {eval_loss:.4f}"
                    )
                # 早停检查
                # 如果有valid，则使用valid的loss进行早停，否则使用train的loss进行早停
                if self._flag_use_early_stop and self._early_stopper.see(
                    epoch,
                    eval_loss if val_split is not None else train_losses["edge"],
                    self.model,
                ):
                    self._early_stopper.load_best(self.model)
                    tqdm.write(self._early_stopper.info())
                    break

        self.all_losses: pd.DataFrame = pd.DataFrame.from_records(
            self._loss_accumulator.all
        )
        if val_split is not None:
            self.all_losses["eval_loss"] = eval_losses

    # def __init__(
    #     self,
    #     model: GMIModel,
    #     device: str = "cuda" if torch.cuda.is_available() else "cpu",
    #     optimizer: Literal["adam", "rmsprop"] = "adam",
    #     lr: float = 0.001,
    #     neg_sample_in_batch: bool = False,
    #     # adversarial_training: bool = True,
    #     adversarial_batching_method: Literal["unique", "divide", "random"] = "divide",
    #     # adversarial_with_feature_nodes: bool = False,
    #     batch_size: int = 128,
    #     disc_node_num_per_batch: int = 200,
    #     # label_smoothing: float | Sequence[int] = 0.1,
    #     # grad_reverse_weight: float | Sequence[int] = 0.2,
    #     # cls_loss_weight: float | Sequence[int] = 0.2,
    #     # clu_loss_weight: float | Sequence[int] = 0.0,
    #     # cls_loss_temp: float | Sequence[int] = 1.0,
    #     # clu_loss_temp: float | Sequence[int] = 1.0,
    #     neg_sampling_mode: Literal["full", "matched", "bipartitle"] = "matched",
    #     loss_type: Literal["margin_ranking, weighted_softmax"] = "margin_ranking",
    #     patience: int | float = 5,  # inf or np.inf表示不使用早停
    #     random_seed: int | None = None,
    # ):
    #     self.device = torch.device(device)
    #     self.neg_sample_in_batch = neg_sample_in_batch
    #     self.adversarial_training = adversarial_training
    #     self.adversarial_batching_method = adversarial_batching_method
    #     self.adversarial_with_feature_nodes = adversarial_with_feature_nodes
    #     self.batch_size = batch_size
    #     self.disc_node_num_per_batch = disc_node_num_per_batch
    #     self.neg_sampling_mode = neg_sampling_mode
    #     # TODO: now the seed can only control the randomness of the data split
    #     self.random_seed = random_seed
    #
    #     # self.label_smoothing = label_smoothing
    #     # self.grad_reverse_weight = grad_reverse_weight
    #     # self.cls_loss_weight = cls_loss_weight
    #     # self.clu_loss_weight = clu_loss_weight
    #     # self.cls_loss_temp = cls_loss_temp
    #     # self.clu_loss_temp = clu_loss_temp
    #
    #     if adversarial_with_feature_nodes:
    #         raise NotImplementedError("adversarial_with_feature_nodes not implemented")
    #
    #     # 初始化模型
    #     self.model = model.to(self.device)
    #
    #     # 初始化优化器
    #     if optimizer == "adam":
    #         self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    #     elif optimizer == "rmsprop":
    #         self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
    #     else:
    #         raise ValueError("optimizer should be 'adam' or 'rmsprop'")
    #
    #     # 初始化损失函数
    #     self.loss_type = loss_type
    #     self.use_weights = self.loss_type == "weighted_softmax"
    #
    #     self._evaluator = Evaluator()
    #     self._loss_accumulator = LossAccumulator()
    #     self._flag_use_early_stop = patience < inf and patience < np.inf
    #     if self._flag_use_early_stop:
    #         self._early_stopper = EarlyStopper(patience=patience)

    def train_epoch(
        self,
        train_dataset: GraphDataset,
        **weights: WEIGHT,
    ):
        self.model.train()

        for batch in tqdm(train_dataset, desc="Batch: ", leave=False):
            loss, loss_dict, others = self.model(
                pos_edges=batch["pos_edges"],
                neg_edges=batch["neg_edges"],
                pos_edges_weights=batch.get("edge_weights", None),
                domain_input=batch.get("input", None),
                domain_label=batch.get("label", None),
                domain_weights=batch.get("weight", None),
                **weights,
            )

            # 反向传播与优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 累计loss和domain acc
            self._loss_accumulator.see(loss_dict)
            if "pred" in others:
                self._evaluator.see(others["pred"], batch["label"])

    def eval_epoch(
        self,
        eval_dataset: GraphDataset,
        w_infonce_temp: float,
        w_cov: float,
    ) -> torch.Tensor:
        self.model.eval()
        # 开始评估
        eval_loss, cnt = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(eval_dataset, desc="Batch(valid): ", leave=False):
                # 获取当前批次的正样本
                loss, _, _ = self.model(
                    batch["pos_edges"],
                    batch["neg_edges"],
                    pos_edges_weights=batch.get("edge_weights", None),
                    domain_input=None,
                    w_infonce_temp=w_infonce_temp,
                    w_loss_cls=0.0,
                    w_loss_clu=0.0,
                    w_cov=w_cov,
                )

                eval_loss += loss.item()
                cnt += 1

        eval_loss /= cnt
        return eval_loss

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
        cell_embeddings = self.model.node_embedding.weight[: self.graph.n_cells]
        cell_groups = self.graph.nodes_df["group"].iloc[: self.graph.n_cells].values
        weights = estimate_balance_weights_glue(
            cell_embeddings.detach().cpu().numpy(),
            cell_groups,
            resolution=resolution,
            cutoff=cutoff,
            power=power,
        )
        return weights
