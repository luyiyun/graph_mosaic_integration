from typing import Literal
from math import ceil

import numpy as np
import torch

from .graph import MosaicDataGraph
from .negative_sample import NegativeSampler


class EdgesDataset:
    def __init__(
        self,
        graph: MosaicDataGraph,
        batch_size: int = 1024,
        subset: np.ndarray | None = None,
        device: torch.device = torch.device("cpu"),
        use_edge_weights: bool = True,
        num_neg_per_pos: int = 4,
        neg_sampling_mode: Literal[
            "full", "matched", "bipartitle"
        ] = "matched",
        shuffle: bool = True,
        neg_sample_in_batch: bool = False,
        return_ndoe_groups: bool = False,
    ):
        self.graph = graph
        self.batch_size = batch_size
        self.device = device
        self.use_edge_weights = use_edge_weights
        self.num_neg_per_pos = num_neg_per_pos
        self.neg_sampling_mode = neg_sampling_mode
        self.shuffle = shuffle
        self.neg_sample_in_batch = neg_sample_in_batch
        self.return_ndoe_groups = return_ndoe_groups

        edges_df = graph.edges_df
        if subset is not None:
            edges_df = edges_df.iloc[subset, :]
        # 将所有需要的内容都先放在device上
        self._edges = torch.tensor(
            edges_df[["src", "dst"]].values,
            dtype=torch.long,
            device=self.device,
        )
        self._edge_weights = (
            torch.tensor(
                edges_df["weight"].values,
                dtype=torch.float32,
                device=self.device,
            )
            if use_edge_weights
            else None
        )
        self._node_group = torch.tensor(
            graph.nodes_df["group"].values,
            dtype=torch.long,
            device=self.device,
        )
        self._edge_group = graph.edge_groups

        self.neg_sampler = NegativeSampler(
            n_nodes=graph.n_nodes,
            num_cell=graph.n_cells,
            num_neg_per_pos=num_neg_per_pos,
            neg_sampling_mode=self.neg_sampling_mode,
            node_group=self._node_group,
            neg_sampling_restrict=self._edge_group,
        )

        self.init()  # TODO: 这样会导致第一次iter的时候相当于运行了两次

    def init(self):
        self._idx = 0
        if self.shuffle:
            indices = torch.randperm(self._edges.shape[0], device=self.device)
            self._edges = self._edges[indices]
            if self.use_edge_weights:
                self._edge_weights = self._edge_weights[indices]
        if not self.neg_sample_in_batch:
            self._neg_edges = self.neg_sampler.negative_sampling(self._edges)

    def __len__(self) -> int:
        return (self._edges.shape[0] + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start, end = (
            idx * self.batch_size, (idx + 1) * self.batch_size,
        )
        # 获取当前批次的正样本
        batch_pos_edges = self._edges[start:end, :]
        # 获取当前批次的负样本
        if self.neg_sample_in_batch:
            batch_neg_edges = self.neg_sampler.negative_sampling(
                batch_pos_edges
            )
        else:
            batch_neg_edges = self._neg_edges[start:end]
        # 获取当前批次的边权重
        batch_edge_weights = (
            self._edge_weights[start:end] if self.use_edge_weights else None
        )

        if self.return_ndoe_groups:
            return {
                "pos_edges": batch_pos_edges,
                "neg_edges": batch_neg_edges,
                "edge_weights": batch_edge_weights,
                "node_groups": self._node_group,
            }
        return {
            "pos_edges": batch_pos_edges,
            "neg_edges": batch_neg_edges,
            "edge_weights": batch_edge_weights,
        }


class NodesDataset:
    def __init__(
        self,
        graph: MosaicDataGraph,
        n_batches: int | None = None,
        batch_size: int | None = None,
        device: torch.device = torch.device("cpu"),
        add_features: bool = False,
        batching_method: Literal["divide", "random", "unique"] = "divide",
        shuffle: bool = True,
    ):
        if batching_method == "divide" and n_batches is None:
            raise ValueError(
                "n_batches should be specified "
                "when batching_method is 'divide'"
            )
        if batching_method == "random" and batch_size is None:
            raise ValueError(
                "batch_size should be specified "
                "when batching_method is 'random'"
            )
        if batching_method == "random" and not shuffle:
            raise ValueError(
                "shuffle should be True when batching_method is 'random'"
            )
        if add_features:
            raise NotImplementedError("add_features is not supported yet")

        self.graph = graph
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.device = device
        self.add_features = add_features
        self.batching_method = batching_method
        self.shuffle = shuffle

        self._n_nodes = graph.n_nodes if add_features else graph.n_cells
        self._nodes_group = torch.tensor(
            graph.nodes_df["group"].values
            if add_features
            else graph.nodes_df.iloc[: graph.n_cells]["group"].values,
            dtype=torch.long,
            device=self.device,
        )
        # NOTE: 这里的adv weights只是cell nodes的，不包括feature nodes
        self._node_adv_weights = (
            torch.tensor(
                graph.nodes_adversarial_weights,
                dtype=torch.float32,
                device=self.device,
            )
            if graph.nodes_adversarial_weights is not None
            else None
        )

        if batching_method == "divide":
            self.batch_size = ceil(self._n_nodes / n_batches)

        self.init()

    def init(self):
        if self.batching_method in ["divide", "random"] and self.shuffle:
            self._indice = torch.randperm(self._n_nodes, device=self.device)
            self._nodes_group = self._nodes_group[self._indice]
            self._node_adv_weights = (
                self._node_adv_weights[self._indice]
                if self._node_adv_weights is not None
                else None
            )

    def __len__(self) -> int:
        if self.batching_method == "divide":
            return self.n_batches
        if self.batching_method == "random":
            return (self._n_nodes + self.batch_size - 1) // self.batch_size
        if self.batching_method == "unique":
            return self.n_batches

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.batching_method == "unique":
            raise ValueError(
                "getitem when batching_method is 'unique' is not defined"
            )

        # TODO: 这里random的实现和之前不太一样。之前的random是从
        #   edges中包含的nodes中随机抽取，现在的random是从所有nodes中
        #   随机抽取。
        if self.batching_method in ["divide", "random"]:
            start = idx * self.batch_size
            end = (idx + 1) * self.batch_size
            batch_domain_input = (
                self._indice[start:end]
                if self.shuffle
                else torch.arange(start, end, device=self.device)
            )
            batch_domain_label = self._nodes_group[start:end]
            batch_adv_weights = (
                self._node_adv_weights[start:end]
                if self._node_adv_weights is not None
                else None
            )
            return {
                "input": batch_domain_input,
                "label": batch_domain_label,
                "weight": batch_adv_weights,
            }

    def batch_from_edges(self, edges: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        当batching_method为unique时，可以通过调用这个函数从每个edges来获取nodes batch
        """
        if self.batching_method != "unique":
            raise ValueError(
                "batch_from_edges is only supported "
                "when batching_method is 'unique'"
            )

        if not self.add_features:
            nodes = edges[edges < self.graph.n_cells]
        else:
            nodes = edges.flatten()
        if nodes.shape[0] == 0:
            return {
                "input": None,
                "label": None,
                "weight": None,
            }

        batch_domain_input = torch.unique(nodes, sorted=False)
        # if self.adversarial_batching_method == "random":
        #     ind = torch.randperm(
        #         batch_domain_input.shape[0],
        #         device=batch_domain_input.device,
        #     )[: self.disc_node_num_per_batch]
        #     batch_domain_input = batch_domain_input[ind]
        batch_domain_label = self._nodes_group[batch_domain_input]
        batch_adv_weights = (
            self._node_adv_weights[batch_domain_input]
            if self._node_adv_weights is not None
            else None
        )
        return {
            "input": batch_domain_input,
            "label": batch_domain_label,
            "weight": batch_adv_weights,
        }


class GraphDataset:
    def __init__(
        self,
        graph: MosaicDataGraph,
        edge_batch_size: int = 1024,
        subset: np.ndarray | None = None,
        device: torch.device = torch.device("cpu"),
        use_edge_weights: bool = True,
        num_neg_per_pos: int = 4,
        neg_sampling_mode: Literal[
            "full", "matched", "bipartitle"
        ] = "matched",
        shuffle: bool = True,
        neg_sample_in_batch: bool = False,
        return_node_groups: bool = True,
        node_discriminate: bool = True,
        node_batch_size: int | None = None,
        disc_add_features: bool = False,
        node_batching_method: Literal["divide", "random", "unique"] = "divide",
    ):
        self.node_discriminate = node_discriminate
        self.node_batching_method = node_batching_method
        self.edge_dataset = EdgesDataset(
            graph,
            batch_size=edge_batch_size,
            subset=subset,
            device=device,
            use_edge_weights=use_edge_weights,
            num_neg_per_pos=num_neg_per_pos,
            neg_sampling_mode=neg_sampling_mode,
            shuffle=shuffle,
            neg_sample_in_batch=neg_sample_in_batch,
            return_ndoe_groups=return_node_groups,
        )
        if node_discriminate:
            self.node_dataset = NodesDataset(
                graph,
                n_batches=len(self.edge_dataset),
                batch_size=node_batch_size,
                device=device,
                add_features=disc_add_features,
                batching_method=node_batching_method,
                shuffle=shuffle,
            )

    def __iter__(self):
        if not self.node_discriminate:
            self.edge_dataset.init()
            for i in range(len(self.edge_dataset)):
                yield self.edge_dataset[i]
        else:  # 因为上面是yield，必须加else
            if self.node_batching_method in ["divide", "random"]:
                self.edge_dataset.init()
                self.node_dataset.init()
                for i in range(len(self.edge_dataset)):
                    edge_batch = self.edge_dataset[i]
                    node_batch = self.node_dataset[i]
                    edge_batch.update(node_batch)
                    yield edge_batch

            if self.node_batching_method == "unique":
                self.edge_dataset.init()
                for i in range(len(self.edge_dataset)):
                    edge_batch = self.edge_dataset[i]
                    node_batch = self.node_dataset.batch_from_edges(
                        edge_batch["pos_edges"]
                    )
                    edge_batch.update(node_batch)
                    yield edge_batch

    def __len__(self) -> int:
        return len(self.edge_dataset)
