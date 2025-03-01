from dataclasses import dataclass

import pandas as pd
import numpy as np
import h5py


@dataclass
class MosaicDataGraph:
    # n_nodes: int
    # n_edges: int
    n_cells: int
    n_feats: int
    n_batch: int
    group_categories: list
    nodes_df: pd.DataFrame
    main_edges_df: pd.DataFrame
    main_edge_groups: np.ndarray
    feat_edges_df: pd.DataFrame | None = None
    feat_edge_groups: np.ndarray | None = None
    nodes_adversarial_weights: np.ndarray | None = None
    

    def __post_init__(self):
        self.edges_df = (
            self.main_edges_df
            if self.feat_edges_df is None
            else pd.concat([self.main_edges_df, self.feat_edges_df], axis=0)
        )
        self.edge_groups = (
            self.main_edge_groups
            if self.feat_edge_groups is None
            else np.concatenate(
                [self.main_edge_groups, self.feat_edge_groups], axis=0
            )
        )
        self.n_edges = self.edges_df.shape[0]
        self.n_nodes = self.nodes_df.shape[0]

    def __repr__(self) -> str:
        res = "MosaicDataGraph\n"
        res += f"  n_nodes={self.n_nodes}\n"
        res += f"  n_edges={self.n_edges}\n"
        res += f"  n_cells={self.n_cells}\n"
        res += f"  n_feats={self.n_feats}\n"
        res += f"  n_batch={self.n_batch}\n"
        res += f"  group_categories={','.join(self.group_categories)}"
        return res

    def save(self, fn: str):
        with h5py.File(fn, "w") as f:
            # f.attrs["n_nodes"] = self.n_nodes
            # f.attrs["n_edges"] = self.n_edges
            f.attrs["n_cells"] = self.n_cells
            f.attrs["n_feats"] = self.n_feats
            f.attrs["n_batch"] = self.n_batch
            f.attrs["group_categories"] = self.group_categories
            f.create_dataset("edge_groups", data=self.main_edge_groups)
            if self.feat_edge_groups is not None:
                f.create_dataset(
                    "feat_edge_groups", data=self.feat_edge_groups
                )
            if self.nodes_adversarial_weights is not None:
                f.create_dataset(
                    "nodes_adversarial_weights",
                    data=self.nodes_adversarial_weights,
                )
        self.nodes_df.to_hdf(fn, key="nodes_df", mode="a")
        self.main_edges_df.to_hdf(fn, key="main_edges_df", mode="a")
        if self.feat_edges_df is not None:
            self.feat_edges_df.to_hdf(fn, key="feat_edges_df", mode="a")

    @classmethod
    def load(self, fn: str) -> "MosaicDataGraph":
        with h5py.File(fn, "r") as f:
            # n_nodes = f.attrs["n_nodes"]
            # n_edges = f.attrs["n_edges"]
            n_cells = f.attrs["n_cells"]
            n_feats = f.attrs["n_feats"]
            n_batch = f.attrs["n_batch"]
            group_categories = f.attrs["group_categories"]

            edge_groups = f["edge_groups"][:]
            feat_edge_groups = (
                f["feat_edge_groups"][:] if "feat_edge_groups" in f else None
            )
            nodes_adversarial_weights = (
                f["nodes_adversarial_weights"][:]
                if "nodes_adversarial_weights" in f
                else None
            )

        nodes_df = pd.read_hdf(fn, "nodes_df")
        main_edges_df = pd.read_hdf(fn, "main_edges_df")
        feat_edges_df = (
            pd.read_hdf(fn, "feat_edges_df") if "feat_edges_df" in f else None
        )

        return MosaicDataGraph(
            # n_nodes=n_nodes,
            # n_edges=n_edges,
            n_cells=n_cells,
            n_feats=n_feats,
            n_batch=n_batch,
            group_categories=group_categories,
            nodes_df=nodes_df,
            main_edges_df=main_edges_df,
            main_edge_groups=edge_groups,
            feat_edges_df=feat_edges_df,
            feat_edge_groups=feat_edge_groups,
            nodes_adversarial_weights=nodes_adversarial_weights,
        )
