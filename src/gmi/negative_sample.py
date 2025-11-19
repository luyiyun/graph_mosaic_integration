from typing import Literal
from dataclasses import dataclass
from logging import getLogger

import torch


logger = getLogger(__name__)


def get_edge_indices(
    n_nodes: int, src: torch.Tensor, dst: torch.Tensor
) -> torch.Tensor:
    """
    计算边的索引。
    参数：
    edges: torch.Size([x, 2]) 边的起始节点和目标节点的索引。
    返回：
    边的索引。
    """
    return torch.minimum(src, dst) * n_nodes + torch.maximum(src, dst)


@dataclass
class NegativeSampler:
    """
    full: 所有未包含在edges中的边均被认为是负采样的候选，比如：feature-feature，feature-cell
    bipartitle: feature-feature和feature-cell的边被排除，只包含cell-feature的边
    visible: 只有包含在可获得组学矩阵中的负边才被认为是负采样的候选
    matched：对于每一条正边，只有其所在组学矩阵中的负边才被认为是其负采样的候选
    """

    n_nodes: int
    n_cells: int
    num_neg_per_pos: int
    all_edges: torch.Tensor
    neg_sampling_mode: Literal["full", "bipartitle", "matched", "visible"]
    node_group: torch.Tensor | None = None
    neg_sampling_restrict: list[tuple[int, int]] | None = None

    def __post_init__(self):
        if self.neg_sampling_mode == "matched":
            assert self.neg_sampling_restrict is not None, (
                "neg_sampling_restrict must be provided "
                "when neg_sampling_mode is 'matched'"
            )
            assert self.node_group is not None, (
                "node_group must be provided "
                "when neg_sampling_mode is 'matched'"
            )
            assert (
                self.node_group.shape[0] == self.n_nodes
            ), "node_group.shape[0]!= n_nodes"

        self.all_edges_idx = get_edge_indices(self.n_nodes, *self.all_edges.T)

    def sample(self, edges: torch.Tensor) -> torch.Tensor:
        """
        根据不同的负采样模式执行负采样操作。

        返回：
        - 负样本的张量。
        """
        src, dst = edges.T
        pos_edges_idx = get_edge_indices(self.n_nodes, src, dst)

        if self.neg_sampling_mode == "full":
            return self.full_sample(pos_edges_idx)
        if self.neg_sampling_mode == "bipartitle":
            return self.bipartitle_sample(pos_edges_idx)
        elif self.neg_sampling_mode == "visible":
            raise NotImplementedError
        elif self.neg_sampling_mode == "matched":
            return self.matched_sample(
                pos_edges_idx, self.node_group[src], self.node_group[dst]
            )
        else:
            raise ValueError(
                f"Unsupported neg_sampling_mode: {self.neg_sampling_mode}"
            )

    def negative_sample_by_candicate_indices(
        self,
        pos_edge_idx: torch.Tensor,
        src_candidates: torch.Tensor,
        dst_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        通过src_candidates和dst_candidates进行负采样，生成不与现有边重复的负样本。
        从src_candidates和dst_candidates中随机采样，生成负样本对，并检查这些负样本对是否在现有的边集（`all_edges_idx`）中存在。
        如果不存在，则将其视为有效的负样本，直到生成指定数量的负样本。
        对于neg_sampling_mode == "full"，src_candidates和dst_candidates是相同的
        参数：
        src_candidates: torch.Size([x])可能的起始节点集合。
        dst_candidates: torch.Size([x])可能的目标节点集合。
        返回：
        一个形状为 (num_samples, num_neg_per_pos, 2) 的张量
        """
        collected_neg_samples, num_all_neg_samples = [], 0
        n_neg_sample = self.num_neg_per_pos * pos_edge_idx.shape[0]
        while num_all_neg_samples < n_neg_sample:
            src_n = src_candidates[
                torch.randint(
                    0,
                    len(src_candidates),
                    (n_neg_sample,),
                    device=src_candidates.device,
                )
            ]
            dst_n = dst_candidates[
                torch.randint(
                    0,
                    len(dst_candidates),
                    (n_neg_sample,),
                    device=dst_candidates.device,
                )
            ]

            # 去掉自环
            mask = src_n != dst_n
            src_n, dst_n = src_n[mask], dst_n[mask]

            idx_n = get_edge_indices(self.n_nodes, src_n, dst_n)
            mask = ~torch.isin(idx_n, self.all_edges_idx)

            valid_neg_samples = torch.stack([src_n[mask], dst_n[mask]], dim=1)
            collected_neg_samples.append(valid_neg_samples)
            num_all_neg_samples += valid_neg_samples.shape[0]
            logger.info(
                f"Generated {num_all_neg_samples}/{n_neg_sample} "
                "valid neg samples"
            )

        return torch.cat(collected_neg_samples, dim=0)[:n_neg_sample].reshape(
            -1, self.num_neg_per_pos, 2
        )

    def full_sample(self, pos_edge_idx: torch.Tensor) -> torch.Tensor:
        """
        full 负采样模式。
        """
        # TODO: 理清逻辑
        all_nodes = torch.arange(self.n_nodes, device=pos_edge_idx.device)
        return self.negative_sample_by_candicate_indices(
            pos_edge_idx, all_nodes, all_nodes
        )

    def bipartitle_sample(self, pos_edges_idx: torch.Tensor) -> torch.Tensor:
        """
        bipartitle 负采样模式。
        # NOTE: 当加入feature之间的边时，理论上无法采到feature之间的负边
        """
        src_candidates = torch.arange(
            self.n_cells, device=pos_edges_idx.device
        )
        dst_candidates = torch.arange(
            self.n_cells, self.n_nodes, device=pos_edges_idx.device
        )
        return self.negative_sample_by_candicate_indices(
            pos_edges_idx, src_candidates, dst_candidates
        )

    def negative_sampling_visible(self):
        """
        visible 负采样模式。
        """
        pass
        # if self.neg_sampling_restrict is None:
        #     raise ValueError(
        #         "neg_sampling_restrict must be provided "
        #         "when neg_sampling_mode is 'visible'"
        #     )
        # logger.info(f"neg_sampling_restrict: {self.neg_sampling_restrict}")
        # collected_neg_samples = []
        # for bi, fi in self.neg_sampling_restrict:
        #     logger.info(f"Restricting to batch {bi} and feature {fi}")
        #     src_candidates = self.nodes[self.nodes[:, 0] == bi][:, 1]
        #     dst_candidates = self.nodes[self.nodes[:, 0] == fi][:, 1]
        #     neg_samples_i = self.negative_sample_by_candicate_indices(
        #         src_candidates, dst_candidates
        #     )
        #     neg_samples_i = neg_samples_i[
        #         : self.total_neg_samples // len(self.neg_sampling_restrict)
        #     ]
        #     collected_neg_samples.append(neg_samples_i)
        # neg_samples = torch.cat(collected_neg_samples, dim=0)
        # return neg_samples.reshape(-1, self.num_neg_per_pos, 2)

    def matched_sample(
        self,
        pos_edges_idx: torch.Tensor,
        src_group: torch.Tensor,
        dst_group: torch.Tensor,
    ) -> torch.Tensor:
        """
        matched 负采样模式。
        """
        neg_samples = torch.empty(
            (pos_edges_idx.shape[0], self.num_neg_per_pos, 2),
            dtype=torch.long,
            device=pos_edges_idx.device,
        )
        logger.info(f"neg_sampling_restrict: {self.neg_sampling_restrict}")

        for bi, fi in self.neg_sampling_restrict:
            logger.info(f"Restricting to batch {bi} and feature {fi}")
            matches = (src_group == bi) & (dst_group == fi)
            count = matches.sum().item()
            logger.info(f"Found {count} edges for restriction {bi}-{fi}")
            if count <= 0:
                # 如果没有出现相关组合的edges，则跳过
                logger.info(f"No edges found for restriction {bi}-{fi}")
                continue

            src_candidates = torch.nonzero(self.node_group == bi)[:, 0]
            dst_candidates = torch.nonzero(self.node_group == fi)[:, 0]
            if len(src_candidates) == 0 or len(dst_candidates) == 0:
                raise ValueError(
                    "No candidates found for batch "
                    f"type {bi} or feature type {fi}"
                )

            # TODO: 理清思路
            neg_samples_i = self.negative_sample_by_candicate_indices(
                pos_edges_idx[matches], src_candidates, dst_candidates
            )
            neg_samples[matches, ...] = neg_samples_i

        return neg_samples
