from typing import Literal
from logging import getLogger

import torch


logger = getLogger(__name__)


class NegativeSampler:
    """
    full: 所有未包含在edges中的边均被认为是负采样的候选，比如：feature-feature，feature-cell
    bipartitle: feature-feature和feature-cell的边被排除，只包含cell-feature的边
    visible: 只有包含在可获得组学矩阵中的负边才被认为是负采样的候选
    matched：对于每一条正边，只有其所在组学矩阵中的负边才被认为是其负采样的候选
    """

    def __init__(
        self,
        # nodes,
        n_nodes: int,
        num_cell: int,
        num_neg_per_pos: int,
        # edges_infor,
        neg_sampling_mode: Literal["full", "bipartitle", "matched", "visible"],
        node_group: torch.Tensor | None = None,
        neg_sampling_restrict: list[tuple[int, int]] | None = None,
    ):
        """
        初始化 NegativeSampler 类的实例。

        参数：
        - nodes: 包含节点信息的张量，第一列是类型，第二列是节点的索引。
        - num_neg_per_pos: 每条正边采多少负边。
        - edges_infor: 包含排序后完整边信息的张量。
        - neg_sampling_mode: 负采样的模式（"full","bipartitle"， "matched", "visible"）。
        - neg_sampling_restrict: 类型约束，用于 "matched" 和 "visible" 模式。
        """
        if neg_sampling_mode == "matched":
            assert (
                neg_sampling_restrict is not None
            ), "neg_sampling_restrict must be provided when neg_sampling_mode is 'matched'"
            assert (
                node_group is not None
            ), "node_group must be provided when neg_sampling_mode is 'matched'"
            assert node_group.shape[0] == n_nodes, "XXXXX"

        # self.nodes = nodes
        self.node_group = node_group  # 所有节点的type，包含batch和mod信息
        self.n_cell_node = num_cell
        self.num_neg_per_pos = num_neg_per_pos
        # self.edges_infor = edges_infor
        self.neg_sampling_mode = neg_sampling_mode
        self.neg_sampling_restrict = neg_sampling_restrict

        self.n_nodes = n_nodes
        # self.total_neg_samples = edges_infor.shape[0] * num_neg_per_pos
        # self.all_edges_idx = torch.minimum(
        #     edges_infor[:, 0], edges_infor[:, 1]
        # ) * self.n_nodes + torch.maximum(edges_infor[:, 0], edges_infor[:, 1])

    def negative_sampling(self, edges: torch.Tensor) -> torch.Tensor:
        """
        根据不同的负采样模式执行负采样操作。

        返回：
        - 负样本的张量。
        """
        src, dst = edges.T
        pos_edges_idx = torch.minimum(src, dst) * self.n_nodes + torch.maximum(
            src, dst
        )
        # total_neg_samples = edges_infor.shape[0] * self.num_neg_per_pos

        if self.neg_sampling_mode == "full":
            return self.negative_sampling_full(pos_edges_idx)
        if self.neg_sampling_mode == "bipartitle":
            return self.negative_sampling_bipartitle(pos_edges_idx)
        elif self.neg_sampling_mode == "visible":
            raise NotImplementedError
        elif self.neg_sampling_mode == "matched":
            return self.negative_sampling_matched(
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

            idx_n = torch.minimum(src_n, dst_n) * self.n_nodes + torch.maximum(
                src_n, dst_n
            )
            mask = ~torch.isin(idx_n, pos_edge_idx)

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

    def negative_sampling_full(
        self, pos_edge_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        full 负采样模式。
        """
        # TODO: 理清逻辑
        all_nodes = torch.arange(self.n_nodes, device=pos_edge_idx.device)
        return self.negative_sample_by_candicate_indices(
            pos_edge_idx, all_nodes, all_nodes
        )

    def negative_sampling_bipartitle(
        self, pos_edges_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        bipartitle 负采样模式。
        # NOTE: 当加入feature之间的边时，理论上无法采到feature之间的负边
        """
        src_candidates = torch.arange(
            self.n_cell_node, device=pos_edges_idx.device
        )
        dst_candidates = torch.arange(
            self.n_cell_node, self.n_nodes, device=pos_edges_idx.device
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

    def negative_sampling_matched(
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
