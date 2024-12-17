from typing import Literal

import torch
import torch.nn.functional as F


def softmax_weighted_loss(
    pos_score: torch.Tensor,
    neg_scores: torch.Tensor,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if edge_weights is None:
        raise ValueError(
            "edge_weights must be provided for 'weighted_softmax' loss."
        )

    # 合并正样本分数和负样本分数
    all_scores = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)

    # 计算分母 sum(exp(s_{e'}))
    denominator = torch.logsumexp(all_scores, dim=1)

    # 计算 softmax 损失
    softmax_loss = denominator - pos_score

    # 加权
    if edge_weights is not None:
        softmax_loss = edge_weights * softmax_loss

    return softmax_loss.mean()


def margin_ranking_loss(
    pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    pos_scores_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores)
    return F.margin_ranking_loss(
        pos_scores_expanded, neg_scores, torch.ones_like(neg_scores), margin
    )


def domain_classification_loss(
    domain_preds: torch.Tensor,
    domain_labels: torch.Tensor,
    label_smoothing: float = 0.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if weights is None:
        return F.cross_entropy(
            domain_preds, domain_labels, label_smoothing=label_smoothing
        )

    loss = F.cross_entropy(
        domain_preds,
        domain_labels,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    loss = (loss * weights).sum() / weights.sum()
    return loss


def graph_mosaic_integration_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    edge_weights: torch.Tensor | None = None,
    domain_preds: torch.Tensor | None = None,
    domain_labels: torch.Tensor | None = None,
    discriminate_weights: torch.Tensor | None = None,
    edge_loss_type: Literal[
        "weighted_softmax", "margin_ranking"
    ] = "weighted_softmax",
    loss_alpha: float = 0.2,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if edge_loss_type == "weighted_softmax":
        edge_loss = softmax_weighted_loss(pos_scores, neg_scores, edge_weights)
    elif edge_loss_type == "margin_ranking":
        edge_loss = margin_ranking_loss(pos_scores, neg_scores)
    else:
        raise ValueError(f"Unsupported edge loss type: {edge_loss_type}")

    total_loss = edge_loss
    if domain_preds is not None and domain_labels is not None:
        domain_loss = domain_classification_loss(
            domain_preds, domain_labels, label_smoothing, discriminate_weights
        )
        total_loss += loss_alpha * domain_loss

        return total_loss, {
            "edge": edge_loss,
            "domain": domain_loss,
            "total": total_loss,
        }
    elif domain_preds is not None or domain_labels is not None:
        raise ValueError(
            "Both domain_preds and domain_labels "
            "must be provided for domain classification."
        )

    return total_loss, {
        "edge": edge_loss,
        "total": total_loss,
    }
