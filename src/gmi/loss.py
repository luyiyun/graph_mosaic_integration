import torch
import torch.nn as nn
import torch.nn.functional as F



class LossType(nn.Module):
    def __init__(self, loss_type="margin_ranking", margin=1.0):
        super(LossType, self).__init__()
        self.loss_type = loss_type
        self.margin = margin

    def softmax_weighted_loss(self, pos_score, neg_scores, edge_weights=None):
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

    def margin_ranking_loss(self, pos_scores, neg_scores):

        loss_fn = nn.MarginRankingLoss(margin=self.margin)
        pos_scores_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores)
        labels = torch.ones_like(neg_scores)
        return loss_fn(pos_scores_expanded, neg_scores, labels)

    def forward(self, pos_scores, neg_scores, edge_weights=None):
        if self.loss_type == "margin_ranking":
            return self.margin_ranking_loss(pos_scores, neg_scores)
        elif self.loss_type == "weighted_softmax":
            return self.softmax_weighted_loss(pos_scores, neg_scores, edge_weights)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")


def compute_loss(pos_scores, neg_scores, pos_domain_preds=None, domain_labels=None, edge_weights=None, loss_fn=None,loss_alpha=0.2,label_smoothing=0.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_fn is None:
        raise ValueError("loss_fn must be provided.")
    # 边的损失
    if loss_fn.loss_type == "weighted_softmax":
        edge_loss = loss_fn(pos_scores, neg_scores, edge_weights)
    else:
        edge_loss = loss_fn(pos_scores, neg_scores)

    domain_loss = None
    if pos_domain_preds is None and domain_labels is None:
        # 如果两个都为 None，只使用边的损失
        total_loss = edge_loss
    elif pos_domain_preds is None  or domain_labels is None:
        # 如果其中一个是 None，但剩下的不是，抛出错误
        raise ValueError("You need to provide all of pos_domain_preds, neg_domain_preds, and domain_labels at the same time.")
    else:

        domain_loss = F.cross_entropy(pos_domain_preds, domain_labels,label_smoothing=label_smoothing)#smooth
        # 总损失是边的损失和领域分类损失之和
        total_loss = edge_loss + loss_alpha *domain_loss


    return total_loss, {"edge": edge_loss, "domain": domain_loss, "total": total_loss}