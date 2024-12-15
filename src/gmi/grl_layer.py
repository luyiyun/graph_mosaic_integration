import torch
import torch.nn as nn
from torch.autograd import Function


class GRlfunc(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        alpha = torch.as_tensor(alpha).to(x.device)
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = -alpha * grad_output
        return grad_input, None
    
class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GRlfunc.apply(x, self.alpha)

