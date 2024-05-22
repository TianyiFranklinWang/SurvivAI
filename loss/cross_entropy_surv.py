import torch
from torch import nn


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (
            torch.log(torch.gather(S_padded, 1, Y) + eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(
        1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1 - alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class CrossEntropySurvLoss(nn.Module):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)
