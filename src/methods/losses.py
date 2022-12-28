"""Define losses for the recommendation systems to use when training."""
import torch


def BPR_Loss(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Compute Bayesian personalized ranking loss from + and - samples.

    Args:
        positive: A tensor of positive samples.
        negative: A tensor of negative samples.

    Returns:
        The Bayesian personalized ranking loss.
    """
    distances = positive - negative
    loss = -torch.sum(torch.log(torch.sigmoid(distances)), 0, keepdim=True)

    return loss
