"""
Metrics.
"""
import torch


EPSILON = 1e-15


def binary_mean_iou(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Calculate binary mean intersection over union.

    Args:
        inputs (torch.Tensor): Output from the forward pass.
        targets (torch.Tensor): Labels.

    Returns (torch.Tensor): Intersection over union value.
    """
    output = (inputs > 0).int()
    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)
    intersection = (targets * output).sum()
    union = targets.sum() + output.sum() - intersection
    result = (intersection + EPSILON) / (union + EPSILON)

    return result
