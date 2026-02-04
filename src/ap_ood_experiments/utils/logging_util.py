import numpy as np
import torch
import wandb


def safe_histogram(tensor, num_bins=32):
    """
    Build a W&B histogram while avoiding numpy bin errors on degenerate tensors.
    """
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach()
        if tensor.numel() == 0:
            return None
        tensor = tensor.to('cpu', dtype=torch.float32).view(-1)
        values = tensor.numpy()
    else:
        values = np.asarray(tensor, dtype=np.float32).ravel()
        if values.size == 0:
            return None
    bins = int(min(num_bins, max(1, values.size)))
    try:
        hist, bin_edges = np.histogram(values, bins=bins)
    except ValueError:
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            return None
        values = np.full_like(values, max_val)
        hist, bin_edges = np.histogram(values, bins=bins)
    return wandb.Histogram(np_histogram=(hist, bin_edges))
