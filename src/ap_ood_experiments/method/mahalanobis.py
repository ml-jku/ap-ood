import logging

import torch
from torch import nn
import wandb
from tqdm import tqdm

from ap_ood_experiments.method import Method
from ap_ood_experiments.utils.dataset_util import dataset_collate_fn


logger = logging.getLogger(__name__)


class MahalanobisMethod(Method):
    requires_aux: bool = False
    
    def __init__(self, device='cpu'):
        super(MahalanobisMethod, self).__init__()
        self.device = device

    def _mean_pooling(self, embeddings, masks):
        embeddings_zero = torch.where(masks.unsqueeze(-1).bool(), embeddings, torch.zeros_like(embeddings))
        return torch.sum(embeddings_zero, dim=1) / torch.sum(masks, dim=1).unsqueeze(-1)

    @torch.no_grad
    def fit(self, id_dataset):
        mean = 0.
        cov = 0.
        
        id_loader = torch.utils.data.DataLoader(
            id_dataset,
            batch_size=256,
            collate_fn=dataset_collate_fn(id_dataset),
        )
        
        for embeddings, input_ids, masks, texts in tqdm(id_loader):
            embeddings = embeddings.float().to(self.device)
            masks = masks.to(self.device)
            B, S, F = embeddings.shape
            mean_pooled_features = self._mean_pooling(embeddings, masks)
            mean += torch.sum(mean_pooled_features, dim=0)

        mean = mean / len(id_dataset)
        self.register_buffer('mean', mean)

        assert list(self.mean.shape) == [F]

        for embeddings, input_ids, masks, texts in tqdm(id_loader):
            embeddings = embeddings.float().to(self.device)
            masks = masks.to(self.device)
            B, S, F = embeddings.shape
            mean_pooled_features = self._mean_pooling(embeddings, masks)
            centered_features = mean_pooled_features - self.mean
            cov += torch.einsum('bf,bg->fg', centered_features, centered_features)

        cov_avg = cov / (len(id_dataset) - 1)
        inv_cov = torch.inverse(cov_avg)
        self.register_buffer('inv_cov', inv_cov)
        assert list(self.inv_cov.shape) == [F, F]

    @torch.no_grad
    def predict(self, embeddings, input_ids, masks, texts):
        B, S, F = embeddings.shape
        mean_pooled = self._mean_pooling(embeddings, masks)
        centered_x = mean_pooled - self.mean
        # return -torch.einsum('bf,bf->b', centered_x, centered_x)
        scores = -torch.einsum('...f,...f->...', torch.einsum('...g,fg->...f', centered_x, self.inv_cov), centered_x)
        scores[torch.argwhere(masks.sum(dim=-1) == 0)] = -1e10
        return scores.cpu()
