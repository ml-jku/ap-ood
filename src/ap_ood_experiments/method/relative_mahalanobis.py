import logging

import torch
from torch import nn
import wandb

from ap_ood_experiments.method import Method
from ap_ood_experiments.utils.dataset_util import dataset_collate_fn


logger = logging.getLogger(__name__)


class RelativeMahalanobisMethod(Method):
    requires_aux: bool = True
    
    def __init__(self, device='cpu'):
        super(RelativeMahalanobisMethod, self).__init__()
        self.device = device

    def _mean_pooling(self, embeddings, masks):
        embeddings_zero = torch.where(masks.unsqueeze(-1).bool(), embeddings, torch.zeros_like(embeddings))
        return torch.sum(embeddings_zero, dim=1) / torch.sum(masks, dim=1).unsqueeze(-1)

    @torch.no_grad()
    def fit_dataset(self, dataset):
        mean = 0.
        cov = 0.
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            collate_fn=dataset_collate_fn(dataset),
        )
        
        for embeddings, input_ids, masks, texts in loader:
            embeddings = embeddings.float().to(self.device)
            masks = masks.to(self.device)
            B, S, F = embeddings.shape
            mean_pooled_features = self._mean_pooling(embeddings, masks)
            mean += torch.sum(mean_pooled_features, dim=0)
            # second_moment += torch.einsum('bf,bg->fg', mean_pooled_features, mean_pooled_features)
        
        mean = mean / len(dataset)
        # self.mean = mean / len(mean_pooled_features)
        
        assert list(mean.shape) == [F]
        
        for embeddings, input_ids, masks, texts in loader:
            embeddings = embeddings.float().to(self.device)
            masks = masks.to(self.device)
            B, S, F = embeddings.shape
            mean_pooled_features = self._mean_pooling(embeddings, masks)
            centered_features = mean_pooled_features - mean
            cov += torch.einsum('bf,bg->fg', centered_features, centered_features)

        cov_avg = cov / (len(dataset) - 1)
        inv_cov = torch.inverse(cov_avg)
        assert list(inv_cov.shape) == [F, F]
        return mean, inv_cov
        
    @torch.no_grad()
    def fit(self, id_dataset, aux_dataset):
        mean_id, inv_cov_id = self.fit_dataset(id_dataset)
        mean_aux, inv_cov_aux = self.fit_dataset(aux_dataset)
        self.register_buffer('mean_id', mean_id)
        self.register_buffer('inv_cov_id', inv_cov_id)
        self.register_buffer('mean_aux', mean_aux)
        self.register_buffer('inv_cov_aux', inv_cov_aux)

    @torch.no_grad()
    def predict(self, embeddings, input_ids, masks, texts):
        B, S, F = embeddings.shape
        mean_pooled = self._mean_pooling(embeddings, masks)
        centered_x_id = mean_pooled - self.mean_id
        # return -torch.einsum('bf,bf->b', centered_x, centered_x)
        id_score = -torch.einsum('...f,...f->...', torch.einsum('...g,fg->...f', centered_x_id, self.inv_cov_id), centered_x_id)
        centered_x_aux = mean_pooled - self.mean_aux
        aux_score = -torch.einsum('...f,...f->...', torch.einsum('...g,fg->...f', centered_x_aux, self.inv_cov_aux), centered_x_aux)
        return id_score - aux_score
