import logging

import torch
from torch import nn
import wandb

from ap_ood_experiments.method import Method


logger = logging.getLogger(__name__)


class PerplexityMethod(Method):
    requires_aux: bool = False
    requires_id: bool = False
    
    def __init__(self, task, device='cpu', normalize=True, allow_shape_mismatch=False):
        super(PerplexityMethod, self).__init__()
        self.task = task
        self.lm_head = task.lm_head().to(device)
        self.device = device
        self.normalize = normalize
        self.allow_shape_mismatch = allow_shape_mismatch
        self.warned = False

    @torch.no_grad
    def fit(self, id_dataset):
        pass

    @torch.no_grad
    def predict(self, embeddings, input_ids, masks, texts):
        B, S, F = embeddings.shape
        input_ids = input_ids[:, 1:]  # remove the pad token
        if self.allow_shape_mismatch:
            if embeddings.shape[-1] != self.lm_head.weight.shape[-1]:
                if not self.warned:
                    logger.warning(f"Embedding shape {embeddings.shape[-1]} does not match LM head shape {self.lm_head.weight.shape[-1]}. Using only embeddings up to index {self.lm_head.weight.shape[-1]}")
                    self.warned = True
                embeddings = embeddings[:, :, :self.lm_head.weight.shape[-1]]
        total_log_perplexity = 0.
        for token_pos in range(input_ids.shape[1]):
            logits = self.lm_head(embeddings[:, token_pos])
            log_ps = torch.log_softmax(logits, dim=-1)
            log_predicted_ps = log_ps[range(len(input_ids)), input_ids[:, token_pos]]
            log_predicted_ps = log_predicted_ps * masks[:, token_pos]
            total_log_perplexity += -log_predicted_ps
        if self.normalize:
            total_log_perplexity = total_log_perplexity / masks.sum(dim=-1)

        total_log_perplexity[torch.argwhere(masks.sum(dim=-1) == 0)] = 1e10

        return -total_log_perplexity
