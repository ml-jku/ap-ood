import logging

import torch
from torch import nn
from tqdm import tqdm

from ap_ood_experiments import EmbeddingType
from ap_ood_experiments.method import Method


logger = logging.getLogger(__name__)


# Implementation of H_{S-IW} in "Uncertainity Estimation in Autoregressive Structured Prediction"
class EntropyMethod(Method):
    requires_aux: bool = False
    requires_id: bool = False
    
    def __init__(self, n_generations, n_batch_generations, task, device='cpu', normalize=True, T=1.):
        super(EntropyMethod, self).__init__()
        self.device = device
        self.T = T
        self.n_generations = n_generations
        self.n_batch_generations = n_batch_generations
        self.normalize=normalize
        self.task = task
        self.task.to(self.device)
        self.output_ctx_len = task.output_ctx_len


    @torch.no_grad()
    def fit(self, id_dataset):
        pass

    @torch.no_grad()
    def predict(self, embeddings, input_ids, masks, texts):
        entropies = []
        for input_id, mask in zip(tqdm(input_ids), masks):
            n_generated = 0
            log_probs_all = []
            while n_generated < self.n_generations:
                num_return_sequences = min(self.n_batch_generations, self.n_generations - n_generated)
                sequences, scores = self.task.generate(input_id, mask, n_generations=num_return_sequences)

                log_probs = [0.0] * num_return_sequences

                active_sequences = [True] * num_return_sequences
                sequence_lengths = [0] * num_return_sequences

                for s, score in enumerate(scores):
                    if s >= self.output_ctx_len - 1:
                        break
                    token_log_probs = torch.nn.functional.log_softmax(score, dim=-1)
                    token_ids = sequences[:, s + 1]

                    for i in range(num_return_sequences):
                        if active_sequences[i]:
                            token_id = token_ids[i].item()
                            log_prob = token_log_probs[i, token_id].item()
                            log_probs[i] += log_prob
                            sequence_lengths[i] += 1

                            if token_id == self.task.eos_token_id:
                                active_sequences[i] = False

                log_probs_all.extend(log_probs)

                n_generated += num_return_sequences
        
            log_probs_all = torch.tensor(log_probs_all)
            sequence_lengths = torch.tensor(sequence_lengths)
            pi = torch.softmax(1 / self.T * log_probs_all, dim=0)
            entropy = -torch.sum(pi / sequence_lengths * log_probs_all)
            entropies.append(entropy)
        
        return -torch.tensor(entropies)
