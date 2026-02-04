from abc import ABC, abstractmethod
import logging

import torch
from torch import nn

from ap_ood_experiments.utils.dataset_util import dataset_collate_fn


logger = logging.getLogger(__name__)


class Method(ABC, nn.Module):
    requires_aux: bool
    requires_id: bool = True
    
    def __init__(self):
        super(Method, self).__init__()
        self.score_fns = {'base': self.predict}

    @abstractmethod
    def fit(self, id_dataset, aux_dataset=None):
        pass

    def predict(self, embeddings, input_ids, masks, texts) -> torch.Tensor:
        # Attention! Use input_ids over tokenizing the text, as the tokenization results can be inconsistent
        # with the input_ids (e.g., when there is an "<unk>" token in the text)
        pass


class DummyMethod(Method):
    def __init__(self, verbose, requires_aux=True):
        super(DummyMethod, self).__init__()
        self.verbose = verbose
        self.requires_aux = requires_aux

    def fit(self, id_dataset, aux_dataset=None):
        
        id_loader = torch.utils.data.DataLoader(
            id_dataset,
            batch_size=256,
            collate_fn=dataset_collate_fn(id_dataset),
        )
        if self.requires_aux:
            aux_loader = torch.utils.data.DataLoader(
                aux_dataset,
                batch_size=256,
                collate_fn=dataset_collate_fn(aux_dataset),
            )
        
        id_samples = 0
        for id_batch, _, _, _ in id_loader:
            id_samples += len(id_batch)

        if self.requires_aux:
            aux_samples = 0
            for aux_batch, _, _, _ in aux_loader:
                aux_samples += len(aux_batch)

            if self.verbose:
                logger.info(f'Fitted {id_samples} ID samples and {aux_samples} auxiliary samples')
        else:
            if self.verbose:
                logger.info(f'Fitted {id_samples} ID samples.')

        return self

    def predict(self, embeddings, input_ids, masks, texts) -> torch.Tensor:
        if self.verbose:
            logger.info(f'Predicted {len(embeddings)} samples.')

        return torch.ones(len(embeddings), dtype=torch.float32)
