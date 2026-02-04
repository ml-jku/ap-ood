from abc import ABC, abstractmethod

from torch import nn

class EmbeddingCreator(nn.Module, ABC):
    def __init__(self):
        super(EmbeddingCreator, self).__init__()
        self.device = 'cpu'
    
    @abstractmethod
    def input_embeddings(self, src_text):
        pass
    
    @abstractmethod
    def output_embeddings(self, src_text):
        pass

    @abstractmethod
    def lm_head(self):
        pass
    
    def to(self, device, dtype=None, non_blocking=False, memory_format=None):
        self.device = device
        return super().to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
    