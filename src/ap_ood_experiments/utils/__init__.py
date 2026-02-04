from hashlib import sha256
import random

import numpy as np
import torch



def logmeanexp(beta, tensor, dim, ignore_negative_inf=False, keepdim=False):
    n = torch.tensor(tensor.size(dim))
    if ignore_negative_inf:
        num_neg_inf = torch.sum((torch.isinf(tensor) & (tensor < 0)).to(torch.int), dim=dim)
        n = n - num_neg_inf
    lse = 1/beta * torch.logsumexp(beta * tensor, dim=dim, keepdim=keepdim)
    return lse - 1/beta * torch.log(n)


def set_seed(seed, string=None):
    if string:
        seed = (seed + int.from_bytes(sha256(string.encode('utf-8')).digest(), 'big')) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
