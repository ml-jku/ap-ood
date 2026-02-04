import torch
from torch import nn


def logmeanexp(beta, tensor, dim, ignore_negative_inf=False, keepdim=False):
    n = torch.tensor(tensor.size(dim))
    if ignore_negative_inf:
        num_neg_inf = torch.sum((torch.isinf(tensor) & (tensor < 0)).to(torch.int), dim=dim)
        n = n - num_neg_inf
    lse = 1/beta * torch.logsumexp(beta * tensor, dim=dim, keepdim=keepdim)
    return lse - 1/beta * torch.log(n)


class APOOD(nn.Module):
    """
    AP-OOD: Attention Pooling for Out-of-Distribution Detection

    :param feature_dim: Feature dimension of the input (D).
    :param n_heads: Number of heads (M).
    :param n_queries: Number of queries (T).
    :param beta: Beta (inverse temperature) for the softmax. If None, then 1/sqrt(D).
    :param return_features: If true, return features (Shape: batch_size x n_heads). Otherwise, return the loss.
    :param similarity: Similarity metric to use ('dot' or 'euclidean').
    :param init_std: Standard deviation for initialization. If 'feature_dim', then 1/sqrt(D).
    :param normalization: Whether to use the normalization term in the loss.
    """
    def __init__(
        self,
        feature_dim,
        n_heads,
        n_queries=1,
        beta=1,
        return_features=False,
        similarity='dot',
        init_std=1.,
        normalization=True,
    ):
        super(APOOD, self).__init__()
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.normalization = normalization
        if beta is not None:
            self.beta = beta
        else:
            self.beta = 1 / torch.sqrt(torch.tensor(feature_dim))
        self.return_features = return_features
        assert similarity in ('dot', 'euclidean')
        self.similarity = similarity
        if init_std == 'feature_dim':
            self.init_std = 1. / torch.sqrt(torch.tensor(feature_dim, dtype=torch.float))
        else:
            self.init_std = init_std

        self.params = nn.Parameter(torch.zeros(self.n_heads, n_queries, feature_dim))  # H, Q, F
        self._init_weights()
        self.register_buffer('_mean', torch.zeros(self.n_heads))
        self.register_buffer('_cum_log_Z', -torch.ones(self.n_heads) * float('inf'))

    def _init_weights(self):
        self.params.data.normal_(mean=0., std=self.init_std)

    def reset_mean(self):
        device = self._mean.device
        self._mean = torch.zeros(self.n_heads).to(device)
        self._cum_log_Z = -torch.ones(self.n_heads) * float('inf')
        self._cum_log_Z = self._cum_log_Z.to(device)

    def _masked_softmax_pooling(self, sims, masks):
        H, B, Q, S = sims.shape
        assert list(masks.shape) == [B, S]
        masks = masks.reshape(1, B, 1, S)
        sims = self.beta * sims
        sims_for_lse = torch.masked_fill(sims, ~masks.bool(), -torch.inf)
        log_Z = torch.logsumexp(sims_for_lse, dim=(-2, -1), keepdim=True)
        p = (sims_for_lse - log_Z).exp_()
        sims.masked_fill_(~masks.bool(), 0.)
        pooled = torch.einsum('hbqs,hbqs->hb', sims, p)

        return pooled, log_Z.flatten(-3, -1)  # H, B

    def _pairwise_similarity(self, x, y, x_mask=None, y_mask=None):
        """
        Pairwise similarity of two tensors with rank 3.
        
        :param x: shape ``(A, I, feature_dim)``, e.g. ``(n_heads, n_queries, feature_dim)``
        :param y: shape ``(B, J, feature_dim)``, e.g. ``(batch_size, sequence_len, feature_dim)``
        
        :return: Pairwise similarities; shape ``(A, B, I, J)``, e.g. ``(n_heads, batch_size, n_queries, sequence_len)``
        """
        if x_mask is None:
            x_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        if y_mask is None:
            y_mask = torch.ones(y.shape[0], y.shape[1], dtype=torch.bool, device=y.device)
        B_x, T_x = x_mask.shape
        B_y, T_y = y_mask.shape
        if self.similarity == 'dot':
            x_flat = x[x_mask]
            y_flat = y[y_mask]
            sims = torch.einsum('af,bf->ab', x_flat, y_flat)
            full_sims = torch.zeros(B_x * T_x, B_y * T_y, dtype=sims.dtype, device=sims.device)
            x_indices = torch.nonzero(x_mask.flatten()).squeeze()
            y_indices = torch.nonzero(y_mask.flatten()).squeeze()
            full_sims[x_indices[:, None], y_indices[None, :]] = sims
            return full_sims.view(B_x, T_x, B_y, T_y).transpose(1, 2)
        elif self.similarity == 'euclidean':
            return -1/2 * torch.einsum('aif,aif->ai', x, x).reshape([x.shape[0], 1, x.shape[1], 1]) + torch.einsum('aif,bjf->abij', x, y) - 1/2 * torch.einsum('bjf,bjf->bj', y, y).reshape([1, y.shape[0], 1, y.shape[1]])

    @torch.no_grad()
    def partial_fit_mean(self, x, mask):
        sims = self._pairwise_similarity(self.params, x, None, mask)  # H, B, Q, S

        features, log_Z = self._masked_softmax_pooling(sims, mask)

        batch_mean, batch_log_Z = self._compute_batch_mean(features, log_Z)
        batch_log_Z = batch_log_Z.flatten(-2, -1)
        p = torch.sigmoid(self._cum_log_Z - batch_log_Z)
        mean = p * self._mean + (1 - p) * batch_mean
        cum_log_Z = torch.logaddexp(self._cum_log_Z, batch_log_Z)
        self._mean = mean
        self._cum_log_Z = cum_log_Z

    def _compute_batch_mean(self, features, log_Z):
        full_log_Z = torch.logsumexp(log_Z, dim=-1, keepdim=True)
        log_p = log_Z - full_log_Z
        p = torch.exp(log_p)
        mean = torch.einsum('hb,hb->h', features, p)
        return mean, full_log_Z

    def normalizer(self):
        return torch.log(torch.einsum('hqf,hqf->h', self.params, self.params))

    @property
    @torch.no_grad()
    def mean(self):
        if torch.any(torch.isinf(self._cum_log_Z)):
            raise ValueError('No mean fitted. When in eval mode, please fit the mean first by supplying your data to partial_fit_mean')
        return self._mean

    def features(self, x, mask, use_for_mean=None):
        if len(x.shape) != 3:
            raise ValueError('Input tensor must have shape (batch_size, sequence_len, feature_dim)')
        
        B, S, F = x.shape
        H, Q, _ = self.params.shape

        if F != self.feature_dim:
            raise ValueError(f'Feature dimension mismatch: {x.shape[-1]} != {self.feature_dim}')
        if mask is not None and mask.shape != (B, S):
            raise ValueError(f'Mask shape must match input tensor shape excluding the last dimension, got {mask.shape} and {x.shape[:-1]}')

        if mask is None:
            mask = torch.ones(x.shape[:-1], device=x.device, dtype=torch.int)

        if use_for_mean is None:
            # use all values in x for mean computation
            use_for_mean = torch.ones([len(x)])

        # Compute pairwise similarity between parameters and x.
        sims = self._pairwise_similarity(self.params, x, None, mask)  # H, B, Q, S
        assert sims.shape == (H, B, Q, S)

        # Compute the softmax-pooled sequence representations
        features, log_Z = self._masked_softmax_pooling(sims, mask)

        if self.training:
            mean_features = features[:, use_for_mean.bool()]
            mean_log_Z = log_Z[:, use_for_mean.bool()]
            mean, _ = self._compute_batch_mean(mean_features, mean_log_Z)
            mean = mean.unsqueeze(-1)
        else:
            mean = self.mean.unsqueeze(1)  # H, 1

        centered_features = features - mean  # H, B

        return centered_features.transpose(0, 1)  # B, H

    def forward(self, x, mask=None, use_for_mean=None):
        centered_features = self.features(x, mask, use_for_mean=use_for_mean)

        if self.return_features:
            return centered_features

        squared_error = centered_features**2  # B, H
        mean_squared_error = torch.mean(squared_error, dim=1)  # B

        if self.normalization:
            mean_squared_error -= torch.mean(self.normalizer())

        return mean_squared_error
