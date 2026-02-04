import itertools
import logging

import torch
from tqdm import tqdm
import wandb

from ap_ood_experiments.method import Method
from ap_ood_experiments.utils.dataset_util import dataset_collate_fn
from ap_ood_experiments.utils.logging_util import safe_histogram


logger = logging.getLogger(__name__)

class APOODOEMethod(Method):
    requires_aux: bool = True
    requires_additional_data: bool = False
    
    def __init__(
        self,
        n_heads,
        n_queries,
        n_features,
        n_steps,
        batch_size,
        detector,
        beta=None,
        optimizer=None,
        log_training=True,
        device='cpu',
        similarity='dot',
        use_scheduler=True,
        lamb=1.,
        average_dimensions=True,
        normalization=False,
        standardize_inputs=False,
        init_std=1.,
        static_batch=False,
        num_workers=16,
        max_mean_batches=None,
    ):
        super(APOODOEMethod, self).__init__()
        self.score_fns = {'mean': self.predict_mean, 'max': self.predict_max}
        self.device = device
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.n_features = n_features
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.lamb = lamb
        self.average_dimensions = average_dimensions
        self.standardize_inputs = standardize_inputs
        self.static_batch = static_batch
        self.num_workers = num_workers
        self.max_mean_batches = max_mean_batches

        self.detector = detector(
            feature_dim=n_features,
            n_heads=n_heads,
            n_queries=n_queries,
            beta=beta,
            similarity=similarity,
            init_std=init_std,
            normalization=normalization,
        )

        self.use_scheduler = use_scheduler
        self.log_training = log_training
        self.to(self.device)
        self.optimizer = optimizer(self.parameters())
    
    def _compute_mean(self, id_dataset):
        loader = torch.utils.data.DataLoader(
            id_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=False,
            collate_fn=dataset_collate_fn(id_dataset),
        )
        logger.info('Computing mean value')
        total_batches = len(loader)
        if self.max_mean_batches is not None:
            total_batches = min(total_batches, self.max_mean_batches)
        with torch.no_grad():
            for i, (embeddings, input_ids, masks, texts) in enumerate(tqdm(loader, total=total_batches)):
                embeddings = embeddings.to(self.device).float()
                masks = masks.to(self.device)
                if self.standardize_inputs:
                    embeddings = (embeddings - self.mean) / (self.std + 1e-10)
                self.detector.partial_fit_mean(embeddings, masks)
                if self.max_mean_batches is not None and (i + 1) >= self.max_mean_batches:
                    break

    @torch.no_grad()
    def fit_standard_scaler(self, id_dataset):
        mean = torch.zeros(self.n_features, device=self.device)
        var = torch.zeros(self.n_features, device=self.device)
        n = 0
        id_loader = torch.utils.data.DataLoader(
            id_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=False,
            collate_fn=dataset_collate_fn(id_dataset),
        )
        logger.info('Fitting standard scaler')
        total_batches = len(id_loader)
        if self.max_mean_batches is not None:
            total_batches = min(total_batches, self.max_mean_batches)
        for i, (embeddings, input_ids, masks, texts) in enumerate(tqdm(id_loader, total=total_batches)):
            embeddings = embeddings.to(self.device).float()
            masks = masks.to(self.device)
            embeddings = embeddings[masks.bool()]
            mean += torch.sum(embeddings, dim=0)
            var += torch.sum(embeddings ** 2, dim=0)
            n += embeddings.shape[0]
            if self.max_mean_batches is not None and (i + 1) >= self.max_mean_batches:
                break
        mean /= n
        var = var / n - mean ** 2
        std = torch.sqrt(var)
        self.mean = mean
        self.std = std

    def fit(self, id_dataset, aux_dataset):
        self.train()

        if self.standardize_inputs:
            self.fit_standard_scaler(id_dataset)

        optimizer = self.optimizer

        if self.use_scheduler:
            # instantiate CosineAnnealing scheduler. The number of steps is ``len(loader) * n_epochs``.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_steps)
        else:
            scheduler = None

        logger.info('Fitting parameters')

        if len(aux_dataset) < len(id_dataset):
            # if the auxiliary dataset is smaller than the id dataset, we need to repeat the auxiliary dataset
            # so that it has at least the same size as the id dataset
            aux_dataset = torch.utils.data.ConcatDataset([aux_dataset] * (len(id_dataset) // len(aux_dataset) + 1))

        num_workers = self.num_workers if not self.static_batch else 0
        persistent_workers = num_workers > 0

        id_loader = torch.utils.data.DataLoader(
            id_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=dataset_collate_fn(id_dataset),
        )
        aux_loader = torch.utils.data.DataLoader(
            aux_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=dataset_collate_fn(aux_dataset),
        )

        if self.static_batch:
            id_batch = next(iter(id_loader))
            aux_batch = next(iter(aux_loader))
            id_loader = itertools.repeat(id_batch)
            aux_loader = itertools.repeat(aux_batch)

        total_steps = 0
        pbar = tqdm(total=self.n_steps)
        while total_steps < self.n_steps:
            for (id_embeddings, _, id_masks, _), (aux_embeddings, _, aux_masks, _) in zip(id_loader, aux_loader):
                if total_steps >= self.n_steps:
                    break
                id_embeddings = id_embeddings.to(self.device).float()  # B, S, F
                id_masks = id_masks.to(self.device)  # B, S
                aux_embeddings = aux_embeddings.to(self.device).float()
                aux_masks = aux_masks.to(self.device)

                if self.standardize_inputs:
                    id_embeddings = (id_embeddings - self.mean) / (self.std + 1e-10)
                    aux_embeddings = (aux_embeddings - self.mean) / (self.std + 1e-10)

                sequence_length = max(id_embeddings.shape[1], aux_embeddings.shape[1])
                if id_embeddings.shape[1] < sequence_length:
                    id_embeddings = torch.nn.functional.pad(id_embeddings, (0, 0, 0, sequence_length - id_embeddings.shape[1]), value=0.)
                    id_masks = torch.nn.functional.pad(id_masks, (0, sequence_length - id_masks.shape[1]), value=0)
                if aux_embeddings.shape[1] < sequence_length:
                    aux_embeddings = torch.nn.functional.pad(aux_embeddings, (0, 0, 0, sequence_length - aux_embeddings.shape[1]), value=0.)
                    aux_masks = torch.nn.functional.pad(aux_masks, (0, sequence_length - aux_masks.shape[1]), value=0)
                embeddings = torch.concat([id_embeddings, aux_embeddings], dim=0)
                masks = torch.concat([id_masks, aux_masks], dim=0)
                y = torch.concat([torch.ones(len(id_embeddings)), torch.zeros(len(aux_embeddings))], dim=0).int()

                scores = self.detector(embeddings, masks, use_for_mean=y.bool())  # B
                
                if not self.average_dimensions:
                    scores = scores * self.detector.n_heads  # detector averages over dimensions

                id_scores = scores[:len(id_embeddings)]
                aux_scores = scores[len(id_embeddings):]
                id_loss = torch.mean(id_scores, dim=0)
                aux_loss = torch.mean(-torch.log(1-torch.exp(-aux_scores)), dim=0)

                total_loss = id_loss + self.lamb * aux_loss

                if self.log_training:
                    log_dict = {
                        'total_loss': total_loss.item(),
                        'id_loss': id_loss.item(),
                        'aux_loss': aux_loss.item(),
                        'lr': optimizer.param_groups[0]['lr'],
                        'ood_scores': safe_histogram(scores.detach()),
                    }
                    if hasattr(self.detector, 'betas'):
                        log_dict['betas'] = safe_histogram(self.detector.betas.detach().float())
                    wandb.log(log_dict)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                total_steps = total_steps + 1
                pbar.update()

        if not getattr(self.detector, 'learn_mean', False):
            self._compute_mean(id_dataset)

    @torch.no_grad()
    def predict_mean(self, embeddings, input_ids, masks, texts):
        self.eval()
        self.detector.eval()

        if self.standardize_inputs:
            embeddings = (embeddings - self.mean) / (self.std + 1e-10)

        score = self.detector(embeddings, masks)
        return -score.cpu()

    @torch.no_grad()
    def predict_max(self, embeddings, input_ids, masks, texts):
        self.eval()
        self.detector.eval()

        if self.standardize_inputs:
            embeddings = (embeddings - self.mean) / (self.std + 1e-10)

        self.detector.return_features = True
        features = self.detector(embeddings, masks)
        squared_distances = features**2
        normalizer = self.detector.normalizer()
        scores = squared_distances - normalizer.unsqueeze(0)

        # when normalizer is infinity, ignore the entry
        scores = torch.where(torch.isinf(normalizer), -torch.inf, scores)

        scores, _ = torch.max(scores, dim=-1)
        self.detector.return_features = False
        return -scores.cpu()

    @torch.no_grad()
    def predict(self, embeddings, input_ids, masks, texts):
        return self.score_fns[self.inference_method](embeddings, input_ids, masks, texts)
