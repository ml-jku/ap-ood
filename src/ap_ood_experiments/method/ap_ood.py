import logging
import itertools

from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from tqdm import tqdm
import wandb

from ap_ood_experiments.method import Method
from ap_ood_experiments.utils.dataset_util import dataset_collate_fn
from ap_ood_experiments.utils.logging_util import safe_histogram


logger = logging.getLogger(__name__)

class APOODMethod(Method):
    requires_aux: bool = False
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
        normalization=False,
        standardize_inputs=False,
        init_std=1.,
        static_batch=False,
        num_workers=16,
        max_mean_batches=None,
    ):
        super(APOODMethod, self).__init__()
        self.device = device
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.n_features = n_features
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.standardize_inputs = standardize_inputs
        self.static_batch = static_batch
        self.num_workers = num_workers
        self.max_mean_batches = max_mean_batches
            
        self.score_fns = {'mean': self.predict_mean, 'max': self.predict_max}

        self.detector = detector(
            feature_dim=n_features,
            n_heads=n_heads,
            n_queries=n_queries,
            beta=beta,
            similarity=similarity,
            normalization=normalization,
            init_std=init_std,
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
        total_batches = len(id_loader)
        if self.max_mean_batches is not None:
            total_batches = min(total_batches, self.max_mean_batches)
        logger.info('Fitting standard scaler')
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
    
    def fit(self, id_dataset):
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

        if self.static_batch:
            embeddings, input_ids, masks, texts = next(iter(id_loader))
            embeddings = embeddings.to(self.device).float()
            masks = masks.to(self.device)
            id_loader = itertools.repeat((embeddings, input_ids, masks, texts))

        total_steps = 0
        pbar = tqdm(total=self.n_steps)

        while total_steps < self.n_steps:
            for embeddings, input_ids, masks, texts in id_loader:
                if total_steps >= self.n_steps:
                    break
                embeddings = embeddings.to(self.device).float()  # B, S, F
                masks = masks.to(self.device)  # B, S

                if self.standardize_inputs:
                    embeddings = (embeddings - self.mean) / (self.std + 1e-10)

                ood_scores = self.detector(embeddings, masks)  # B
                total_loss = torch.mean(ood_scores, dim=0)

                normalizer = torch.mean(self.detector.normalizer())

                variance = torch.mean(ood_scores + normalizer, dim=0)

                if self.log_training:
                    ood_scores_tensor = ood_scores.detach()
                    wandb.log({
                        'total_loss': total_loss.item(),
                        'lr': optimizer.param_groups[0]['lr'],
                        'normalizer': normalizer.item(),
                        'variance': variance.item(),
                        'ood_scores': safe_histogram(ood_scores_tensor),
                    })

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                total_steps = total_steps + 1
                pbar.update()

        self._compute_mean(id_dataset)

    @torch.no_grad()
    def predict_mean(self, embeddings, input_ids, masks, texts):
        self.eval()
        self.detector.eval()

        if self.standardize_inputs:
            embeddings = (embeddings - self.mean) / (self.std + 1e-10)

        score = self.detector(embeddings, masks)
        score[torch.argwhere(masks.sum(dim=-1) == 0)] = -1e10
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
        scores, _ = torch.max(scores, dim=-1)
        scores[torch.argwhere(masks.sum(dim=-1) == 0)] = 1e10
        self.detector.return_features = False
        return -scores.cpu()
