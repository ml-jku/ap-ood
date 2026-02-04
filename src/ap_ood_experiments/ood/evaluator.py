from collections import defaultdict
from collections.abc import Mapping
import logging
from typing import Dict

import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from ap_ood_experiments.utils.dataset_util import dataset_collate_fn, load_dataset_embeddings, load_dataset_embeddings_ragged

# load_dataset_embeddings = load_dataset_embeddings_ragged


logger = logging.getLogger(__name__)


class OODEvaluator:
    def __init__(self, id_dataset, out_datasets, metrics, logger, batch_size, id_embedding_type, ood_embedding_type, config, device=None, task=None, use_runtime=False):
        self.in_dataset = id_dataset
        self.out_datasets = out_datasets
        self.metrics = metrics
        self.logger = logger
        self.device = device
        self.batch_size = batch_size
        self.id_embedding_type = id_embedding_type
        self.ood_embedding_type = ood_embedding_type
        self.config = config
        self.num_max_samples = config.num_max_eval_samples  # Max samples per dataset to evaluate
        self.task = task
        self.use_runtime = use_runtime

    def _dataset_label(self, dataset_cfg):
       return dataset_cfg['storage']

    def evaluate(self, method, epoch=None, prefix=None):
        if prefix is not None:
            prefix = f'_{prefix}_'
        else:
            prefix = '_'
        results = {}
        metric_results = defaultdict(list)
        with torch.no_grad():
            in_scores = self.compute_scores(self.in_dataset, method, self.id_embedding_type, num_samples=self.num_max_samples)
            for score_name, scores in in_scores.items():
                in_mean_score = torch.mean(scores, dim=0).detach().item()
                results[f'{prefix}{score_name}/mean_id_score'] = in_mean_score
                logger.info(f'{prefix}{score_name}/mean_id_score: {in_mean_score}')
            for out_dataset in self.out_datasets:
                if out_dataset is None:
                    logger.info('Skipping None out_dataset')
                    continue
                out_scores = self.compute_scores(out_dataset, method, self.ood_embedding_type, num_samples=self.num_max_samples)
                dataset_label = self._dataset_label(out_dataset)
                for score_name, scores in out_scores.items():
                    for metric_name, metric in self.metrics.items():
                        metric_result = metric(in_scores[score_name], scores)
                        qualifier = f'{prefix}{score_name}/{metric_name}.{dataset_label}'
                        results[qualifier] = metric_result
                        metric_results[f'{score_name}/{metric_name}'].append(metric_result)
                        logger.info(f'{qualifier}: {metric_result}')
                    
        # Add mean results
        for score_name, scores in out_scores.items():
            for metric_name in self.metrics.keys():
                metric_result = metric_results[f'{score_name}/{metric_name}']
                qualifier = f'{prefix}{score_name}/{metric_name}.mean'
                results[qualifier] = np.mean(metric_result)
            

        if self.logger:
            self.logger.log(results, epoch=epoch)

        return results

    def compute_scores(self, dataset, method, embedding_type, num_samples=10000) -> Dict[float, dict]:
        total_samples = 0
        score_fns = method.score_fns
        scores = {k: [] for k in score_fns.keys()}
        dataset_label = self._dataset_label(dataset)
        dataset = load_dataset_embeddings(
            dataset,
            embedding_type,
            self.config,
            task=self.task,
            use_runtime=self.use_runtime,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=dataset_collate_fn(dataset),
        )
        pbar = tqdm(total=min(len(dataset), num_samples), desc=f'Computing scores ({dataset_label})')
        for embeddings, input_ids, masks, texts in loader:
            embeddings = embeddings.to(self.device).float()
            masks = masks.to(self.device)
            # set masked embeddings to 0
            embeddings = torch.where(masks.unsqueeze(-1).bool(), embeddings, torch.zeros_like(embeddings))
            if total_samples >= num_samples:
                break
            pbar.update(min(len(embeddings), num_samples - total_samples))
            total_samples += len(embeddings)
            for score_name, score_fn in score_fns.items():
                score = score_fn(embeddings, input_ids, masks, texts)
                scores[score_name].append(score)
        return {k: torch.concat(v, dim=0)[:num_samples] for k,v in scores.items()}
