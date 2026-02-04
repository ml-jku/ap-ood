import json
from collections.abc import Mapping
from logging import getLogger
import numpy as np
import os
from pathlib import Path
import shutil
import time

from datasets import load_dataset
from hydra.utils import instantiate
import torch
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from ap_ood_experiments import EmbeddingType


logger = getLogger(__name__)


class CollectionDataset(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.collections = list(args)
        self.length = len(self.collections[0])
        # if not all(len(collection) == self.length for collection in self.collections):
        #    raise ValueError('All collections need to have the same length.')
    
    def __getitem__(self, i):
        return tuple(collection[i] for collection in self.collections)
    
    def __len__(self):
        return self.length


class RaggedDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, input_ids, start_idxs, texts):
        self.embeddings = embeddings
        self.input_ids = input_ids
        self.start_idxs = start_idxs
        self.texts = texts
        sequence_lengths = [(self.start_idxs[i + 1] - self.start_idxs[i]).item() for i in range(len(self.start_idxs) - 1)]
        sequence_lengths.append((len(self.embeddings) - self.start_idxs[-1]).item())
        self.sequence_length = max(sequence_lengths)
        #padding_size = self.sequence_length - sequence_lengths[-1]
        #if padding_size > 0:
        #    self.embeddings = torch.cat([self.embeddings, torch.zeros(padding_size, self.embeddings.shape[1], dtype=self.embeddings.dtype)], dim=0)
        self.length = len(self.start_idxs)
    
    def __getitem__(self, i):
        start_idx = self.start_idxs[i].item()
        end_idx = self.start_idxs[i + 1].item() if i + 1 < self.length else len(self.input_ids)
        mask = torch.zeros(self.sequence_length, dtype=torch.bool)
        mask[:end_idx - start_idx] = 1
        embeddings = self.embeddings[start_idx:start_idx + self.sequence_length]
        if embeddings.shape[0] < self.sequence_length:
            assert start_idx + self.sequence_length > len(self.embeddings)
            padding_size = self.sequence_length - embeddings.shape[0]
            embeddings = torch.cat([embeddings, torch.zeros(padding_size, embeddings.shape[1], dtype=embeddings.dtype)], dim=0)
        assert embeddings.shape[0] == self.sequence_length
        input_ids = torch.zeros(self.sequence_length, dtype=self.input_ids.dtype)
        input_ids[:end_idx - start_idx] = self.input_ids[start_idx:end_idx]
        return embeddings, input_ids, mask, self.texts[i]

    def __len__(self):
        return self.length


def convert_to_ragged_format(embeddings, input_ids, masks):
    B, S, F = embeddings.shape
    embeddings_list = []
    input_ids_list = []
    start_idxs = []
    total_length = 0
    for i in range(B):
        start_idxs.append(total_length)
        new_embeddings = embeddings[i][masks[i].bool()]
        embeddings_list.append(new_embeddings)
        total_length += len(new_embeddings)
        input_ids_list.append(input_ids[i][masks[i].bool()])
    return torch.concat(embeddings_list, dim=0), torch.concat(input_ids_list, dim=0), torch.tensor(start_idxs)


def convert_from_ragged_format(embeddings, input_ids, start_idxs):
    B = len(start_idxs)
    sequence_lenghts = [(start_idxs[i+1]-start_idxs[i]).item() for i in range(B-1)]
    sequence_lenghts.append((len(embeddings)-start_idxs[-1]).item())
    S = max(sequence_lenghts)
    F = embeddings.shape[1]
    embeddings_out = torch.zeros((B, S, F), dtype=embeddings.dtype)
    input_ids_out = torch.zeros((B, S), dtype=input_ids.dtype)
    masks_out = torch.zeros((B, S), dtype=torch.bool)
    for i in tqdm(range(B)):
        start_idx = start_idxs[i].item()
        end_idx = start_idxs[i + 1].item() if i + 1 < B else len(embeddings)
        length = end_idx - start_idx
        embeddings_out[i, :length] = embeddings[start_idx:end_idx]
        input_ids_out[i, :length] = input_ids[start_idx:end_idx]
        masks_out[i, :length] = 1
    return embeddings_out, input_ids_out, masks_out


def load_dataset_preprocess(*args, **kwargs):
    if 'remove' in kwargs:
        removes = kwargs['remove']
        del kwargs['remove']
    else:
        removes = None
    if 'rename' in kwargs:
        renames = kwargs['rename']
        del kwargs['rename']
    else:
        renames = None
    dataset = load_dataset(*args, **kwargs)
    if removes:
        dataset = dataset.remove_columns(removes)
    if renames:
        for rename in renames:
            dataset = dataset.rename_column(**rename)
    return dataset

def _maybe_get_attr(cfg, key, default=None):
    if isinstance(cfg, Mapping) and key in cfg:
        return cfg[key]
    if hasattr(cfg, key):
        return getattr(cfg, key)
    return default


def _get_model_name(config):
    model_cfg = getattr(config, 'model', None)
    model_name = _maybe_get_attr(model_cfg, 'model_name') if model_cfg is not None else None
    if model_name is None:
        raise ValueError('Model configuration must define "model_name" to locate stored embeddings.')
    return model_name


def _sanitize_model_name(name: str) -> str:
    return name.replace('/', '_')


def _get_storage_subpath(dataset_cfg):
    storage = _maybe_get_attr(dataset_cfg, 'storage')
    if storage:
        return Path(storage)
    dataset_path = _maybe_get_attr(dataset_cfg, 'path')
    split = _maybe_get_attr(dataset_cfg, 'split')
    if dataset_path is None:
        raise ValueError('Dataset configuration missing "storage" (preferred) or "path" for embedding storage resolution.')
    sanitized = dataset_path.replace('/', '_')
    if split:
        return Path(sanitized) / split
    return Path(sanitized)


def get_embedding_dataset_dir(dataset_cfg, config):
    embedding_root = os.environ.get('EMBEDDING_ROOT')
    if not embedding_root:
        raise EnvironmentError('Environment variable EMBEDDING_ROOT must be set to locate stored embeddings.')
    model_name = _get_model_name(config)
    model_dir = Path(embedding_root) / _sanitize_model_name(model_name)
    storage_subpath = _get_storage_subpath(dataset_cfg)
    return model_dir / storage_subpath


def get_storage_subpath(dataset_cfg):
    return _get_storage_subpath(dataset_cfg)


def _apply_dataset_filters(dataset, n_data=None, seed=None):
    if n_data:
        if n_data > len(dataset):
            raise ValueError(f'n_data {n_data} is larger than dataset size {len(dataset)}')
        rng = np.random.default_rng(seed)
        subset_idxs = rng.choice(len(dataset), n_data, replace=False)
        dataset = Subset(dataset, subset_idxs)
    return dataset


class RuntimeEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, embedding_type, task):
        if task is None:
            raise ValueError('Runtime embedding computation requires an instantiated task.')

        self.base_dataset = base_dataset
        self.embedding_type = embedding_type
        self.task = task
        self.length = len(self.base_dataset)

        if embedding_type == EmbeddingType.INPUT:
            self.embedding_fn = self.task.input_embeddings
        elif embedding_type == EmbeddingType.OUTPUT:
            self.embedding_fn = self.task.output_embeddings
        else:
            raise ValueError(f'Unsupported embedding type {embedding_type}')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError('Index out of bounds.')
        return self.base_dataset[idx]

    def collate_fn(self, samples):
        batch = default_collate(samples)

        with torch.no_grad():
            texts, input_ids, embeddings, masks = self.embedding_fn(batch)

        if not isinstance(texts, (list, tuple)):
            texts = [texts]

        def _to_cpu_tensor(value, dtype=None):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().to(dtype=dtype) if dtype else value.detach().cpu()
            return torch.as_tensor(value, dtype=dtype)

        embeddings_cpu = _to_cpu_tensor(embeddings)
        input_ids_cpu = _to_cpu_tensor(input_ids, dtype=torch.long)
        if isinstance(masks, torch.Tensor):
            masks_cpu = masks.detach().cpu().bool()
        else:
            masks_cpu = torch.as_tensor(masks, dtype=torch.bool)

        return embeddings_cpu, input_ids_cpu, masks_cpu, texts


def _load_dataset_embeddings_runtime(dataset_cfg, embedding_type, config, task, n_data=None):
    runtime_cfg = _maybe_get_attr(dataset_cfg, 'runtime')
    if runtime_cfg is None:
        raise ValueError('Runtime dataset configuration missing "runtime" entry.')

    dataset = instantiate(runtime_cfg)

    seed = getattr(config, 'seed', 0) + getattr(config, 'addtl_seed', 0)
    dataset = _apply_dataset_filters(dataset, n_data=n_data, seed=seed)

    return RuntimeEmbeddingDataset(
        base_dataset=dataset,
        embedding_type=embedding_type,
        task=task,
    )


def load_dataset_embeddings(dataset_cfg, embedding_type, config, n_data=None, task=None, use_runtime=False):
    if use_runtime:
        return _load_dataset_embeddings_runtime(dataset_cfg, embedding_type, config, task, n_data)

    dataset_path = get_embedding_dataset_dir(dataset_cfg, config)
    ragged_dataset_path = dataset_path / f'ragged_{embedding_type.name.lower()}_seed={config.seed}.pt'
    ragged_text_path = dataset_path / f'ragged_{embedding_type.name.lower()}_text_seed={config.seed}.json'
    if ragged_dataset_path.exists() and ragged_text_path.exists():
        return load_dataset_embeddings_ragged(dataset_path, embedding_type, config, n_data)
    else:
        logger.info(f'Ragged dataset {ragged_dataset_path} not found, falling back to {embedding_type.name.lower()}_seed={config.seed}.pt')
        return load_dataset_embeddings_full(dataset_path, embedding_type, config, n_data)


def dataset_collate_fn(dataset):
    if hasattr(dataset, 'collate_fn') and dataset.collate_fn is not None:
        return dataset.collate_fn
    if isinstance(dataset, Subset):
        return dataset_collate_fn(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        for sub_dataset in dataset.datasets:
            collate = dataset_collate_fn(sub_dataset)
            if collate is not None:
                return collate
    return None


def load_dataset_embeddings_full(dataset, embedding_type, config, n_data=None):
    dataset_path = Path(dataset) / f'{embedding_type.name.lower()}_seed={config.seed}.pt'
    text_path = Path(dataset) / f'{embedding_type.name.lower()}_text_seed={config.seed}.json'
    cache_path = os.environ['LOCAL_CACHE_ROOT']
    if cache_path:
        cache_path = Path(cache_path)
        dataset_cache_path = cache_path / str(dataset_path.resolve())[1:]
        os.makedirs(dataset_cache_path.parent, exist_ok=True)
        in_progress_file = Path(str(dataset_cache_path) + '.inprogress')
        while in_progress_file.exists():
            time.sleep(5)
            # logger.info(f'File copy from {dataset_path} to {dataset_cache_path} in progress on another process.')
        if not dataset_cache_path.exists():
            in_progress_file.touch()
            logger.info(f'Copying {dataset_path} to {dataset_cache_path}')
            shutil.copyfile(dataset_path, dataset_cache_path)
            logger.info(f'Finished copying {dataset_path} to {dataset_cache_path}')
            os.remove(in_progress_file)
        else:
            logger.info(f'Found cache file {dataset_cache_path}')
    else:
        dataset_cache_path = dataset_path

    logger.info(f'Loading dataset embeddings {dataset_cache_path}')
    dataset = torch.load(dataset_cache_path, weights_only=True)
    # texts don't have to be cached because of their very small size
    with open(text_path) as f:
        texts = json.load(f)
    embeddings = dataset['embeddings']
    masks = dataset['masks']
    dataset = CollectionDataset(embeddings.share_memory_(), dataset['input_ids'].share_memory_(), masks.share_memory_(), texts)
    if n_data:
        if n_data > len(dataset):
            raise ValueError(f'n_data {n_data} is larger than dataset size {len(dataset)}')
        subset_idxs = np.random.choice(len(dataset), n_data, replace=False)
        dataset = torch.utils.data.Subset(dataset, subset_idxs)
    return dataset


def load_dataset_embeddings_ragged(dataset, embedding_type, config, n_data=None):
    dataset_path = Path(dataset) / f'ragged_{embedding_type.name.lower()}_seed={config.seed}.pt'
    text_path = Path(dataset) / f'ragged_{embedding_type.name.lower()}_text_seed={config.seed}.json'
    cache_path = os.environ['LOCAL_CACHE_ROOT']
    if cache_path:
        cache_path = Path(cache_path)
        dataset_cache_path = cache_path / str(dataset_path.resolve())[1:]
        os.makedirs(dataset_cache_path.parent, exist_ok=True)
        in_progress_file = Path(str(dataset_cache_path) + '.inprogress')
        while in_progress_file.exists():
            time.sleep(5)
            # logger.info(f'File copy from {dataset_path} to {dataset_cache_path} in progress on another process.')
        if not dataset_cache_path.exists():
            in_progress_file.touch()
            logger.info(f'Copying {dataset_path} to {dataset_cache_path}')
            shutil.copyfile(dataset_path, dataset_cache_path)
            logger.info(f'Finished copying {dataset_path} to {dataset_cache_path}')
            os.remove(in_progress_file)
        else:
            logger.info(f'Found cache file {dataset_cache_path}')
    else:
        dataset_cache_path = dataset_path

    logger.info(f'Loading dataset embeddings {dataset_cache_path}')
    #dataset = torch.load(dataset_cache_path, weights_only=True)
    dataset = torch.load(dataset_cache_path, weights_only=True, map_location='cpu', mmap=True)
    # texts don't have to be cached because of their very small size
    with open(text_path) as f:
        texts = json.load(f)
    embeddings = dataset['embeddings']
    input_ids = dataset['input_ids']
    start_idxs = dataset['start_idxs']
    # embeddings, input_ids, masks = convert_from_ragged_format(embeddings, input_ids, start_idxs)
    dataset = RaggedDataset(embeddings, input_ids.share_memory_(), start_idxs.share_memory_(), texts)
    if n_data:
        if n_data > len(dataset):
            raise ValueError(f'n_data {n_data} is larger than dataset size {len(dataset)}')
        subset_idxs = np.random.choice(len(dataset), n_data, replace=False)
        dataset = torch.utils.data.Subset(dataset, subset_idxs)
    return dataset
