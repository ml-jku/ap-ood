from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
import json
import torch
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig
import gc

from ap_ood_experiments import EmbeddingType
from ap_ood_experiments.task import EmbeddingCreator
from ap_ood_experiments.utils import set_seed
from ap_ood_experiments.utils.dataset_util import (
    convert_to_ragged_format,
    get_embedding_dataset_dir,
    get_storage_subpath,
)


def save_text(texts, storage_dir, embedding_type, seed=None):
    storage_dir = Path(storage_dir)
    os.makedirs(storage_dir, exist_ok=True)
    embedding_type_name = embedding_type.name.lower() if isinstance(embedding_type, EmbeddingType) else str(embedding_type).lower()
    text_path = storage_dir / f'ragged_{embedding_type_name}_text_seed={seed}.json'

    with open(text_path, 'w') as f:
        json.dump(texts, f, indent=4)


def save_tensor(tensor, storage_dir, embedding_type, seed=None):
    storage_dir = Path(storage_dir)
    os.makedirs(storage_dir, exist_ok=True)
    embedding_type_name = embedding_type.name.lower() if isinstance(embedding_type, EmbeddingType) else str(embedding_type).lower()
    tensor_path = storage_dir / f'ragged_{embedding_type_name}_seed={seed}.pt'

    torch.save(tensor, tensor_path)


def create_embeddings_for_dataset(dataset_cfg, task, embedding_type, n_embeddings, config):
    dataset = instantiate(dataset_cfg.runtime)
    storage_subpath = get_storage_subpath(dataset_cfg)
    storage_dir = get_embedding_dataset_dir(dataset_cfg, config)
    storage_label = str(storage_subpath)

    set_seed(config.seed, storage_label)

    if embedding_type == EmbeddingType.INPUT:
        embedding_fn = task.input_embeddings
        batch_size = config.input_batch_size
    elif embedding_type == EmbeddingType.OUTPUT:
        embedding_fn = task.output_embeddings
        batch_size = config.output_batch_size
    elif embedding_type == EmbeddingType.OUTPUT_HUMAN:
        embedding_fn = task.output_embeddings_human
        batch_size = config.output_batch_size
    else:
        raise ValueError()
    collate_fn = config.get('collate_fn')
    if collate_fn is None:
        collate_fn = torch.utils.data.default_collate
    else:
        collate_fn = instantiate(collate_fn)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    texts = []
    input_ids = []
    embeddings = []
    masks = []
    
    n_samples = min(n_embeddings, len(dataset))
    
    # Track collected count to avoid O(N^2) sum
    current_offset = 0
    
    with tqdm(desc=f'Processing {storage_label}', total=n_samples) as pbar:
    
        for i, batch in enumerate(loader):
            text, input_id, embedding, mask = embedding_fn(batch)
            texts.extend(text)
            input_ids.append(input_id.cpu())
            
            # Optimized progress tracking
            batch_len = len(embedding)
            update_step = min(batch_len, n_samples - current_offset)
            current_offset += update_step
            pbar.update(update_step)

            embeddings.append(embedding.bfloat16().detach().cpu())
            masks.append(mask.detach().cpu())

            # Explicit memory cleanup
            del input_id, embedding, mask, text
            
            # Periodic cache clearing
            if i % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
            if current_offset >= n_embeddings:
                break

    input_ids = torch.concat(input_ids, dim=0)[:n_embeddings]
    embeddings = torch.concat(embeddings, dim=0)[:n_embeddings]
    masks = torch.concat(masks, dim=0)[:n_embeddings]
    texts = texts[:n_embeddings]

    #assert len(embeddings) == n_embeddings
    #assert len(masks) == n_embeddings

    save_text(
        texts=texts,
        storage_dir=storage_dir,
        embedding_type=embedding_type,
        seed=config.seed,
    )

    embeddings, input_ids, start_idxs = convert_to_ragged_format(embeddings, input_ids, masks)

    save_tensor(
        tensor={'embeddings': embeddings, 'input_ids': input_ids, 'start_idxs': start_idxs},
        storage_dir=storage_dir,
        embedding_type=embedding_type,
        seed=config.seed,
    )



@torch.no_grad()
@hydra.main(config_path='config_create_embeddings', version_base='1.2')
def main(config):

    task: EmbeddingCreator = instantiate(config.task).to(config.device).eval()

    embedding_type: EmbeddingType = EmbeddingType[config.embedding_type]

    id_datasets = config.data.id
    aux_datasets = config.data.aux
    ood_datasets = config.data.ood

    def iter_dataset_configs(group):
        if isinstance(group, (ListConfig, list)):
            return group
        if isinstance(group, (DictConfig, dict)):
            return group.values()
        raise ValueError('Dataset configuration must be a list or dict of dataset specs.')

    if config.run_id:
        for dataset_conf in iter_dataset_configs(id_datasets):
            create_embeddings_for_dataset(
                dataset_cfg=dataset_conf,
                task=task,
                embedding_type=embedding_type,
                n_embeddings=config.n_embeddings_id,
                config=config,
            )

    if config.run_aux:   
        for dataset_conf in iter_dataset_configs(aux_datasets):
            create_embeddings_for_dataset(
                dataset_cfg=dataset_conf,
                task=task,
                embedding_type=embedding_type,
                n_embeddings=config.n_embeddings_aux,
                config=config,
            )

    if config.run_ood:  
        for dataset_conf in iter_dataset_configs(ood_datasets):
            create_embeddings_for_dataset(
                dataset_cfg=dataset_conf,
                task=task,
                embedding_type=embedding_type,
                n_embeddings=config.n_embeddings_ood,
                config=config,
            )


if __name__ == '__main__':
    main()
