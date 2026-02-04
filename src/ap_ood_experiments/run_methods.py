from dotenv import load_dotenv
load_dotenv()

import logging
import traceback

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import wandb

from ap_ood_experiments import EmbeddingType
from ap_ood_experiments.ood import OODEvaluator
from ap_ood_experiments.utils import set_seed
from ap_ood_experiments.utils.dataset_util import load_dataset_embeddings


logger = logging.getLogger(__name__)


def run_method(method, id_dataset_train, id_dataset_test, aux_dataset_train, aux_dataset_val, ood_datasets, embedding_type, config, task=None):
    method = instantiate(method)

    embedding_loader_mode = config.embedding_loader.mode

    use_runtime_all = embedding_loader_mode == 'model'
    use_runtime_id_train = embedding_loader_mode in ('model', 'hybrid')

    if method.requires_id:
        id_dataset = load_dataset_embeddings(
            id_dataset_train,
            embedding_type,
            config,
            task=task,
            use_runtime=use_runtime_id_train,
        )

        logger.info(f'Fitting method {method}')
        if method.requires_aux:
            logger.info(f'{method} requires auxiliary data. Loading...')
            if embedding_type in (EmbeddingType.INPUT, EmbeddingType.OUTPUT):
                aux_dataset_ = load_dataset_embeddings(
                    aux_dataset_train,
                    embedding_type,
                    config,
                    n_data=config.n_aux_data,
                    task=task,
                    use_runtime=use_runtime_all,
                )
            method.fit(id_dataset, aux_dataset_)
            del aux_dataset_
        else:
            method.fit(id_dataset)

        del id_dataset

    out_datasets = [id_dataset_test, aux_dataset_train, aux_dataset_val] + ood_datasets
    ood_train_evaluator = OODEvaluator(
        id_dataset=id_dataset_train,
        out_datasets=out_datasets,
        metrics=instantiate(config.metrics),
        logger=instantiate(config.train_logger),
        batch_size=config.eval_batch_size,
        device=config.device,
        id_embedding_type=embedding_type,
        ood_embedding_type=embedding_type,
        config=config,
        task=task,
        use_runtime=use_runtime_all,
    )
    ood_aux_evaluator = OODEvaluator(
        id_dataset=id_dataset_test,
        out_datasets=[aux_dataset_train],
        metrics=instantiate(config.metrics),
        logger=instantiate(config.aux_logger),
        batch_size=config.eval_batch_size,
        device=config.device,
        id_embedding_type=embedding_type,
        ood_embedding_type=embedding_type,
        config=config,
        task=task,
        use_runtime=use_runtime_all,
    )
    ood_aux_val_evaluator = OODEvaluator(
        id_dataset=id_dataset_test,
        out_datasets=[aux_dataset_val],
        metrics=instantiate(config.metrics),
        logger=instantiate(config.aux_val_logger),
        batch_size=config.eval_batch_size,
        device=config.device,
        id_embedding_type=embedding_type,
        ood_embedding_type=embedding_type,
        config=config,
        task=task,
        use_runtime=use_runtime_all,
    )
    ood_test_evaluator = OODEvaluator(
        id_dataset=id_dataset_test,
        out_datasets=ood_datasets,
        metrics=instantiate(config.metrics),
        logger=instantiate(config.test_logger),
        batch_size=config.eval_batch_size,
        device=config.device,
        id_embedding_type=embedding_type,
        ood_embedding_type=embedding_type,
        config=config,
        task=task,
        use_runtime=use_runtime_all,
    )

    if config.eval_train:
        ood_train_evaluator.evaluate(method, epoch=None)
    if config.eval_aux:
        ood_aux_evaluator.evaluate(method, epoch=None)
    if config.eval_aux_val:
        ood_aux_val_evaluator.evaluate(method, epoch=None)
    if config.eval_test:
        scores = ood_test_evaluator.evaluate(method, epoch=None)
        return scores
    return


@hydra.main(config_path='config_run_methods', config_name='summarization-pegasus-xsum-input', version_base='1.2')
def main(config):
    
    logger.info('Starting run...')
    
    try:
        wandb.init(
            project='AP-OOD',
            config={
                'hydra': OmegaConf.to_container(
                    config,resolve=True,
                    throw_on_missing=True
        )})
        set_seed(config.seed + config.addtl_seed)
        print(OmegaConf.to_yaml(config))
        
        embedding_type: EmbeddingType = EmbeddingType[config.embedding_type]
        
        mode = config.embedding_loader.mode

        task = None
        if mode in ('model', 'hybrid'):
            task = instantiate(config.task).to(config.device).eval()

        metrics = run_method(
            method=config.method,
            id_dataset_train=config.data.id.train,
            id_dataset_test=config.data.id.test,
            aux_dataset_train=config.data.aux.train,
            aux_dataset_val=config.data.aux.val,
            ood_datasets=config.data.ood,
            embedding_type=embedding_type,
            config=config,
            task=task,
        )
        wandb.finish()
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e


if __name__ == '__main__':
    main()
