import os
import datasets
from dotenv import load_dotenv
import numpy as np
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb

from ap_ood_experiments.transformer.transformer import Transformer
from ap_ood_experiments.dataset.wmt import WMTDataset


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def is_main_process():
    return get_rank() == 0

def setup_ddp():
    # torchrun sets these
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank

def cleanup_ddp():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def eval_loss(model, loader, tokenizer, src_lang, trg_lang, input_ctx_len, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            tokenized = tokenizer(batch['translation'][src_lang],
                                  text_target=batch['translation'][trg_lang],
                                  truncation=True,
                                  padding=True,
                                  max_length=input_ctx_len,
                                  return_tensors='pt')

            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            target_input_ids = tokenized['labels'].to(device)
            target_attention_mask = (target_input_ids != tokenizer.pad_token_id).int()
            target_input_ids_input = torch.concat(
                [torch.full([len(target_input_ids), 1], tokenizer.pad_token_id, device=device), target_input_ids],
                dim=1
            )[:, :-1]
            target_attention_mask_input = torch.concat(
                [torch.full([len(target_attention_mask), 1], 1, device=device), target_attention_mask],
                dim=1
            )[:, :-1]

            predictions = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                target_input_ids=target_input_ids_input,
                                target_attention_mask=target_attention_mask_input)

            predictions = torch.masked_select(predictions, target_attention_mask.unsqueeze(-1).bool()).reshape(-1, len(tokenizer))
            target_input_ids = torch.masked_select(target_input_ids, target_attention_mask.bool())
            loss = nn.CrossEntropyLoss(reduction='none')(predictions, target_input_ids)
            losses.append(loss)
        losses = torch.concat(losses, dim=0)
        loss = torch.mean(losses, dim=0)
    return loss


def save_checkpoint(model, optimizer, epoch, n_steps, scheduler, filename='checkpoint.pth.tar'):
    if not is_main_process():
        return
    os.makedirs(os.path.split(filename)[0], exist_ok=True)
    print(f"=> saving checkpoint '{filename}'")
    state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    state = {
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict(),
        'n_steps': n_steps,
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, filename)


def main():
    load_dotenv()
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.environ['WMT_MODEL_CHECKPOINT']

    # W&B only on rank 0
    if is_main_process():
        wandb.init(project='ap_ood_transformer')

    torch.backends.cuda.enable_flash_sdp(True)

    train_dataset = WMTDataset(split='train', cache_dir=os.environ['WMT_ROOT'])
    test_dataset = WMTDataset(split='test', cache_dir=os.environ['WMT_ROOT'])

    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small')
    model = Transformer(len(tokenizer)).to(device)

    input_ctx_len = 512
    src_lang = 'en'
    trg_lang = 'fr'

    per_device_batch_size = 128
    gradient_step_batch_size = 1024
    max_steps = 1_000_000
    n_eval_steps = 1_000

    # global batch = per_device_batch_size * world_size * gradient_accumulation_steps
    assert gradient_step_batch_size % (per_device_batch_size * max(world_size, 1)) == 0, \
        "gradient_step_batch_size must be divisible by per_device_batch_size * world_size"
    gradient_accumulation_steps = gradient_step_batch_size // (per_device_batch_size * max(world_size, 1))

    # Sampler for DDP
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    # eval on rank 0 only; plain loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=per_device_batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=max_steps)

    n_steps = 0
    epoch = 0

    # Wrap with DDP after optional load
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    pbar = tqdm(total=max_steps, disable=not is_main_process())
    pbar.update(n_steps)
    losses = []
    n_tokens = []

    try:
        while n_steps < max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for batch in train_loader:
                model.train()
                tokenized = tokenizer(batch['translation'][src_lang],
                                      text_target=batch['translation'][trg_lang],
                                      truncation=True,
                                      padding=True,
                                      max_length=input_ctx_len,
                                      return_tensors='pt')

                input_ids = tokenized['input_ids'].to(device, non_blocking=True)
                attention_mask = tokenized['attention_mask'].to(device, non_blocking=True).bool()
                target_input_ids = tokenized['labels'].to(device, non_blocking=True).long()
                target_attention_mask = (target_input_ids != tokenizer.pad_token_id)
                target_input_ids_input = torch.concat(
                    [torch.full([len(target_input_ids), 1], tokenizer.pad_token_id, device=device), target_input_ids],
                    dim=1
                )[:, :-1]
                target_attention_mask_input = torch.concat(
                    [torch.full([len(target_attention_mask), 1], True, device=device), target_attention_mask],
                    dim=1
                )[:, :-1]

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    predictions = model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        target_input_ids=target_input_ids_input,
                                        target_attention_mask=target_attention_mask_input)

                    predictions = torch.masked_select(predictions, target_attention_mask.unsqueeze(-1).bool()).reshape(-1, len(tokenizer))
                    target_input_ids = torch.masked_select(target_input_ids, target_attention_mask.bool())
                    loss = nn.CrossEntropyLoss()(predictions, target_input_ids)

                (loss / gradient_accumulation_steps).backward()
                losses.append(loss.detach())
                n_tokens.append(len(target_input_ids))

                if len(losses) == gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    n_steps += 1
                    if is_main_process():
                        pbar.update()

                        wandb.log({
                            'train/loss': (sum([l.item() for l in losses]) / len(losses)),
                            'general/epoch': epoch,
                            'general/step': n_steps,
                            'train/lr': optimizer.param_groups[0]['lr'],
                            'train/n_tokens': int(sum(n_tokens)),
                        }, step=n_steps)

                    losses = []
                    n_tokens = []

                    if n_steps % n_eval_steps == 0 and n_steps != 0 and is_main_process():
                        val_loss = eval_loss(model.module if isinstance(model, DDP) else model,
                                             test_loader, tokenizer, src_lang, trg_lang, input_ctx_len, device)
                        wandb.log({
                            'val/loss': val_loss.item(),
                            'general/epoch': epoch,
                            'general/step': n_steps,
                            'train/lr': optimizer.param_groups[0]['lr']
                        }, step=n_steps)

                        print(f'Saving model in: {checkpoint_path}')
                        save_checkpoint(model, optimizer, epoch, n_steps, scheduler, filename=checkpoint_path)

                    if n_steps >= max_steps:
                        break

            epoch += 1
    finally:
        if is_main_process():
            wandb.finish()
        cleanup_ddp()


if __name__ == '__main__':
    main()
