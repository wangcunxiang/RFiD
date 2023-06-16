# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
# import wandb
import math
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options
from transformers import AutoTokenizer

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_train_batch_size,
        drop_last=True,
        num_workers=0,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while step < opt.total_steps:
        epoch += 1
        for i, batch in tqdm(enumerate(train_dataloader), desc="Training "):
            step += 1
            (idx, labels, _, context_ids, context_mask, golden) = batch

            if not opt.cpu:
                context_ids = context_ids.cuda()
                context_mask = context_mask.cuda()
                golden = golden.cuda() if golden is not None else None
                labels = labels.cuda()

            train_loss = model(
                input_ids=context_ids,
                attention_mask=context_mask,
                golden=golden,
                labels=labels
            )[0]

            train_loss.backward()

            # if opt.local_rank in [-1, 0] and not opt.no_wandb:
            #     wandb.log({'train_loss': train_loss.item(), 'iteration': step})
            #     wandb.log({'lr': scheduler.get_last_lr()[0], 'iteration': step})

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()
            if math.isnan(train_loss.item()):
                logger.warning("NAN sample {}".format(idx))

            if step % opt.eval_freq == 0:
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    # if opt.local_rank in [-1, 0] and not opt.no_wandb:
                    #     wandb.log({'dev_em': 100 * dev_em, 'iteration': step})
                    #     wandb.log({'stable_train_loss': curr_loss / opt.eval_freq, 'iteration': step})
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break

def evaluate(model, dataset, tokenizer, collator, opt):
    logger.info("Start evaluating")
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_eval_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), desc="Evaluating "):
            (idx, _, _, context_ids, context_mask, _) = batch

            if not opt.cpu:
                context_ids = context_ids.cuda()
                context_mask = context_mask.cuda()

            outputs = model.generate(
                input_ids=context_ids,
                attention_mask=context_mask,
                add_loss=opt.add_loss,
                max_length=opt.answer_maxlength
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers']
                score = src.evaluation.ems(ans, gold)
                total += 1
                exactmatch.append(score)
            # if outputs['encoder_hidden_states'] != None:
            #     encoder_last_hidden_state = outputs['encoder_hidden_states'][-1][:,0,:]
            #     logits = model.classifier(encoder_last_hidden_state)
            #     preds_cls = torch.argmax(logits, dim=-1)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    print('name = {}'.format(opt.name))

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = 't5-' + opt.model_size
    model_class = src.model.FiDT5

    #load data
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True) #transformers.T5Tokenizer.from_pretrained(model_name)
    collator = src.data.Collator(
        opt.text_maxlength,
        tokenizer,
        answer_maxlength=opt.answer_maxlength,
        add_loss=opt.add_loss,
        extra_decoder_inputs=opt.extra_decoder_inputs,
    )

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)

    if opt.model_size is not None and opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        config = t5.config
        # config.update({'return_dict_in_generate': True if opt.add_loss is not None else False})
        # config.update({'output_hidden_states': True if opt.add_loss is not None else False})
        model = src.model.FiDT5(config, opt)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank if not opt.cpu else 'cpu')
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
        # if opt.add_loss == 'binary': # using "positive" and "negative" to initialize positive and negative labels
        #     pos_emb = model.shared(torch.LongTensor(tokenizer.encode("positive")[:1]).to(model.device))
        #     neg_emb = model.shared(torch.LongTensor(tokenizer.encode("negative")[:1]).to(model.device))
        #     model.classifier.weight = torch.nn.Parameter(torch.cat([neg_emb, pos_emb], dim=0))
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )
