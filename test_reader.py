# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json

import torch
import transformers
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer


import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model

def evaluate(model, dataset, dataloader, tokenizer, opt):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage() 
    total = 0
    exactmatch = []
    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(write_path / ('{}.json'.format(opt.name)), 'a')
    if opt.split_psg_subset:
        n = opt.n_context
        import math
        pow_n = int(math.pow(2, n))
        a = [[]]
        psg_idx_map = []
        golden_psg_shap_values = 0
        golden_psg_shap_values_ave = 0
        for i in range(n):
            b = [j + [1] for j in a]
            c = [j + [0] for j in a]
            a = b + c
        for i in range(n):
            j = 0
            pow_i = int(math.pow(2, i))
            tmp = []
            while j < pow_n:
                for k in range(pow_i):
                    tmp.append([j+k, j+k+pow_i])
                j += pow_i * 2
            psg_idx_map.append(tmp)
    else:
        pow_n = 1
    if opt.sum_golden_cross_att:
        golden_psg_att = 0
        non_golden_psg_att = 0
    Os = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            (idx, _, _, context_ids, context_mask, golden) = batch

            if opt.write_crossattention_scores:
                model.reset_score_storage()

            if not opt.cpu:
                context_ids = context_ids.cuda()
                context_mask = context_mask.cuda()

            outputs = model.generate(
                input_ids=context_ids,
                attention_mask=context_mask,
                add_loss=opt.add_loss,
                max_length=opt.answer_maxlength
            )
            batch_scores = []
            if opt.output_attentions and opt.split_psg_subset:
                bsz = idx.shape[0]
                cross_attentions = outputs['cross_attentions']
                cross_attentions_ = 0
                # if opt.cat_emb:
                #     text_maxlength = opt.text_maxlength+1
                # else:
                #     text_maxlength = opt.text_maxlength
                for step in cross_attentions[:5]:
                    for layer in step:
                        cross_attentions_ += layer
                ave_psg_att = cross_attentions_.sum() / bsz / opt.n_context / pow_n * 2  # *2 because half cells are 0
                bsz = idx.shape[0]
                for j in range(len(golden)):
                    for k in range(len(golden[0])):
                        if golden[j][k] == 1:
                            for pair in psg_idx_map[k]:
                                s = pair[0]
                                t = pair[1]
                                value = int(batch_scores[j + s * bsz][k]) - int(batch_scores[j + t * bsz][k])
                                assert value > 0
                                golden_psg_shap_values += value
                                golden_psg_shap_values_ave += ave_psg_att
                print("\ngolden_psg_shap_values = {}\ngolden_psg_shap_values_ave = {}".format(golden_psg_shap_values,
                                                                                              golden_psg_shap_values_ave))
                outputs = outputs['sequences']
            if opt.output_attentions and opt.sum_golden_cross_att:
                bsz = idx.shape[0]
                cross_attentions = outputs['cross_attentions']
                cross_attentions_ = 0
                for step in cross_attentions[:5]:  # [:5] only use the first 5 tokens
                    # for layer in step: # sum up all layers
                    #     cross_attentions_ += layer
                    cross_attentions_ += step[-1]  # only sum last layers
                if opt.cat_emb:
                    text_maxlength = opt.text_maxlength+1
                else:
                    text_maxlength = opt.text_maxlength
                batch_scores = cross_attentions_.sum(1).sum(1).view(bsz, opt.n_context, text_maxlength).sum(-1).tolist()
                for j in range(len(golden)):
                    tmp_golden_psg_att = 0
                    tmp_non_golden_psg_att = 0
                    cnt = 0
                    for k in range(len(golden[0])):
                        if golden[j][k] == 1:
                            tmp_golden_psg_att += batch_scores[j][k]
                            cnt += 1
                        else:
                            tmp_non_golden_psg_att += batch_scores[j][k]
                    if cnt == 0 or cnt == len(golden[0]):
                        continue
                    golden_psg_att += (tmp_golden_psg_att / cnt)
                    non_golden_psg_att += (tmp_non_golden_psg_att / (len(golden[0]) - cnt))
                # print("\ngolden_psg_att = {}\nnon_golden_psg_att = {}".format(golden_psg_att, non_golden_psg_att))
                outputs = outputs['sequences']
            if opt.write_crossattention_scores:
                crossattention_scores = model.get_crossattention_scores(context_mask.cuda())

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k%idx.shape[0]]]
                if 'answers' in example:
                    score = src.evaluation.ems(ans, example['answers'])
                    exactmatch.append(score)
                    if not opt.output_attentions:
                        batch_scores.append(score)

                if opt.write_results:
                    output = {
                        'id':example['id'],
                        'question':example['question'],
                        'golden_answers':example['answers'],
                        'generated_answer':ans,
                        'ctxs':example['ctxs']
                    }
                    Os.append(output)
                    #fw.write(str(example['id']) + "\n" + ans + '\n' + str(score) + '\n' + example['question'] + '\' + str(example['answers']) + '\n')
                if opt.write_crossattention_scores:
                    for j in range(context_ids.size(1)):
                        example['ctxs'][j]['score'] = crossattention_scores[k, j].item()

                total += 1
            if (i + 1) % opt.eval_print_freq == 0:
                log = f'Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}'
                if len(exactmatch) == 0:
                    log += '| no answer to compute scores'
                else:
                    log += f' | average = {np.mean(exactmatch):.3f}'
                logger.warning(log)

    logger.warning(f'Process rank:{opt.global_rank}, total {total} | average = {np.mean(exactmatch):.3f}')
    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    if opt.write_results:
        json.dump(Os, fw, indent=1)
    
    return score, total


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_eval_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)
    print(opt.name)

    tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True, local_files_only=True)

    collator_function = src.data.Collator(opt.text_maxlength, tokenizer, add_loss=opt.add_loss)
    eval_examples = src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size
    )
    eval_dataset = src.data.Dataset(
        eval_examples, 
        opt.n_context, 
    )

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_eval_batch_size,
        num_workers=0,
        collate_fn=collator_function
    )
    
    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path, opt)
    model = model.to(opt.device)

    logger.info("Start eval")
    exactmatch, total = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    logger.info(f'EM {100*exactmatch:.2f}, Total number of example {total}')

    # if opt.write_results and opt.is_main:
    #     glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
    #     write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
    #     src.util.write_output(glob_path, write_path)
    if opt.write_crossattention_scores:
        src.util.save_distributed_dataset(eval_dataset.data, opt)

