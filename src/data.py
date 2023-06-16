# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import regex
import string
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target
        elif 'answers' in example:
            return random.choice(example['answers'])
        else:
            return None


    def check_answers(self, answers, passage):
        def remove_articles(text):
            return regex.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        passage = white_space_fix(remove_articles(remove_punc(lower(passage.strip()))))
        for a in answers:
            a_new = white_space_fix(remove_articles(remove_punc(lower(a.strip()))))
            if a_new in passage:
                return 1
        return 0

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question'] + " ?"
        target = self.get_target(example)
        answers = example['answers']

        if 'ctxs' in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            golden =[self.check_answers(answers, c['title'] +'. ' + c['text']) for c in contexts]
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores, golden = None, None, None

        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'candidate_answers': answers,
            'passages' : passages,
            'scores' : scores,
            'golden' : golden
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

def encode_passages_spans(batch_text_passages, batch_answers, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    answers_token_ids = []
    for k, text_passages in enumerate(batch_text_passages):
        answer = batch_answers[k].lower().strip()
        answer_starts_lens = [(psg.lower().index(answer), len(answer)) if answer in psg.lower() and psg.lower().index(answer)+ len(answer) <max_length else None for i, psg in enumerate(text_passages)]
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True,
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

        offset_mapping = p['offset_mapping']
        answer_token_id = []
        for i, a_s_l in enumerate(answer_starts_lens):
            o_m = offset_mapping[i]
            answer_token_start_end = []
            if a_s_l is not None:
                a_s = a_s_l[0]
                a_e = a_s_l[0] + a_s_l[1]
                for j, map in enumerate(o_m):
                    if a_s >= map[0] and a_s < map[1]:
                        if len(answer_token_start_end) == 0:
                            answer_token_start_end.append(j)
                    if a_e > map[0] and a_e <= map[1]:
                        assert len(answer_token_start_end) == 1
                        answer_token_start_end.append(j)
                        break
            else:
                answer_token_start_end = [max_length, max_length]
            if len(answer_token_start_end) < 2:
                print("*******")
                answer_token_start_end = [max_length, max_length]
            answer_token_id.append(answer_token_start_end)
        answers_token_ids.append(answer_token_id)
    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool(), torch.LongTensor(answers_token_ids)

def encode_passages_group_tagger(batch_text_passages, batch_candidate_answers, tokenizer, max_length):
    psg_num = len(batch_text_passages[0])
    passage_ids, passage_masks, token_labels = [], [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True,
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

        answers = [ans.lower().strip() for ans in batch_candidate_answers[k]]
        passages = [psg.lower().strip() for psg in batch_text_passages[k]]
        ans_loc = [[] for _ in range(psg_num)]
        passages_with_golden = []
        for pid, psg in enumerate(passages):
            for aid, ans in enumerate(answers):
                try:
                    _ = regex.match(ans, psg)
                except:
                    continue
                    #print('ANS: {} not match the rule of REGEX.'.format(ans))
                else:
                    for m in regex.finditer(ans, psg):
                        ans_loc[pid] += [(m.start(), m.end())]
                        if pid not in passages_with_golden:
                            passages_with_golden += [pid]

        p['token_labels'] = p['input_ids'].new(p['input_ids'].shape).fill_(0)
        offset_mapping_with_golden = p['offset_mapping'][passages_with_golden]

        for pid, passage_offset in enumerate(offset_mapping_with_golden):
            for tid, token_offset in enumerate(passage_offset):
                for ans_offset in ans_loc[passages_with_golden[pid]]:
                    ans_begin, ans_end = ans_offset
                    token_begin, token_end = token_offset
                    if token_begin < ans_end and token_end > ans_begin:
                        p['token_labels'][passages_with_golden[pid]][tid] = 1
                        # for distance in range(-15, 16, 1):
                        #     if tid+distance > 0 and tid+distance < max_length and  p['token_labels'][passages_with_golden[pid]][tid+distance] == -100:
                        #         p['token_labels'][passages_with_golden[pid]][tid+distance] = 0


        token_labels.append(p['token_labels'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    token_labels = torch.cat(token_labels, dim=0)
    # for testing, the marked tokens should be all related to answers
    '''
    print('[ANS]')
    print(batch_candidate_answers)
    print('[PSG WITH ANS]')
    print(tokenizer.batch_decode((passage_ids * (token_labels!=-100).bool().int())[0,:,:], skip_special_tokens=True))
    import pdb; pdb.set_trace()
    '''
    return passage_ids, passage_masks.bool(), token_labels

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20, add_loss=None, extra_decoder_inputs=False):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.add_loss = add_loss
        self.extra_decoder_inputs = extra_decoder_inputs

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding='max_length',
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        if self.extra_decoder_inputs:
            extra_decoder_inputs = ['The answer to question ' + ex['question'] + ' is ' + ex['target'] for ex in batch]


        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        # if self.add_loss == None:
        #     passage_ids, passage_masks = encode_passages(text_passages,
        #                                                  self.tokenizer,
        #                                                  self.text_maxlength)
        #     golden = None
        if self.add_loss == None or self.add_loss in ["binary", "mse"]:
            passage_ids, passage_masks = encode_passages(text_passages,
                                                         self.tokenizer,
                                                         self.text_maxlength)
            golden = torch.tensor([ex['golden'] for ex in batch])
        elif self.add_loss in ["span"]:
            passage_ids, passage_masks, answer_token_ids = encode_passages_spans(
                text_passages,
                [ex['target'] for ex in batch],
                self.tokenizer,
                self.text_maxlength)
            golden = answer_token_ids
        elif self.add_loss in ["group_tagger"]:
            passage_ids, passage_masks, answer_token_ids = encode_passages_group_tagger(
                text_passages,
                [ex['candidate_answers'] for ex in batch],
                self.tokenizer,
                self.text_maxlength,
            )
            golden = answer_token_ids
        elif self.add_loss in ["binary_token"]:
            passage_ids, passage_masks, answer_token_ids = encode_passages_group_tagger(
                text_passages,
                [ex['candidate_answers'] for ex in batch],
                self.tokenizer,
                self.text_maxlength,
            )
            golden_psg = torch.tensor([ex['golden'] for ex in batch])
            golden = torch.cat([golden_psg.unsqueeze(-1), answer_token_ids], dim=-1)
        else:
            raise ValueError("Loss {} is not used".format(self.add_loss))

        return (index, target_ids, target_mask, passage_ids, passage_masks, golden)

def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        # if global_rank > -1 and not k%world_size==global_rank:
        #     continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples

class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            padding='max_length',
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['passages'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex['passages'] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            passages,
            self.tokenizer,
            self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, scores)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = self.title_prefix + " " + example[2] + " " + \
            self.passage_prefix + " " + example[1]
        return example[0], text

class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            padding='max_length',
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()

        return index, text_ids, text_mask
