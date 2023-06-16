import copy
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import random
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.checkpoint import checkpoint

from transformers.file_utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    ModelOutput,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import T5Config
from transformers.models.t5.modeling_t5 import (
    T5PreTrainedModel,
    T5LayerNorm,
    T5LayerFF,
    T5Stack,
    T5_START_DOCSTRING,
    PARALLELIZE_DOCSTRING,
    DEPARALLELIZE_DOCSTRING,
    T5_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,
    logger,
)

@dataclass
class Seq2SeqLMOutput(ModelOutput):
    """
        Base class for sequence-to-sequence language models outputs.

        Args:
            loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
                Language modeling loss.
            logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
                `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
                `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
                blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
            decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
                Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
                shape `(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
            decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
                Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
                sequence_length)`.

                Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
                self-attention heads.
            cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
                Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
                sequence_length)`.

                Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
                weighted average in the cross-attention heads.
            encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder of the model.
            encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
                Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
                shape `(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
            encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
                Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
                sequence_length)`.

                Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
                self-attention heads.
        """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    classifer_logits:  torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None



# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config, opt):
        super().__init__(config)
        self.model_dim = config.d_model
        self.add_loss = opt.add_loss
        self.add_type_emb = opt.add_type_emb
        self.cat_emb = opt.cat_emb
        self.rerank = opt.rerank
        self.change_golden = opt.change_golden

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        if opt.add_loss == 'binary' or opt.add_loss == 'binary_token':
            self.num_labels = 2
            self.passage_classifier = nn.Linear(config.hidden_size, self.num_labels)
            # self._init_weights(self.passage_classifier)
        if opt.add_loss == 'group_tagger' or opt.add_loss == 'binary_token':
            self.num_labels = 2
            self.token_classifier = nn.Linear(config.hidden_size, self.num_labels)
            self.passage_classifier = nn.Linear(config.hidden_size, self.num_labels)
            # self._init_weights(self.token_classifier)
            # self._init_weights(self.passage_classifier)
            self.tagger_emb = nn.Embedding(self.num_labels, config.d_model, padding_idx=0)
        if opt.add_type_emb:
            self.additional_type_emb = nn.Embedding(opt.text_maxlength + config.pad_token_id + 1, config.d_model,
                                               padding_idx=config.pad_token_id)
        if opt.cat_emb:
            self.additional_emb = nn.Embedding(2, config.d_model)
        if opt.add_loss == 'span':
            self.num_labels = 2
            self.span_outputs = nn.Linear(config.hidden_size, self.num_labels)
            # self._init_weights(self.span_outputs)
        self.pad_token_id = config.pad_token_id
        self.text_maxlength = opt.text_maxlength
        self.n_eval_bsz = opt.per_gpu_eval_batch_size
        self.sample_pos_neg = opt.sample_pos_neg

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def sample_pos_neg(self, golden, num_negs=7):
        neg_pos_ids = [[(i-1).nonzero().tolist(), i.nonzero().tolist()] for i in golden]
        matrix = torch.zeros(golden.shape).bool()
        for i in range(golden.shape[0]):
            if len(neg_pos_ids[i][1])<1 or len(neg_pos_ids[i][1])>=golden.shape[1]/2 or len(neg_pos_ids[i][0])<num_negs:
                continue
            neg = random.sample(neg_pos_ids[i][0], 7)
            pos = random.sample(neg_pos_ids[i][1], 1)
            neg_pos = neg + pos
            for j in neg_pos:
                matrix[i][j] = True
        return matrix

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        golden=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        >>> ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        loss_token = None
        loss_cls = None
        loss_span = None
        if self.add_loss != None:
            encoder_last_hidden_state = hidden_states[:]
            bsz, total_length, d_model = encoder_last_hidden_state.shape
            passage_length = total_length // self.encoder.n_passages
            encoder_last_hidden_state = encoder_last_hidden_state.view(bsz, self.encoder.n_passages, passage_length, -1)
        if self.add_loss == 'binary':
            encoder_last_hidden_state = encoder_last_hidden_state[:, :, 0, :]
            logits = self.passage_classifier(encoder_last_hidden_state)
            preds_cls = torch.argmax(logits, dim=-1)
            if self.rerank:
                scores = logits[:, :, 1].tolist()
                shape = logits.shape
                idx_scores = [[(j, scores[i][j]) for j in range(shape[1])] for i in range(shape[0])]
                sorted_idx_scores = [sorted(idx_score, key=lambda x: x[1], reverse=True) for idx_score in idx_scores]
                sorted_idx = torch.LongTensor(sorted_idx_scores)[:, :, 0].to(hidden_states.device)
                hidden_states = hidden_states.view(bsz, self.encoder.n_passages, passage_length, -1)
                hidden_states = torch.stack(
                    [torch.stack([hidden_states[i][sorted_idx[i][j]] for j in range(shape[1])]) for i in
                     range(shape[0])])
                hidden_states = hidden_states.view(bsz, self.encoder.n_passages * passage_length, -1)
                attention_mask = attention_mask.view(bsz, self.encoder.n_passages, passage_length)
                attention_mask = torch.stack(
                    [torch.stack([attention_mask[i][sorted_idx[i][j]] for j in range(shape[1])]) for i in
                     range(shape[0])])
                attention_mask = attention_mask.view(bsz, self.encoder.n_passages * passage_length)
            if self.add_type_emb:
                position_ids = torch.arange(self.pad_token_id + 1, self.pad_token_id + 1 + self.text_maxlength,
                                            dtype=torch.long, device=hidden_states.device)
                one_embedding = self.additional_type_emb(position_ids)
                zero_embedding = self.additional_type_emb(
                    torch.full(position_ids.shape, self.pad_token_id, device=hidden_states.device))
                type_embedding = torch.cat([zero_embedding.unsqueeze(0), one_embedding.unsqueeze(0)], dim=0)
                hidden_states = hidden_states + type_embedding[preds_cls].view(hidden_states.shape).to(hidden_states.device)
            if self.cat_emb:
                self.additional_emb = self.additional_emb.to(hidden_states.device)
                cated_emb = self.additional_emb(preds_cls).unsqueeze(dim=2)
                hidden_states = hidden_states.view(bsz, self.encoder.n_passages, passage_length, -1)
                hidden_states = torch.cat([hidden_states, cated_emb], dim=2)
                hidden_states = hidden_states.view(bsz, (passage_length+1)* self.encoder.n_passages, -1)
                attention_mask = attention_mask.view(bsz, self.encoder.n_passages, passage_length)
                cated_att_mask = preds_cls.unsqueeze(dim=-1).bool().to(attention_mask.device)
                attention_mask = torch.cat([attention_mask, cated_att_mask], dim=-1)
                attention_mask = attention_mask.view(bsz, (passage_length+1)* self.encoder.n_passages)
            if self.training:
                loss_fct = CrossEntropyLoss()
                if self.sample_pos_neg:
                    matrix = self.sample_pos_neg(golden=golden)
                    new_logits = logits[matrix]
                    new_golden = golden[matrix]
                else:
                    new_logits = logits
                    new_golden = golden
                if len(new_logits) > 0:
                    loss_cls = loss_fct(new_logits.view(-1, self.num_labels), new_golden.view(-1))
                else:
                    loss_cls = 0
        elif self.add_loss == 'mse':
            if self.training:
                encoder_last_hidden_state = hidden_states[:, :, 0, :]
                similarity_matrices = torch.bmm(encoder_last_hidden_state, encoder_last_hidden_state.transpose(-2, -1))
                # torch.stack([torch.matmul(embs, embs.T) for embs in encoder_last_hidden_state])
                golden = torch.unsqueeze(golden, dim=-1).float()
                golden_matrix = torch.bmm(golden, golden.transpose(-2, -1))
                similarity_matrices = torch.triu(similarity_matrices, diagonal=1)  # to
                golden_matrix = torch.triu(golden_matrix, diagonal=1)
                loss_fct = MSELoss()
                loss_cls = loss_fct(similarity_matrices, golden_matrix)
        elif self.add_loss == 'span':
            logits = self.span_outputs(encoder_last_hidden_state)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()
            start_logits = start_logits.view(start_logits.shape[0]*start_logits.shape[1], -1)
            end_logits = end_logits.view(end_logits.shape[0] * end_logits.shape[1], -1)

            start_positions = None
            end_positions = None
            if self.training:
                start_positions = golden[:,:,0].to(hidden_states.device)
                end_positions = golden[:,:,1].to(hidden_states.device)
                start_positions = start_positions.view(start_positions.shape[0] * start_positions.shape[1],)
                end_positions = end_positions.view(end_positions.shape[0] * end_positions.shape[1],)
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss_span = (start_loss + end_loss) / 2
                if torch.isnan(loss_span):
                    loss_span = None
        elif self.add_loss == 'group_tagger':
            token_logits = self.token_classifier(encoder_last_hidden_state)
            preds_token = torch.argmax(token_logits, dim=-1)
            self.tagger_emb = self.tagger_emb.to(hidden_states.device)
            type_emb = self.tagger_emb(preds_token)
            hidden_states = hidden_states + type_emb.view(hidden_states.shape)

            if self.training:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = token_logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, golden.view(-1), torch.tensor(loss_fct.ignore_index).type_as(golden)
                    )
                    loss_token = loss_fct(active_logits, active_labels)
                else:
                    loss_token = loss_fct(token_logits.view(-1, self.num_labels), golden.view(-1))
            '''
            torch.save(logits, 'token_logits.pt')
            torch.save(golden, 'token_golden.pt')
            torch.save(hidden_states, 'encoder_hidden_states.pt')
            torch.save(input_ids, 'input_ids.pt')
            import pdb; pdb.set_trace()
            '''
        elif self.add_loss == 'binary_token':
            if self.training:
                golden_token = golden[:, :, 1:].contiguous().to(hidden_states.device)
                golden_psg = golden[:, :, 0].contiguous().to(hidden_states.device)

            # token_level
            token_logits = self.token_classifier(encoder_last_hidden_state)
            if self.cat_emb:
                preds_token = torch.argmax(token_logits, dim=-1)
                self.tagger_emb = self.tagger_emb.to(hidden_states.device)
                type_emb = self.tagger_emb(preds_token)
                hidden_states = hidden_states + type_emb.view(hidden_states.shape)

            if self.training:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = token_logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, golden_token.view(-1), torch.tensor(loss_fct.ignore_index).type_as(golden_token)
                    )
                    loss_token = loss_fct(active_logits, active_labels)
                else:
                    loss_token = loss_fct(token_logits.view(-1, self.num_labels), golden_token.view(-1))

            # passage_level
            encoder_last_hidden_state = encoder_last_hidden_state[:, :, 0, :]
            logits = self.passage_classifier(encoder_last_hidden_state)
            preds_cls = torch.argmax(logits, dim=-1)
            if self.add_type_emb:
                position_ids = torch.arange(self.pad_token_id + 1, self.pad_token_id + 1 + self.text_maxlength,
                                            dtype=torch.long, device=hidden_states.device)
                one_embedding = self.additional_type_emb(position_ids)
                zero_embedding = self.additional_type_emb(
                    torch.full(position_ids.shape, self.pad_token_id, device=hidden_states.device))
                type_embedding = torch.cat([zero_embedding.unsqueeze(0), one_embedding.unsqueeze(0)], dim=0)
                hidden_states = hidden_states + type_embedding[preds_cls].view(hidden_states.shape).to(
                    hidden_states.device)
            if self.cat_emb:
                self.additional_emb = self.additional_emb.to(hidden_states.device)
                cated_emb = self.additional_emb(preds_cls).unsqueeze(dim=2)
                hidden_states = hidden_states.view(bsz, self.encoder.n_passages, passage_length, -1)
                hidden_states = torch.cat([hidden_states, cated_emb], dim=2)
                hidden_states = hidden_states.view(bsz, (passage_length + 1) * self.encoder.n_passages, -1)
                attention_mask = attention_mask.view(bsz, self.encoder.n_passages, passage_length)
                cated_att_mask = preds_cls.unsqueeze(dim=-1).bool().to(attention_mask.device)
                attention_mask = torch.cat([attention_mask, cated_att_mask], dim=-1)
                attention_mask = attention_mask.view(bsz, (passage_length + 1) * self.encoder.n_passages)
            if self.training:
                loss_fct = CrossEntropyLoss()
                if self.sample_pos_neg:
                    matrix = self.sample_pos_neg(golden=golden)
                    new_logits = logits[matrix]
                    new_golden = golden_psg[matrix]
                else:
                    new_logits = logits
                    new_golden = golden_psg
                if len(new_logits) > 0:
                    loss_cls = loss_fct(new_logits.view(-1, self.num_labels), new_golden.view(-1))
                else:
                    loss_cls = 0
        else:
            if not self.add_loss == None:
                raise ("No such extra loss")


        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        if self.split_psg_subset:
            if hidden_states.shape[0] == self.n_eval_bsz:
                bsz, total_length, _ = hidden_states.shape
                passage_length = total_length // self.encoder.n_passages
                attention_mask = attention_mask.view(bsz, self.encoder.n_passages, passage_length)
                cnt = 1
                for i in range(self.n_context):
                    cnt *= 2
                    hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
                    attention_mask_ = attention_mask.clone()
                    attention_mask_[:,i,:] = 0
                    attention_mask = torch.cat([attention_mask, attention_mask_], dim=0)
                attention_mask = attention_mask.view(bsz * cnt, self.encoder.n_passages * passage_length)
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if loss_token != None:
            loss += loss_token
        if loss_cls != None:
            loss += loss_cls
        if loss_span != None:
            loss += loss_span
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
