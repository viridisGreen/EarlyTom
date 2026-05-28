from functools import partial
from typing import Callable, Optional, Tuple, Union, List

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel, Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding, logger

class Qwen2Model_fastvid(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        # import os
        # self.fastvid_retention_ratio = float(os.getenv("RETAIN_RATIO", 0.1))
        # self.fastvid_DySeg_c = int(os.getenv("DYSET_C", 8))
        # self.fastvid_DySeg_tau = float(os.getenv("tau", 0.9))
        # self.fastvid_STPrune_d = float(os.getenv("D", 0.4))
        # self.fastvid_DTM_p = int(os.getenv("DTM_P", 4))
        # self.fastvid_DTM_alpha = float(os.getenv("DTM_A", 0.6))
        # self.video_start_idx = None
        # self.video_token_len = None
        # self.frame_num = None
        # self.frame_attn_weights = None
        # self.frame_global_features = None

    def set_my_kwargs(self, video_start_idx, video_token_len, frame_num, frame_attn_weights, frame_global_features):
        self.video_start_idx = video_start_idx
        self.video_token_len = video_token_len
        self.frame_num = frame_num
        self.frame_attn_weights = frame_attn_weights
        self.frame_global_features = frame_global_features

        import os
        self.fastvid_retention_ratio = float(os.getenv("RETAIN_RATIO", 0.1))
        self.fastvid_DySeg_c = int(os.getenv("DYSET_C", 8))
        self.fastvid_DySeg_tau = float(os.getenv("tau", 0.9))
        self.fastvid_STPrune_d = float(os.getenv("D", 0.4))
        self.fastvid_DTM_p = int(os.getenv("DTM_P", 4))
        self.fastvid_DTM_beta = float(os.getenv("DTM_B", 0.6))
    
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        seq_length = hidden_states.shape[1]

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        ############################### FastVID ###############################
        if seq_length > 1:
            device_type = hidden_states.device
            hidden_states_dim = hidden_states.shape[-1]
            frame_token_len = self.video_token_len // self.frame_num
            batchframe_indices = torch.arange(self.frame_num, device=device_type).unsqueeze(1)
            alltoken_indices = torch.arange(self.video_token_len, device=device_type).view(self.frame_num, frame_token_len) + self.video_start_idx
            
            video_hidden_states = hidden_states[:, self.video_start_idx:self.video_start_idx+self.video_token_len, :].squeeze(0)
            video_hidden_states = video_hidden_states.reshape(self.frame_num, frame_token_len, -1)
            frame_attn_weights = self.frame_attn_weights.reshape(self.frame_num, frame_token_len)

            ############ DySeg ############
            frame_global_features = self.frame_global_features
            frame_global_features = frame_global_features / frame_global_features.norm(dim=1, keepdim=True) 
            similarity_matrix = (frame_global_features[:-1] * frame_global_features[1:]).sum(dim=1)

            cut_indices_topk = torch.topk(similarity_matrix, self.fastvid_DySeg_c - 1, largest=False).indices
            cut_indices_cos = torch.nonzero(similarity_matrix < self.fastvid_DySeg_tau, as_tuple=False).squeeze(1)
            cut_indices = torch.unique(torch.cat([cut_indices_topk, cut_indices_cos])).sort().values
            padded = F.pad(cut_indices, (1, 1), value=-1)
            padded[-1] = self.frame_num - 1
            segment_sizes = padded.diff().tolist()
            
            ############ STPrune ############
            keep_indexs = ()
            keep_indexs += (torch.arange(self.video_start_idx,device=device_type),)
            keep_indexs += (torch.arange(self.video_start_idx+self.video_token_len,seq_length,device=device_type),)
            start_tokens = hidden_states[0,:self.video_start_idx,:]
            end_tokens = hidden_states[0,self.video_start_idx+self.video_token_len:,:]
            final_tokens = [start_tokens, end_tokens]
            
            frame_retain_num = int(frame_token_len * self.fastvid_retention_ratio)

            frame_salient_num = frame_retain_num - int(frame_retain_num * self.fastvid_STPrune_d)
            frm_salient_num_list = [frame_salient_num] * self.frame_num
            
            frm_context_num_list = torch.zeros(self.frame_num, dtype=torch.int, device=device_type)
            frame_context_num = frame_retain_num - frame_salient_num

            ############ Compute Anchor Token Distribution ############
            offset = 0
            for seg_i_len in segment_sizes:
                seg_context_num = frame_context_num * seg_i_len
                temp_num = (seg_i_len + self.fastvid_DTM_p - 1) //  self.fastvid_DTM_p
                cur_frm_context_num = seg_context_num // temp_num

                end = offset + seg_i_len
                seg_indices = torch.arange(seg_i_len - 1, -1, -1, device=device_type) 
                mask = (seg_indices % self.fastvid_DTM_p == 0)
            
                frm_context_num_list[offset:end][mask] = cur_frm_context_num
                offset = end

            ############ ATS ############
            salient_indexes = torch.topk(frame_attn_weights, frame_salient_num, dim=1).indices

            batch_indices = batchframe_indices.expand(-1, frame_salient_num)
            salient_tokens = video_hidden_states[batch_indices, salient_indexes]
            salient_global_indexes = alltoken_indices[batch_indices, salient_indexes]
            
            final_tokens.append(salient_tokens.view(-1, hidden_states_dim))
            keep_indexs += (salient_global_indexes.view(-1),)

            ############ Parallel Density Score Computation ############
            all_indices = torch.arange(frame_token_len, device=device_type).unsqueeze(0).expand(self.frame_num, -1)
            all_indices_mask = torch.ones_like(all_indices, dtype=torch.bool)
            all_indices_mask.scatter_(1, salient_indexes, False)
            filtered_indices = all_indices[all_indices_mask].view(self.frame_num, frame_token_len - frame_salient_num)
            
            batch_indices = batchframe_indices.expand(-1, frame_token_len - frame_salient_num)
            token_filtered = video_hidden_states[batch_indices, filtered_indices]
            alltoken_filtered_indices = alltoken_indices[batch_indices, filtered_indices]
            
            tmp_frm_hidden_states = token_filtered
            dist_matrix = torch.cdist(tmp_frm_hidden_states.float(), tmp_frm_hidden_states.float()) / (hidden_states_dim ** 0.5)

            dist_nearest, index_nearest = torch.topk(dist_matrix, k=4, dim=-1, largest=False)
            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            density = density + torch.rand(
                density.shape, device=device_type, dtype=density.dtype) * 1e-6
    
            density_mask = density[:, None, :] > density[:, :, None]
            density_mask = density_mask.type(tmp_frm_hidden_states.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist_0, index_parent = (dist_matrix * density_mask + dist_max * (1 - density_mask)).min(dim=-1)
    
            density_score = dist_0 * density

            sampled_indexs = torch.topk(density_score, k=frame_context_num, dim=-1).indices

            ############ DTM for Single-Frame Segment ############
            batch_indices = batchframe_indices.expand(-1, frame_context_num)
            frm_context_tokens = token_filtered[batch_indices, sampled_indexs]
            frm_context_global_indexes = alltoken_filtered_indices[batch_indices, sampled_indexs]
            
            to_be_merge_tokens = token_filtered / token_filtered.norm(dim=-1, keepdim=True)
            merge_target_tokens = to_be_merge_tokens[batch_indices, sampled_indexs]
    
            similarity = torch.bmm(to_be_merge_tokens, merge_target_tokens.transpose(1,2))
            assign_one_hot = torch.zeros(self.frame_num, frame_token_len - frame_salient_num, frame_context_num, dtype=token_filtered.dtype, device=device_type)
            assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)

            avg_weights = (1 / (assign_one_hot.sum(dim=1).unsqueeze(-1) + 1)).clamp(min=self.fastvid_DTM_beta)

            counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
            aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), token_filtered) / counts
            
            frm_context_tokens = avg_weights * frm_context_tokens + (1 - avg_weights) * aggregated_hidden

            context_for_frame_mask = (frm_context_num_list == frame_context_num)
            context_for_frame_num = context_for_frame_mask.sum()
            
            context_for_frame_tokens = frm_context_tokens[context_for_frame_mask]
            context_for_frame_global_indexes = frm_context_global_indexes[context_for_frame_mask]
            
            final_tokens.append(context_for_frame_tokens.view(-1, hidden_states_dim))
            keep_indexs += (context_for_frame_global_indexes.view(-1),)

            ############ DTM for Multi-Frame Segment ############
            idx_seg_start = 0
            for seg_i_len in segment_sizes:
                if seg_i_len > 1: 
                    cur_seg_context_num_list = frm_context_num_list[idx_seg_start:idx_seg_start+seg_i_len]
                    cur_seg_context_num = cur_seg_context_num_list[-1]
                    
                    cur_seg_target_mask = (cur_seg_context_num_list > frame_context_num)
                    cur_seg_target_num = cur_seg_target_mask.sum()

                    cur_seg_density_score = density_score[idx_seg_start:idx_seg_start+seg_i_len]
                    cur_seg_density_score = cur_seg_density_score[cur_seg_target_mask]
                    
                    cur_seg_token_filtered = token_filtered[idx_seg_start:idx_seg_start+seg_i_len]
                    cur_seg_token_target = cur_seg_token_filtered[cur_seg_target_mask]
                    cur_seg_token_filtered = cur_seg_token_filtered.view(1, -1, hidden_states_dim).expand(cur_seg_target_num,-1,-1)
                    
                    cur_seg_alltoken_indices = alltoken_filtered_indices[idx_seg_start:idx_seg_start+seg_i_len]
                    cur_seg_alltoken_indices = cur_seg_alltoken_indices[cur_seg_target_mask]
                   
                    sampled_indexs = torch.topk(cur_seg_density_score, k=cur_seg_context_num, dim=-1).indices
                    batch_indices = batchframe_indices[:cur_seg_target_num].expand(-1, cur_seg_context_num)
                    cur_context_tokens = cur_seg_token_target[batch_indices, sampled_indexs]
                    cur_context_global_indexes = cur_seg_alltoken_indices[batch_indices, sampled_indexs]
                    
                    to_be_merge_tokens = cur_seg_token_filtered / cur_seg_token_filtered.norm(dim=-1, keepdim=True)
                    merge_target_tokens = cur_context_tokens / cur_context_tokens.norm(dim=-1, keepdim=True)
            
                    similarity = torch.bmm(to_be_merge_tokens, merge_target_tokens.transpose(1,2))
                    assign_one_hot = torch.zeros(cur_seg_target_num, to_be_merge_tokens.shape[1], cur_seg_context_num, dtype=token_filtered.dtype, device=device_type)
                    assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
        
                    avg_weights = (1 / (assign_one_hot.sum(dim=1).unsqueeze(-1) + 1)).clamp(min=self.fastvid_DTM_beta)
        
                    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
                    aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), cur_seg_token_filtered) / counts
                    
                    cur_context_tokens = avg_weights * cur_context_tokens + (1 - avg_weights) * aggregated_hidden

                    final_tokens.append(cur_context_tokens.view(-1, hidden_states_dim))
                    keep_indexs += (cur_context_global_indexes.view(-1),)
                
                idx_seg_start += seg_i_len

            hidden_states = torch.cat(final_tokens, dim=0)
            keep_indexs = torch.cat(keep_indexs, dim=0)
        
            sorted_indexs = torch.argsort(keep_indexs)
            hidden_states = hidden_states[sorted_indexs].unsqueeze(0)
            keep_indexs = keep_indexs[sorted_indexs]
        
            if causal_mask is not None:
                causal_mask = causal_mask[:,:,:hidden_states.shape[1],:hidden_states.shape[1]]
            position_ids = keep_indexs.unsqueeze(0)
            cache_position = keep_indexs
            position_embeddings = (position_embeddings[0][:,keep_indexs,:], position_embeddings[1][:,keep_indexs,:])
        ##############################################################
        
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


    # Copied from transformers.models.phi3.modeling_phi3.Phi3Model._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

