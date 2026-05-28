"""
# Adapted from https://huggingface.co/MILVLG/imp-v1-3b/blob/main/vision_encoder.py
"""

from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from functools import partial, reduce
from PIL import Image
import torch
import torch.utils.checkpoint
from torch import nn
import os
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from llava.utils import rank0_print

import torch.nn.functional as F
from tome.utils import bipartite_soft_matching, merge_wavg, merge_source


class SigLipImageProcessor:
    def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(384, 384), crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


class SigLipVisionConfig(PretrainedConfig):
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1152,
        image_mean=(0.5, 0.5, 0.5),
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.image_mean = image_mean

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SigLipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type " f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.")

        return cls.from_dict(config_dict, **kwargs)


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPVisionModelOutput with CLIP->SigLip
class SigLipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SigLipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:" f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        import os
        wrapper = os.environ.get("WRAPPER")
        if wrapper in ["visionzip", "tome"]:
            raw_key_states = key_states.clone()
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is" f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}")
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        if wrapper in ["visionzip", "tome"]:
            return attn_output, attn_weights, raw_key_states.mean(1)
        else:
            return attn_output, attn_weights, None


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->SigLip
class SigLipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->SigLip
class SigLipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights, metric = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # ------ ToMe ------ #
        import os
        wrapper = os.environ.get("WRAPPER")
        if wrapper in ["tome"]:
            if isinstance(self._tome_info["r"], list):
                r = self._tome_info["r"].pop(0)
            else:
                r = self._tome_info["r"][0]

            if r > 0:
                # Apply ToMe here
                merge, _ = bipartite_soft_matching(
                    metric,
                    r,
                    self._tome_info["class_token"],
                )
                if self._tome_info["trace_source"]:
                        self._tome_info["source"] = merge_source(
                            merge, hidden_states, self._tome_info["source"]
                        )
                hidden_states, self._tome_info["size"] = merge_wavg(merge, hidden_states, self._tome_info["size"])
                hidden_states = hidden_states.to(dtype=residual.dtype)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # if hasattr(self, "_info") and self._info["r"] is not None:
        #     r = self._info["r"][layer_idx]
        #     if r > 0:
        #         self.metric = metric

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SigLipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SigLipVisionConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->SigLip
class SigLipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SigLipEncoderLayer`].

    Args:
        config: SigLipVisionConfig
    """

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        # robustness: use defaults if env not set
        self.ema = float(os.environ.get("EMA", 0.9))
        self.mode = os.environ.get("MERGE_MODE", "mixing")
        assert self.mode in ['naive_mixing', 'mixing', 'merging'], "Wrong mode"
        self.tau = float(os.environ.get("T", 0.6))
        self.max_window = int(os.environ.get("M", 8))
        self.prune_layers = [int(x) for x in os.environ.get("PRUNE_LAYERS", "6,21,23").split(",")]

    # ---------------------------
    # compression
    # ---------------------------
    def compression(self, hidden_state: torch.Tensor, attn_weight: torch.Tensor,
                          tau: float = None, last_layer: bool = False, init_layer: bool = False):
        """
        Inputs:
            hidden_state: [T, S, D]
            attn_weight: [T, S, S] or [T, H, S, S] or other attn shapes (kept compatible)
        Returns:
            merged_frames: [T_new, S, D]
            merged_attn: [T_new, ...]  (same per-frame attn format)
            selected_groups: List[(start,end)]
        """
        hidden_state_normed = F.normalize(hidden_state, p=2, dim=-1)
        num_frames = hidden_state_normed.size(0)

        if num_frames <= 1:
            return hidden_state, attn_weight, [(0, num_frames - 1)]

        selected_frames = []
        start_idx = 0
        ref_feat = hidden_state_normed[0]  # [S, D]
        sim = 1.0

        # --- main loop ---
        for i in range(1, num_frames):
            curr_feat = hidden_state_normed[i]
            cur_sim = F.cosine_similarity(ref_feat, curr_feat, dim=-1).mean()  # scalar
            sim = self.ema * cur_sim + (1 - self.ema) * sim  # EMA update
            # sim_list.append(sim.item())
            if sim < tau or (i - start_idx + 1) >= self.max_window:
                end_idx = i - 1
                selected_frames.append((start_idx, end_idx))
                start_idx = i
                ref_feat = hidden_state_normed[i]
                sim = 1.0  # reset

        # --- append the last segment ---
        selected_frames.append((start_idx, num_frames - 1))

        # --- merge frames and attn ---
        merged_frames, merged_attn, _ = self.coarse_segment(
            hidden_state, attn_weight, selected_frames, hidden_state_normed
        )

        # --- optional: last layer refinement ---
        if last_layer:
            selected_frames_refined = []
            merged_frames_normed = F.normalize(merged_frames, p=2, dim=-1)
            start_idx = 0
            ref_feat = merged_frames_normed[0]
            sim = 1.0
            for i in range(1, merged_frames_normed.shape[0]):
                cur_feat = merged_frames_normed[i]
                cur_sim = F.cosine_similarity(ref_feat, cur_feat, dim=-1).mean()
                sim = self.ema * cur_sim + (1 - self.ema) * sim
                if sim < tau or (i - start_idx + 1) >= self.max_window:
                    end_idx = i - 1
                    selected_frames_refined.append((start_idx, end_idx))
                    start_idx = i
                    ref_feat = merged_frames[i]
                    sim = 1.0
            selected_frames_refined.append((start_idx, merged_frames.shape[0] - 1))
            selected_frames = selected_frames_refined

        return merged_frames, merged_attn, selected_frames

    # -------------------------
    # coarse_segment (robust)
    # -------------------------
    def coarse_segment(self, pooled_image_feat: torch.Tensor, attn_weights: torch.Tensor, groups, hidden_state_normed):
        """
        pooled_image_feat: [T, S, D]
        attn_weights: [T, ...] (commonly [T, H, S, S] or [T, S, S])
        groups: list of (start, end)
        returns:
            mixed_feats: [T_new, S, D]
            mixed_attns: [T_new, ...]
            mixed_index: list of sizes per group
        """
        out_feats, out_attns, mixed_index = [], [], []

        for (s, e) in groups:
            length = e - s + 1
            if length <= 0:
                continue
            if length == 1:
                f = pooled_image_feat[s].unsqueeze(0)  # [1,S,D]
                a = attn_weights[s].unsqueeze(0)
            elif length == 2:
                f = torch.stack([pooled_image_feat[s], pooled_image_feat[e]], dim=0)
                a = torch.stack([attn_weights[s], attn_weights[e]], dim=0)
            else:
                # take middle frames and fine-segment them
                mid_feat = pooled_image_feat[s+1:e]  # [L, S, D]
                mid_att = attn_weights[s+1:e]

                tau = F.cosine_similarity(hidden_state_normed[s], hidden_state_normed[e], dim=-1).mean(dim=-1) # scalar
                mf, ma = self.fine_segment(mid_feat, mid_att, tau=tau)
                f = torch.cat([pooled_image_feat[s].unsqueeze(0), mf, pooled_image_feat[e].unsqueeze(0)], dim=0)
                a = torch.cat([attn_weights[s].unsqueeze(0), ma, attn_weights[e].unsqueeze(0)], dim=0)

            out_feats.append(f)
            out_attns.append(a)
            mixed_index.append(f.size(0))

        # concat all groups along time
        if len(out_feats) == 0:
            return pooled_image_feat, attn_weights, []
        merged_feats = torch.cat(out_feats, dim=0)
        merged_attns = torch.cat(out_attns, dim=0)
        return merged_feats, merged_attns, mixed_index

    # -------------------------
    # fine_segment (vectorized & token-aware)
    # -------------------------
    def fine_segment(self, feat: torch.Tensor, attn: torch.Tensor, tau):
        """
        feat: [T, S, D]
        attn: [T, ...] where ... is either [S, S] or [H, S, S]
        tau: scalar or [S] (token-wise)
        returns:
            new_feat: [T_new, S, D]
            new_attn: [T_new, ...]
        Behavior:
            - If mode == naive_mixing: simply average all tokens in the group -> returns [1,S,D]
            - If mode == mixing: perform adjacent-pair token-aware merging using per-token similarities
        """
        if feat.size(0) < 2:
            return feat, attn

        if self.mode == "naive_mixing":
            mf = feat.mean(dim=0, keepdim=True)  # [1, S, D]
            ma = attn.mean(dim=0, keepdim=True)
            return mf, ma

        # mixing mode: compute token-level sims between adjacent frames
        T, S, _ = feat.shape
        feat_norm = F.normalize(feat, dim=-1)  # [T, S, D]
        sims = F.cosine_similarity(feat_norm[:-1], feat_norm[1:], dim=-1).mean(dim=-1)  # [T-1]

        mixed_feats, mixed_attn = [], []
        i = 0
        while i < T - 1:
            cur_sim = sims[i]
            if i + 1 < T - 1:
                next_sim = sims[i + 1]
                should_merge = cur_sim > tau and cur_sim > next_sim
            else:
                next_sim = 0.0
                should_merge = cur_sim > tau
            if should_merge:
                weight = cur_sim / (cur_sim + next_sim) if next_sim != 0 else 1.0
                mixed_feats.append(weight * feat[i] + (1 - weight) * feat[i+1])
                mixed_attn.append(weight * attn[i] + (1 - weight) * attn[i+1])
                i += 2
            else:
                mixed_feats.append(feat[i])
                mixed_attn.append(attn[i])
                i += 1
        mixed_feats = torch.stack(mixed_feats, dim=0)
        mixed_attn = torch.stack(mixed_attn, dim=0)
        return mixed_feats, mixed_attn

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        import os
        wrapper = os.environ.get("WRAPPER")

        hidden_states = inputs_embeds
        for layer_idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                if wrapper in ["visionzip", "holitom", "earlytom"]:
                    if layer_idx==len(self.layers) - 1:
                        encoder_states = encoder_states + (hidden_states,)
                else:
                    encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                    layer_idx=layer_idx,
                )

            hidden_states = layer_outputs[0]
            if wrapper in ["earlytom"]:
                if layer_idx in self.prune_layers:
                    hidden_states, attn_merge, selected_frames = self.compression(
                        hidden_states, layer_outputs[1], self.tau,
                        last_layer=True if layer_idx == self.prune_layers[-1] else False,
                        init_layer = True if layer_idx == self.prune_layers[0] else False
                    )
            else:
                selected_frames = None
            if output_attentions:
                if wrapper in ["visionzip", "holitom"]:
                    if layer_idx==len(self.layers) - 1:
                        all_attentions = all_attentions + (layer_outputs[1],)
                elif wrapper == "earlytom":
                    if layer_idx==len(self.layers) - 1:
                        all_attentions = all_attentions + (attn_merge,)
                else:
                    all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return (BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions), 
                  selected_frames)


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.head = SigLipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)
        
        encoder_outputs, selected_frames = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        ), selected_frames


class SigLipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        # ------  FastVid Modification ------ #
        import os
        wrapper = os.environ.get("WRAPPER")
        if wrapper in ["fastvid"]:
            hidden_state, attn_weights = self.attention(probe, hidden_state, hidden_state)
        else:
            hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        if wrapper in ["fastvid"]:
            return hidden_state[:, 0], attn_weights
        return hidden_state[:, 0]


class SigLipVisionModel(SigLipPreTrainedModel):
    config_class = SigLipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SigLipEncoderLayer"]

    def __init__(self, config: SigLipVisionConfig):
        super().__init__(config)

        self.vision_model = SigLipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, SigLipVisionModel

        >>> model = SigLipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SigLipVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.config = SigLipVisionConfig()

        self.vision_tower_name = vision_tower

        self.image_processor = SigLipImageProcessor()

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.vision_tower = SigLipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = nn.Identity()
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
                assert image_features.shape[-2] == 729
                image_features.append(image_feature)
        else:
            import os
            wrapper = os.environ.get("WRAPPER")
            if wrapper in ["visionzip"]:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
                attn_weights  = image_forward_outs[0].attentions[-1]
                hidden_states = image_forward_outs[0].hidden_states[-1]
                metric = self.vision_tower.vision_model.encoder.layers[-1].metric
                return hidden_states, attn_weights.mean(dim=1).mean(dim=1), metric, images.dtype
            if wrapper in ["holitom"]:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
                attn_weights  = image_forward_outs[0].attentions[-1]
                hidden_states = image_forward_outs[0].hidden_states[-1]
                return hidden_states, attn_weights.mean(dim=1).mean(dim=1), None, images.dtype
            if wrapper in ["earlytom"]:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
                attn_weights  = image_forward_outs[0].attentions[-1]
                hidden_states = image_forward_outs[0].hidden_states[-1]
                selected_frames = image_forward_outs[-1]
                return hidden_states, attn_weights, None, images.dtype, selected_frames
            if wrapper in ["tome"]:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features = image_forward_outs[0].hidden_states[-1]
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features = image_forward_outs[0].hidden_states[-1]
                assert image_features.shape[-2] == 729

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
        # return self.model_config["vision_cfg"]["image_size"] // self.model_config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config.image_size
    
# ------ FastVid ------ #
class SigLipVisionAbstract(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.config = SigLipVisionConfig()

        self.vision_tower_name = vision_tower
            
        if not delay_load:
            rank0_print(f"Loading vision abstract: {vision_tower}")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.vision_abstract = SigLipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

        del self.vision_abstract.vision_model.embeddings
        del self.vision_abstract.vision_model.encoder
            
        self.vision_abstract.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        last_hidden_state = self.vision_abstract.vision_model.post_layernorm(images)
        pooled_output, attn_weights = self.vision_abstract.vision_model.head(last_hidden_state)
        return pooled_output, attn_weights

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
        # return self.model_config["vision_cfg"]["image_size"] // self.model_config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config.image_size