from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random
from ipdb import set_trace as st
from torch.cuda import nvtx


class LlavaMetaForCausalLM_earlytom(ABC):

    def encode_images(self, images):
        image_features, _ = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features


    def encode_images_multi(self, images):
        image_features, attn_weights, metric, images_dtype, selected_frames = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        attn_weights = attn_weights.mean(dim=(1,2))
        image_features = self.get_model().mm_projector(image_features)
        return image_features, attn_weights, metric, images_dtype, selected_frames


    def cluster_dpc_knn(self, x, cluster_num, k=7):
        with torch.no_grad():
            batch_size, seq_len, embed_dim = x.shape

            dist_matrix = torch.cdist(x.float(), x.float()) / (embed_dim ** 0.5)    # (batch_size, seq_len, seq_len)

            # get local density
            dist_nearest, index_nearest = torch.topk(dist_matrix, k, dim=-1, largest=False) # (batch_size, seq_len, k)
            density = (-(dist_nearest ** 2).mean(dim=-1)).exp() # (batch_size, seq_len)
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(
                density.shape, device=density.device, dtype=density.dtype) * 1e-6

            # get distance indicator
            mask = (density[:, None, :] > density[:, :, None]).type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1).values[:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # select the cluster center according to the score
            score = dist * density
            _, index_center = score.topk(cluster_num, dim=-1)

            return index_center, dist_matrix


    def merge_tokens_by_clustering(self, feat, target_indices, dist_matrix, cluster_num, Beta):
        batch_size, seq_len, embed_dim = feat.shape
        all_indices = torch.arange(seq_len, device=feat.device)
        all_indices = all_indices.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
        non_target_indices = torch.zeros((batch_size, seq_len-cluster_num), dtype=torch.long, device=feat.device)
        for b in range(batch_size):
            non_target_mask = ~torch.isin(all_indices[b], target_indices[b])
            non_target_indices[b] = all_indices[b][non_target_mask]
        # non_target_indices (batch_size, seq_len-cluster_num)

        non_target_feat = torch.gather(
            feat,
            dim=1,
            index=non_target_indices.unsqueeze(-1).expand(-1, -1, feat.size(-1))
        )   # (batch_size, seq_len-cluster_num, embed_dim)

        dist_matrix = torch.gather(
            dist_matrix,
            dim=1,
            index=non_target_indices.unsqueeze(-1).expand(-1, -1, dist_matrix.size(-1))
        )   # (batch_size, seq_len-cluster_num, seq_len)
        dist_matrix = torch.gather(
            dist_matrix,
            dim=2,
            index=target_indices.unsqueeze(1).expand(-1, dist_matrix.size(1), -1)
        )   # (batch_size, seq_len-cluster_num, cluster_num)

        idx_cluster = torch.argmin(dist_matrix, dim=-1) # (batch_size, seq_len-cluster_num)

        cluster_tokens = []
        for b in range(batch_size):
            batch_tokens = []
            for i in range(cluster_num):
                mask = (idx_cluster[b] == i)
                if mask.any():
                    cluster_features = non_target_feat[b][mask]
                    import os
                    if os.environ.get("NO_BETA", "0") == "0":
                        # rank0_print("USE_BETA")
                        cluster_means = cluster_features.mean(dim=0)
                        batch_tokens.append(Beta * feat[b][target_indices[b][i]] + (1 - Beta) * cluster_means)
                    else:
                        # rank0_print("NO_BETA")
                        all_features = torch.cat([feat[b][target_indices[b][i]].unsqueeze(0), cluster_features], dim=0)
                        batch_tokens.append(all_features.mean(dim=0))
                else:
                    batch_tokens.append(feat[b][target_indices[b][i]])
            cluster_tokens.append(torch.stack(batch_tokens))
        cluster_tokens = torch.stack(cluster_tokens)  # shape: (batch_size, cluster_num, embed_dim)

        return cluster_tokens


    def add_newline_token(self, feat, pos, grid_size, newline_token):
        row_pos = pos // grid_size
        expanded_feat_list = []
        for cur_feat, cur_row_pos in zip(feat, row_pos):
            expanded_feat = []
            for row in range(grid_size):
                find_row_feat = cur_feat[cur_row_pos == row]
                if len(find_row_feat) > 0:
                    expanded_feat.append(torch.cat((find_row_feat, newline_token), dim=0))
                else:
                    expanded_feat.append(find_row_feat)
            batch_feat = torch.cat(expanded_feat, dim=0)
            expanded_feat_list.append(batch_feat)

        image_feat = torch.cat(expanded_feat_list, dim=0)
        return image_feat


    # --- Modified --- #
    def merge_tokens_by_attention_frame(self, feat, attn, pos, retain_ratio, D, Beta, K):
        batch_size, seq_len, embed_dim = feat.shape
        dominant_num = round(math.ceil(seq_len * retain_ratio) * (1 - D))
        contextual_num = math.ceil(seq_len * retain_ratio) - dominant_num

        ## Dominant Visual Tokens
        if dominant_num > 0:
            all_indices = attn.topk(dominant_num, dim=1).indices
            mask = torch.ones_like(feat[:, :, 0], dtype=torch.bool, device=feat.device).scatter_(1, all_indices, False)  # (batch_size, seq_len) False means retained tokens
            dominant_tokens = feat.masked_select(~mask.unsqueeze(-1)).view(batch_size, dominant_num, embed_dim)
            dominant_pos = pos.masked_select(~mask).view(batch_size, dominant_num)
        else:
            mask = torch.ones_like(feat[:, :, 0], dtype=torch.bool, device=feat.device)
            dominant_tokens = torch.empty((-1, embed_dim), device=feat.device)
            dominant_pos = torch.empty((batch_size, 0), device=feat.device)

        ## Contextual Visual Tokens
        if contextual_num > 0:
            feat_filtered = feat.masked_select(mask.unsqueeze(-1)).view(batch_size, seq_len - dominant_num, embed_dim) 
            contextual_pos = pos.masked_select(mask.unsqueeze(-1)).view(batch_size, seq_len - dominant_num)
            target_indices, dist_matrix = self.cluster_dpc_knn(feat_filtered, contextual_num, k=min(K,contextual_num))
            target_indices = torch.sort(target_indices, dim=-1)[0]
            contextual_pos = torch.stack([contextual_pos[b][target_indices[b]] for b in range(batch_size)])
            contextual_tokens = self.merge_tokens_by_clustering(feat_filtered, target_indices, dist_matrix, contextual_num, Beta)
        else:
            contextual_tokens = torch.empty((batch_size, 0, embed_dim), device=feat.device)
            contextual_pos = torch.empty((batch_size, 0), device=feat.device)

        image_feat = []
        image_pos = []
        for b in range(batch_size):
            batch_tokens = torch.cat([dominant_tokens[b], contextual_tokens[b]], dim=0)
            batch_pos = torch.cat([dominant_pos[b], contextual_pos[b]], dim=0)
            image_feat.append(batch_tokens)
            image_pos.append(batch_pos)
        image_feat = torch.stack(image_feat)
        image_pos = torch.stack(image_pos)
        
        return image_feat, image_pos


    def merge_tokens_by_local_window(self, feat, attn, pos, retain_ratio, local_k=1, num_windows=None):
        B, N, C = feat.shape
        assert attn.shape == (B, N)
        device = feat.device
        target_n = max(1, round(N * retain_ratio))

        # ---------- 1. Automatically infer num_windows ----------
        if num_windows is None:
            num_windows = max(1, math.ceil(target_n / local_k))

        # Ensure each window can at least take local_k tokens
        min_win_size = local_k
        if N // num_windows < min_win_size:
            local_k = max(1, N // num_windows)
            num_windows = math.ceil(target_n / local_k)

        # ---------- 2. Calculate the size of each window (non-uniform but ensures coverage) ----------
        base_size = N // num_windows
        remainder = N % num_windows
        window_sizes = [base_size + 1 if i < remainder else base_size for i in range(num_windows)]

        # ---------- 3. Split using torch.split (automatically handles boundaries) ----------
        feat_windows = torch.split(feat, window_sizes, dim=1)   # list[(B, sz, C)]
        attn_windows = torch.split(attn, window_sizes, dim=1)   # list[(B, sz)]
        pos_windows = torch.split(pos, window_sizes, dim=1)   # list[(B, sz)]

        selected = []
        selected_pos = []
        cur_pos = 0
        for wf, wa in zip(feat_windows, attn_windows):
            win_size = wf.shape[1]
            k = min(local_k, win_size)
            if k == 0:
                continue

            # Local top-k (by attn)
            _, idx_local = wa.topk(k=k, dim=1)
            idx_global = idx_local + cur_pos
            cur_pos += win_size

            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, k)
            token = feat[batch_idx, idx_global, :]
            pos_ = pos[batch_idx, idx_global]

            selected.append(token)
            selected_pos.append(pos_)

        # ---------- 4. Merge & Precisely match target_n ----------
        if not selected:
            return feat[:, :target_n, :], pos[:, target_n]

        merged = torch.cat(selected, dim=1)
        merged_pos = torch.cat(selected_pos, dim=1)

        cur_n = merged.shape[1]

        if cur_n > target_n:
            step = cur_n / target_n
            idx = torch.round(torch.arange(0, cur_n, step, device=device)).long()
            merged = merged[:, idx, :]
            merged_pos = merged_pos[:, idx]
        elif cur_n < target_n:
            pad = target_n - cur_n
            pad_tok = merged[:, -1:, :].expand(-1, pad, -1)
            pad_pos = merged_pos[:, -1:].expand(-1, pad)

            merged = torch.cat([merged, pad_tok], dim=1)
            merged_pos = torch.cat([merged_pos, pad_pos], dim=1)

        return merged, merged_pos


    def spatial_compression(self, static_feat, static_attn, dynamic_feat, dynamic_attn, 
                            static_pos, dynamic_pos, window_size, retain_ratio, D, Beta, K, 
                            images_dtype, mm_newline_position
                        ):
        
        has_static = static_feat is not None
        newline_token = self.model.image_newline[None].to(
            static_feat.device if has_static else dynamic_feat.device
        ) if mm_newline_position == "grid" else None

        grid_size = int(math.sqrt(dynamic_feat.shape[1]))
        # if has_static:
        #     grid_size = int(math.sqrt(dynamic_feat.shape[1] + static_feat.shape[0]))
        # else:
        #     grid_size = int(math.sqrt(dynamic_feat.shape[1]))
        if window_size <= 2:
            dynamic_feat, dynamic_pos = self.merge_tokens_by_attention_frame(dynamic_feat, dynamic_attn, dynamic_pos, retain_ratio, D, Beta, K)
            if mm_newline_position != "grid":
                feat = dynamic_feat.flatten(0, 1)  # (seq_len, embed_dim)
            else:
                feat = dynamic_feat.flatten(0, 1)
                pos = dynamic_pos.flatten(0, 1)
                feat = self.add_newline_token(feat, pos, grid_size, newline_token).to(images_dtype)
            return feat.to(images_dtype)
        else:
            dynamic_feat, dynamic_pos = self.merge_tokens_by_attention_frame(dynamic_feat, dynamic_attn, dynamic_pos, retain_ratio, D, Beta, K)
            static_feat, static_pos = self.merge_tokens_by_local_window(static_feat, static_attn, static_pos, retain_ratio)

            if mm_newline_position != "grid":
                if has_static:
                    feat = torch.cat([dynamic_feat[0,:,:], static_feat.flatten(0, 1), dynamic_feat[-1,:,:]])
                else:
                    feat = dynamic_feat.flatten(0,1)
            else:
                if has_static:
                    first_feat = dynamic_feat[0, :, :]
                    middle_feat = static_feat.flatten(0, 1)
                    end_feat = dynamic_feat[-1, :, :]
                    feat = torch.cat([first_feat, middle_feat, end_feat], dim=0)
                    pos = torch.cat([dynamic_pos[0,: ], static_pos.flatten(0, 1), dynamic_pos[-1,: ]], dim=0)
                    feat = self.add_newline_token(first_feat, pos, grid_size, newline_token)
                else:
                    pos = dynamic_feat.shape[0] * dynamic_feat.shape[1]
                    feat = self.add_newline_token(dynamic_feat, pos, grid_size, newline_token)
            return feat.to(images_dtype)

    def divided_static_dynamic(self, image_feat, attn_weights, selected_frames):

        T, S, D = image_feat.shape
        device = image_feat.device
        all_pos_indices = torch.arange(S, device=device).unsqueeze(0)  # (1, S)

        static_feat_list, static_attn_list, static_pos_list = [], [], []
        dynamic_feat_list, dynamic_attn_list, dynamic_pos_list = [], [], []

        for start, end in selected_frames:
            group_frames = end - start + 1

            if group_frames <= 2:
                dyn_feat = image_feat[start:end+1]                    # (≤2, S, D)
                dyn_attn = attn_weights[start:end+1]                  # (≤2, S)
                dyn_pos  = all_pos_indices.expand(group_frames, -1)   # (≤2, S)

                static_feat_list.append(None)
                static_attn_list.append(None)
                static_pos_list.append(None)

                dynamic_feat_list.append(dyn_feat)
                dynamic_attn_list.append(dyn_attn)
                dynamic_pos_list.append(dyn_pos)

            else:
                dyn_feat = torch.stack([image_feat[start], image_feat[end]], dim=0)   # (2, S, D)
                dyn_attn = torch.stack([attn_weights[start], attn_weights[end]], dim=0)
                dyn_pos  = all_pos_indices.expand(2, -1)                              # (2, S)

                static_feat = image_feat[start+1:end]                  # (mid, S, D)
                static_attn = attn_weights[start+1:end]                # (mid, S)
                static_pos  = all_pos_indices.expand(static_feat.shape[0], -1)

                static_feat_list.append(static_feat)
                static_attn_list.append(static_attn)
                static_pos_list.append(static_pos) 

                dynamic_feat_list.append(dyn_feat)
                dynamic_attn_list.append(dyn_attn)
                dynamic_pos_list.append(dyn_pos)

        return (static_feat_list, static_attn_list, static_pos_list,
                dynamic_feat_list, dynamic_attn_list, dynamic_pos_list)
    

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        import os
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if images is None and os.getenv("INNER_k") is not None and os.getenv("INNER_r") is not None:
                self.model.image_token_posi = [-1]
                self.model.image_tokens = [0]
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        if type(images) is list or images.ndim == 5:
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")
            
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            init_frames = concat_images.shape[0]
            encoded_image_features, attn_weights, _, images_dtype, selected_frames = self.encode_images_multi(concat_images)
            
            # ------ EarlyTom Parameters ------ #
            retain_ratio = float(os.environ.get("RETAIN_RATIO", 0.1))
            tau = float(os.environ.get("T", 0.8))
            Beta = float(os.environ.get("BETA", 0.6))
            D = float(os.environ.get("D", 0))
            K = int(os.environ.get("K", 7))
            NO_BETA = os.environ.get("NO_BETA", "1")
            rank0_print(f"retain_ratio: {retain_ratio}, tau: {tau}, Beta: {Beta}, D: {D}, K: {K}, NO_BETA: {NO_BETA}")
            encoded_image_features = torch.split(encoded_image_features, encoded_image_features.shape[0])
            image_features = []

            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:

                    # ------ Spatial Merging ------ #
                    pooled_image_feat = self.get_2dPool(image_feat)
                    attn_weights = attn_weights.unsqueeze(-1)
                    attn_weights = self.get_2dPool(attn_weights)
                    attn_weights = attn_weights.squeeze(-1)

                    reduced_frames, seq_len, _ = pooled_image_feat.shape
                    rank0_print(f"Selected frames: {selected_frames}")
                    total_tokens = init_frames * seq_len
                    reduced_tokens = (init_frames - reduced_frames) * seq_len
                    retain_ratio = min(retain_ratio / ((total_tokens - reduced_tokens)/total_tokens), 1)
                    rank0_print(f"Initial total frames: {init_frames}, reduced frames: {int(init_frames - reduced_frames)}")
                    rank0_print(f"After static pruning, retain ratio: {retain_ratio}")
                    static_feat, static_attn, static_pos, dynamic_feat, dynamic_attn, dynamic_pos \
                        = self.divided_static_dynamic(pooled_image_feat, attn_weights, selected_frames)
 
                    segment_features = []
                    for idx, (start, end) in enumerate(selected_frames):
                        window_size = end - start + 1
                        # Our fixed spatial compression
                        segment_features.append(
                            self.spatial_compression(
                                static_feat[idx], static_attn[idx], dynamic_feat[idx], dynamic_attn[idx],
                                static_pos[idx], dynamic_pos[idx],
                                window_size, retain_ratio, D, Beta, K, images_dtype, mm_newline_position
                            )
                        )
                    image_features.append(torch.cat(segment_features, dim=0))
                else:
                    image_features.append(image_feat)
            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    if image_idx in video_idx_in_batch:  # video operations
                        if mm_newline_position == "grid":
                            # # Grid-wise
                            # image_feature = self.add_token_per_grid(image_feature)
                            # if getattr(self.config, "add_faster_video", False):
                            #     faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                            #     # Add a token for each frame
                            #     concat_slow_fater_token = []
                            #     # import pdb; pdb.set_trace()
                            #     for _ in range(image_feature.shape[0]):
                            #         if _ % self.config.faster_token_stride == 0:
                            #             concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                            #         else:
                            #             concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                            #     # import pdb; pdb.set_trace()
                            #     image_feature = torch.cat(concat_slow_fater_token)

                            #     # print("!!!!!!!!!!!!")

                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))

                        elif mm_newline_position == "one_token":
                            # one-token
                            # FIXME if you use token compression method, please
                            # image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        if os.getenv("INNER_k") is not None and os.getenv("INNER_r") is not None:
            image_token_posi = []
            prompt_len = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            if os.getenv("INNER_k") is not None and os.getenv("INNER_r") is not None:
                image_index = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                if image_index == []:
                    image_token_posi.append(-1)
                else:
                    image_token_posi.append(image_index[0])

                # record input instruction length in inference mode
                if not self.training:
                    if image_index == []:
                        prompt_len.append(cur_input_ids.shape[0])
                    else:
                        prompt_len.append(cur_input_ids.shape[0] - 1)   # consider image place holder

            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])

            text_token_count = sum([x.shape[0] for x in cur_labels_noim])
            vision_token_count = len(image_features[cur_image_idx])
            rank0_print(f"Batch {batch_idx}: Text tokens: {text_token_count} Original Vision tokens: {vision_token_count}")

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)


        if os.getenv("INNER_k") is not None and os.getenv("INNER_r") is not None:
            self.model.image_token_posi = image_token_posi
            self.model.prompt_len = prompt_len
            self.model.image_tokens = [image_feature.shape[0] for image_feature in image_features]

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
