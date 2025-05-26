import torch
import torch.nn.functional as F
import math
import warnings # Import warnings module

class ScaleKVCluster:
    def __init__(self, cache_sizes, window_size=16, base_capacity=121, kernel_size=5, pooling='maxpool', layer_idx=0):
        # --- Input Validation ---
        if not isinstance(window_size, int) or window_size <= 0:
             raise ValueError("window_size must be a positive integer.")
        if not isinstance(base_capacity, int) or base_capacity <= 0:
             raise ValueError("base_capacity must be a positive integer.")
        if not isinstance(kernel_size, int) or kernel_size <= 0:
             raise ValueError("kernel_size must be a positive integer.")
        if pooling not in ['maxpool', 'avgpool', 'pool']:
             raise ValueError("pooling must be 'maxpool' or 'avgpool'.")
        if not isinstance(layer_idx, int) or layer_idx < 0:
             raise ValueError("layer_idx must be a non-negative integer.")
        if not isinstance(cache_sizes, list) or not cache_sizes:
             raise ValueError("cache_sizes must be a non-empty list.")

        self.window_size = window_size
        self.base_capacity = base_capacity
        self.kernel_size = kernel_size
        # Standardize pooling name
        self.pooling = 'avgpool' if pooling == 'pool' else pooling
        self.layer_idx = layer_idx
        self.has_reached_base = False
        self.cache_sizes = cache_sizes
        self.SCALE = [1, 4, 16, 36, 64, 144, 256, 400, 576, 1024, 1600, 2304, 4096]


    def _select_window_indices(self, current_scale_size, current_scale_start, device, bsz, num_heads, key_len):
        if current_scale_size <= 0:
             return torch.zeros((bsz, num_heads, 0), device=device, dtype=torch.long)

        H_float = math.sqrt(current_scale_size)
        grid_size_float = math.sqrt(self.window_size)

        is_scale_square = (H_float == int(H_float))
        is_window_square = (grid_size_float == int(grid_size_float))

        if not (is_scale_square and is_window_square):
             warnings.warn(f"Layer {self.layer_idx}: Cannot form square grid. current_scale_size={current_scale_size} (sqrt={H_float}), window_size={self.window_size} (sqrt={grid_size_float}). Window selection might be suboptimal or empty.", UserWarning)
             actual_window_size_fallback = min(self.window_size, key_len)
             if actual_window_size_fallback > 0:
                 start_index = key_len - actual_window_size_fallback
                 indices = torch.arange(start_index, key_len, device=device, dtype=torch.long)
                 return indices.unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, -1)
             else:
                 return torch.zeros((bsz, num_heads, 0), device=device, dtype=torch.long)


        H = W = int(H_float)
        grid_size = int(grid_size_float)

        if H < grid_size or W < grid_size:
             warnings.warn(f"Layer {self.layer_idx}: Scale dimensions ({H}x{W}) too small for window grid ({grid_size}x{grid_size}). Selecting recent tokens instead.", UserWarning)
             actual_window_size_fallback = min(self.window_size, key_len)
             if actual_window_size_fallback > 0:
                  start_index = key_len - actual_window_size_fallback
                  indices = torch.arange(start_index, key_len, device=device, dtype=torch.long)
                  return indices.unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, -1)
             else:
                  return torch.zeros((bsz, num_heads, 0), device=device, dtype=torch.long)

        h_patch, w_patch = H // grid_size, W // grid_size
        all_h, all_w = [], []

        for i in range(grid_size):
            for j in range(grid_size):
                h_center = i * h_patch + (h_patch - 1) // 2
                w_center = j * w_patch + (w_patch - 1) // 2
                h_center = max(0, min(H - 1, h_center))
                w_center = max(0, min(W - 1, w_center))
                h_coord = torch.full((bsz, num_heads), h_center, device=device, dtype=torch.long)
                w_coord = torch.full((bsz, num_heads), w_center, device=device, dtype=torch.long)
                all_h.append(h_coord)
                all_w.append(w_coord)

        if not all_h:
            return torch.zeros((bsz, num_heads, 0), device=device, dtype=torch.long)

        h_coords = torch.stack(all_h, dim=2)
        w_coords = torch.stack(all_w, dim=2)


        global_indices = (h_coords * W + w_coords) + current_scale_start

        global_indices = torch.clamp(global_indices, min=0, max=key_len - 1)

        unique_indices_list = []
        max_unique_count = 0
        needs_padding = False
        for b in range(bsz):
             head_indices_list = []
             for h in range(num_heads):
                  unique_vals = torch.unique(global_indices[b, h])
                  head_indices_list.append(unique_vals)
                  current_count = unique_vals.size(0)
                  if h == 0 and b == 0:
                       max_unique_count = current_count
                  elif current_count != max_unique_count:
                       needs_padding = True
                       max_unique_count = max(max_unique_count, current_count)

             unique_indices_list.append(head_indices_list)

        final_stacked_indices = []
        target_count = min(self.window_size, max_unique_count)

        for b in range(bsz):
            padded_head_indices = []
            for h in range(num_heads):
                head_indices = unique_indices_list[b][h]
                current_count = head_indices.size(0)
                if current_count >= target_count:
                    padded_head_indices.append(head_indices[:target_count])
                else:
                    num_padding = target_count - current_count
                    pad_value = head_indices[-1].item() if current_count > 0 else 0
                    padding = torch.full((num_padding,), pad_value, dtype=torch.long, device=device)
                    padded_head_indices.append(torch.cat([head_indices, padding], dim=0))

            try:
                 final_stacked_indices.append(torch.stack(padded_head_indices, dim=0))
            except Exception as e:
                 warnings.warn(f"Layer {self.layer_idx} B{b}: Stacking padded unique indices failed: {e}. Falling back to raw indices.", UserWarning)
                 raw_b_indices = global_indices[b]
                 if raw_b_indices.size(1) > self.window_size:
                      raw_b_indices = raw_b_indices[:, :self.window_size]
                 elif raw_b_indices.size(1) < self.window_size:
                     num_padding = self.window_size - raw_b_indices.size(1)
                     pad_value = raw_b_indices[0, -1].item() if raw_b_indices.size(1) > 0 else 0
                     padding = torch.full((num_heads, num_padding), pad_value, dtype=torch.long, device=device)
                     raw_b_indices = torch.cat([raw_b_indices, padding], dim=1)

                 final_stacked_indices.append(raw_b_indices)


        final_indices = torch.stack(final_stacked_indices, dim=0)

        if final_indices.size(2) != self.window_size:
             warnings.warn(f"Layer {self.layer_idx}: Final window indices shape {final_indices.shape} has {final_indices.size(2)} tokens, expected {self.window_size}. Check padding/unique logic.", UserWarning)
             if final_indices.size(2) > self.window_size:
                 final_indices = final_indices[:, :, :self.window_size]
             elif final_indices.size(2) < self.window_size:
                 num_padding = self.window_size - final_indices.size(2)
                 pad_value = final_indices[0, 0, -1].item() if final_indices.numel() > 0 else 0
                 padding = torch.full((bsz, num_heads, num_padding), pad_value, dtype=torch.long, device=device)
                 final_indices = torch.cat([final_indices, padding], dim=2)


        return final_indices


    def update_kv(self, key_states, query_states, value_states, scale_ind):
        device = key_states.device
        bsz, num_heads, key_len, head_dim = key_states.shape

        if key_len <= self.base_capacity and not self.has_reached_base:
            return key_states, value_states
        else:
            self.has_reached_base = True

        original_scale_ind = scale_ind
        if scale_ind == 12:
             scale_ind_for_capacity = 11
             self.has_reached_base = False
        else:
             scale_ind_for_capacity = scale_ind

        if not (0 <= original_scale_ind < len(self.SCALE)):
             warnings.warn(f"Layer {self.layer_idx}: Original scale_ind {original_scale_ind} out of bounds for SCALE (len {len(self.SCALE)}). Clamping.")
             original_scale_ind = max(0, min(original_scale_ind, len(self.SCALE) - 1))

        cache_sizes_idx = scale_ind_for_capacity + 2
        if not (0 <= cache_sizes_idx < len(self.cache_sizes)):
             warnings.warn(f"Layer {self.layer_idx}: Calculated cache_sizes index {cache_sizes_idx} is out of bounds (len {len(self.cache_sizes)}). Adjusting.")
             cache_sizes_idx = max(0, min(cache_sizes_idx, len(self.cache_sizes) - 1))

        if self.layer_idx >= len(self.cache_sizes[cache_sizes_idx]):
             warnings.warn(f"Layer {self.layer_idx}: layer_idx {self.layer_idx} is out of bounds for cache_sizes[{cache_sizes_idx}] (len={len(self.cache_sizes[cache_sizes_idx])}). Using base_capacity.")
             self.max_capacity = self.base_capacity
        else:
            self.max_capacity = self.cache_sizes[cache_sizes_idx][self.layer_idx]

        if key_len <= self.max_capacity:
            return key_states, value_states

        current_scale_size = self.SCALE[original_scale_ind]
        current_scale_start = max(0, key_len - current_scale_size)
        current_scale_size_actual = key_len - current_scale_start

        selected_window_global = self._select_window_indices(
            current_scale_size_actual, current_scale_start, device, bsz, num_heads, key_len
        )
        actual_window_size = selected_window_global.size(2)

        if self.max_capacity < actual_window_size:
             warnings.warn(f"Layer {self.layer_idx}: Determined max_capacity ({self.max_capacity}) is less than actual selected window size ({actual_window_size}). "
                           f"This likely indicates an issue with cache_sizes configuration. Will only keep the {actual_window_size} window tokens.", UserWarning)
             self.max_capacity = actual_window_size
             keep_size = 0

        elif self.window_size > 0 and actual_window_size == 0:
             warnings.warn(f"Layer {self.layer_idx}: Window selection returned 0 indices (expected {self.window_size}). "
                           f"Proceeding by selecting top-{self.max_capacity} tokens based on pooling only.", UserWarning)
             last_query = query_states[:, :, -1:, :]
             attn_weights_approx = torch.matmul(last_query, key_states.transpose(2, 3)) / math.sqrt(head_dim)
             attn_scores = attn_weights_approx.squeeze(2)

             if self.pooling == 'avgpool':
                 pooled_scores = F.avg_pool1d(attn_scores, self.kernel_size, padding=self.kernel_size//2, stride=1)
             else:
                 pooled_scores = F.max_pool1d(attn_scores, self.kernel_size, padding=self.kernel_size//2, stride=1)

             _, final_indices = pooled_scores.topk(self.max_capacity, dim=-1)

        else:
            if actual_window_size < self.window_size:
                 warnings.warn(f"Layer {self.layer_idx}: Selected only {actual_window_size} unique window indices (target {self.window_size}). Proceeding.", UserWarning)

            keep_size = self.max_capacity - actual_window_size

            if keep_size > 0:
                seq_len = query_states.size(2)
                offset = key_len - seq_len
                local_indices = torch.clamp(selected_window_global - offset, 0, seq_len - 1)

                if local_indices.shape != selected_window_global.shape:
                     raise RuntimeError(f"Shape mismatch: local_indices {local_indices.shape}, selected_window_global {selected_window_global.shape}")

                window_query = torch.gather(
                    query_states, 2,
                    local_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                )

                attn_weights = torch.matmul(window_query, key_states.transpose(2, 3)) / math.sqrt(head_dim)

                mask = torch.ones((bsz, num_heads, 1, key_len), device=device, dtype=torch.bool)

                mask.scatter_(dim=3, index=selected_window_global.unsqueeze(2), value=False)
                attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

                attn_scores = attn_weights.sum(dim=2)

                if self.pooling == 'avgpool':
                    pooled_scores = F.avg_pool1d(attn_scores, self.kernel_size, padding=self.kernel_size//2, stride=1)
                else:
                    pooled_scores = F.max_pool1d(attn_scores, self.kernel_size, padding=self.kernel_size//2, stride=1)

                masked_pooled_scores = pooled_scores.clone()
                masked_pooled_scores.scatter_(dim=2, index=selected_window_global, value=float('-inf'))

                num_non_window = key_len - actual_window_size
                actual_keep_size = min(keep_size, num_non_window)
                if actual_keep_size < keep_size:
                     warnings.warn(f"Layer {self.layer_idx}: Requested keep_size {keep_size} > available non-window tokens {num_non_window}. Keeping only {actual_keep_size}.", UserWarning)
                if actual_keep_size <= 0:
                     selected_rest_global = torch.zeros((bsz, num_heads, 0), dtype=torch.long, device=device)
                else:
                    _, selected_rest_global = masked_pooled_scores.topk(actual_keep_size, dim=-1)

                combined = torch.cat([selected_rest_global, selected_window_global], dim=-1)

            else: 
                 combined = selected_window_global

            sorted_combined, _ = torch.sort(combined, dim=-1)
            final_indices = sorted_combined

            final_count = final_indices.size(2)
            expected_count = self.max_capacity
            if final_count != expected_count:
                 warnings.warn(f"Layer {self.layer_idx}: Final index count ({final_count}) != expected max_capacity ({expected_count}). Check logic.", UserWarning)
                 if final_count > expected_count:
                      final_indices = final_indices[:, :, :expected_count]

        if final_indices.numel() == 0:
             warnings.warn(f"Layer {self.layer_idx}: No indices selected for keeping. Returning empty tensors.", UserWarning)
             return torch.zeros_like(key_states[:,:,:0,:]), torch.zeros_like(value_states[:,:,:0,:])

        final_indices_expanded = final_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        final_indices_expanded = torch.clamp(final_indices_expanded, 0, key_len - 1)

        try:
            sorted_k = torch.gather(key_states, 2, final_indices_expanded)
            sorted_v = torch.gather(value_states, 2, final_indices_expanded)
        except IndexError as e:
             raise IndexError(f"Layer {self.layer_idx}: Gather operation failed. Indices might be out of bounds. "
                             f"KeyLen={key_len}, Max Index Attempted={final_indices.max().item()}, "
                             f"Indices Shape={final_indices.shape}. Error: {e}")

        return sorted_k, sorted_v