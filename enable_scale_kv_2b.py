import types
import torch

from infinity.models.basic import SelfAttention
from scale_kv import ScaleKVCluster

import json

def get_cache_sizes():
    with open('budgets/2b_10%_budget.json', 'r') as f:
        scale_budgets = json.load(f)

    cache_sizes = [None] * 14
    
    for scale_number in range(9, 14):
        scale_key = f"Scale {scale_number}"
        if scale_key in scale_budgets:
            scale_data = scale_budgets[scale_key]
            max_cache = scale_data["Max"]
            min_cache = scale_data["Min"]
            block_cache = []
            for block_num in range(32):
                block_key = f"Block {block_num}"
                value = scale_data[block_key]
                cache = max_cache if value == 1 else min_cache
                block_cache.append(cache)
            cache_sizes[scale_number] = block_cache
        else:
            cache_sizes[scale_number] = []
    
    return cache_sizes

def enable_scale_kv(model, window_size=16, max_capacity=121, 
                             kernel_size=5, pooling='maxpool'):
    
    def patched_self_attn_forward(self, x, attn_bias_or_two_vector=None,
                                 attn_fn=None, scale_schedule=None,
                                 rope2d_freqs_grid=None, scale_ind=0):
        output = self._original_forward(
            x, attn_bias_or_two_vector, attn_fn,
            scale_schedule, rope2d_freqs_grid, scale_ind
        )
        
        if self.caching and self.cached_k is not None and self.cached_v is not None:
            B, L, C = x.shape
            qkv = self.mat_qkv(x)
            q = qkv.view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)[0]
            
            new_k, new_v = self.kv_cluster.update_kv(
                self.cached_k,
                q,
                self.cached_v,
                scale_ind
            )
            
            self.cached_k = new_k
            self.cached_v = new_v

        return output
    
    cache_sizes = get_cache_sizes()


    layer_idx = 0
    for module in model.modules():
        if isinstance(module, SelfAttention):
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
                module.forward = types.MethodType(patched_self_attn_forward, module)

            module.kv_cluster = ScaleKVCluster(
                cache_sizes = cache_sizes,
                window_size=window_size,
                base_capacity=max_capacity,
                kernel_size=kernel_size,
                pooling=pooling,
                layer_idx=layer_idx
            )
            layer_idx += 1

    return model