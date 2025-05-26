import argparse
import os
import cv2
import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import trange

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.run_infinity import load_tokenizer, load_visual_tokenizer, load_transformer, gen_one_img, dynamic_resolution_h_w, h_div_w_templates

from enable_scale_kv_8b import enable_scale_kv

def main():
    
    model_path='weights/infinity_8b_weights'
    vae_path='weights/infinity_vae_d56_f8_14_patchify.pth'
    text_encoder_ckpt = 'weights/flan-t5-xl'
    args=argparse.Namespace(
        output_root="samples/scalekv_8b",
        pn='1M',
        model_path=model_path,
        cfg_insertion_layer=0,
        vae_type=14,
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_8b',
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=1,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/dev/shm',
        checkpoint_type='torch_shard',
        seed=0,
        bf16=1,
        enable_model_cache=1,
        cfg = 3,
        tau = 1.0,
        h_div_w = 1/1,
        enable_positive_prompt=0,
        split = None
    )
    
    # Initialize distributed process group
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Load COCO 2017 validation dataset
    dataset = load_dataset("phiyodr/coco2017", split="validation")
    
    # Handle split argument
    if args.split is not None:
        assert args.split[0] < args.split[1], "split[0] must be < split[1]"
        chunk_size = (5000 + args.split[1] - 1) // args.split[1]
        global_start = args.split[0] * chunk_size
        global_end = min((args.split[0] + 1) * chunk_size, 5000)
    else:
        global_start, global_end = 0, 5000

    # Calculate per-GPU indices
    total_samples = global_end - global_start
    per_gpu = (total_samples + world_size - 1) // world_size
    start_idx = global_start + rank * per_gpu
    end_idx = min(start_idx + per_gpu, global_end)

    # Load models
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)
    
    infinity = enable_scale_kv(infinity, window_size=16, max_capacity=650, kernel_size=5, pooling='maxpool')


    # Prepare scale schedule based on aspect ratio
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - args.h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    # Create output directory (synchronized)
    output_dir = args.output_root
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()

    # Generate images for assigned indices
    for i in trange(start_idx, end_idx, disable=rank != 0, desc=f"Rank {rank}"):
        sample = dataset[i]
        prompt = sample['captions'][0]
        
        # Generate image
        generated_image = gen_one_img(
            infinity,
            vae,
            text_tokenizer,
            text_encoder,
            prompt,
            g_seed=i,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=args.cfg,
            tau_list=args.tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            enable_positive_prompt=args.enable_positive_prompt,
        )

        # Save image
        output_path = os.path.join(output_dir, f"{i:04d}.png")
        cv2.imwrite(output_path, generated_image.cpu().numpy())

if __name__ == "__main__":
    main()