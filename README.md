<div align="center">
<h1>ScaleKV: Memory-Efficient Visual Autoregressive Modeling with Scale-Aware KV Cache Compression</h1>

  <div align="center">
  <a href="https://opensource.org/license/mit-0">
    <img alt="MIT" src="https://img.shields.io/badge/License-MIT-4E94CE.svg">
  </a>
  <a href="https://arxiv.org/abs/2505.19602">
    <img src="https://img.shields.io/badge/Paper-Arxiv-darkred.svg" alt="Paper">
  </a>
</div>
</div>

> **Memory-Efficient Visual Autoregressive Modeling with Scale-Aware KV Cache Compression**   
> [Kunjun Li](https://kunjun-li.github.io/), [Zigeng Chen](https://github.com/czg1225), [Cheng-Yen Yang](https://yangchris11.github.io/), [Jenq-Neng Hwang](https://people.ece.uw.edu/hwang/)   
> [University of Washington](https://www.washington.edu/)，[National University of Singapore](https://nus.edu.sg/)

<!-- ![figure](assets/intro.png) -->
<div align="center">
  <img src="assets/teaser.png" width="100%" ></img>
  <br>
</div>
<br>


## 💡 Introduction
We propose Scale-Aware KV Cache (ScaleKV), a novel KV Cache compression framework tailored for VAR’s next-scale prediction paradigm. ScaleKV leverages on two critical observations: varying cache demands across transformer layers and distinct attention patterns at different scales. Based on these insights, we categorizes transformer layers into two functional groups termed drafters and refiners, implementing adaptive cache management strategies based on these roles and optimize multi-scale inference by identifying each layer's function at every scale, enabling adaptive cache allocation that aligns with specific computational demands of each layer. On Infinity-8B, it achieves 10x memory reduction from 85 GB to 8.5 GB with negligible quality degradation (GenEval score remains at 0.79 and DPG score marginally decreases from 86.61 to 86.49).

<!-- ![figure](assets/intro.png) -->
<div align="center">
  <img src="assets/overview.png" width="100%" ></img>
  <img src="assets/method.png" width="100%" ></img>
  <br>
</div>
<br>


## 🔥Updates
* 🔥 **May 26, 2025**: Our paper is available now!
* 🔥 **May 25, 2025**: Code repo is released! Arxiv paper will come soon!

## 🔧  Installation:
### Reequirements
```bash
pip install -r requirements.txt
```

### Model Checkpoints
Download google flan-t5-xl:
```bash
pip install -U huggingface_hub
huggingface-cli download google/flan-t5-xl --local-dir ./weights/flan-t5-xl
```

Download Infinity-2B:
```bash
huggingface-cli download FoundationVision/Infinity --include "infinity_2b_reg.pth" --local-dir ./weights/
huggingface-cli download FoundationVision/Infinity --include "infinity_vae_d32reg.pth" --local-dir ./weights/
```

Download Infinity-8B:
```bash
huggingface-cli download FoundationVision/Infinity --include "infinity_8b_weights/**" --local-dir ./weights/infinity_8b_weights
huggingface-cli download FoundationVision/Infinity --include "infinity_vae_d56_f8_14_patchify.pth" --local-dir ./weights/

```

## ⚡ Quick Start:

Sample images with ScaleKV-Compressed Infinity-8B (10% KV Cache):
```python
python infer_8B.py
```

Sample images with ScaleKV-Compressed Infinity-2B (10% KV Cache):
```python
python infer_2B.py
```

## ⚡ Sample & Evaluations
### Sampling 5000 images from COCO-2017 captions with Infinity-8B.

```python
torchrun --nproc_per_node=$N_GPUS scripts/sample_8b.py
```

Sample images with ScaleKV compressed Infinity-8B (10% KV Cache):
```python
torchrun --nproc_per_node=$N_GPUS scripts/sample_kv_8b.py
```

After you sample all the images, you can calculate PSNR, LPIPS and FID with:
```python
python scripts/compute_metrics.py --input_root0 samples/gt_8b --input_root1 samples/scalekv_8b
```

### Sampling 5000 images from COCO captions with Infinity-2B.
```python
torchrun --nproc_per_node=$N_GPUS scripts/sample_2b.py
```

```python
torchrun --nproc_per_node=$N_GPUS scripts/sample_kv_2b.py
```

```python
python scripts/compute_metrics.py --input_root0 samples/gt_2b --input_root1 samples/scalekv_2b
```

## 📚 Key Results
<div align="center">
<img src="assets/picture.png" width="100%">
</div>

<div align="center">
<img src="assets/exp.png" width="100%">
</div>


<div align="center">
<img src="assets/mem.png" width="100%">
</div>

## Acknowlegdement
Thanks to [Infinity](https://github.com/FoundationVision/Infinity) for their wonderful work and codebase!


## Citation
If our research assists your work, please give us a star ⭐ or cite us using:
```
@article{li2025scalekv,
  title={Memory-Efficient Visual Autoregressive Modeling with Scale-Aware KV Cache Compression},
  author={Li, Kunjun and Chen, Zigeng and Yang, Cheng-Yen and Hwang, Jenq-Neng},
  journal={arXiv preprint arXiv:2505.19602},
  year={2025}
}
```
