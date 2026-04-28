# RAPID Evaluation Pipeline

Real-time Adversarial Protection for Image Diffusion (RAPID) evaluation framework.

## Setup

```bash
# Install dependencies
uv sync

# Clone Restormer repository
git clone https://github.com/swz30/Restormer.git Restormer

# Download pretrained weights
# (Add instructions here)
```

## Data Preparation

Download the dataset from HuggingFace:

```bash
wget "https://huggingface.co/datasets/mattymchen/celeba-hq/resolve/main/data/train-00000-of-00006-bae07ad6d4d89a77.parquet" -O celeba-hq/dataset.parquet
```

## Usage

```bash
python main.py --num-images 200 --methods all --output-dir results/
```

This command evaluates all attack methods (clean, photoguard, facelock, editshield, rapid) on 200 images starting from index 4000 in the CelebA-HQ dataset.

## Results

```
============================================================
EVALUATION SUMMARY
============================================================
             LPIPS    SSIM      PSNR  CLIP_Score  inference_time
method                                                                  
clean       0.0000  1.0000  100.0000      0.2151          0.0000
editshield  0.5299  0.4825   15.8522      0.2277          6.3787
facelock    0.5496  0.4799   15.7744      0.2292         10.2833
photoguard  0.5589  0.4882   13.1951      0.2289          2.3800
rapid       0.4777  0.6279   19.1762      0.2142          0.1021
```