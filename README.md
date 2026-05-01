# RAPID: Restorative Amortized Protection for Image Diffusion

A comprehensive evaluation framework for testing adversarial protection methods against image editing attacks on text-to-image diffusion models.

## Overview

RAPID evaluates protection methods against prompt-based image editing attacks using Stable Diffusion InstructPix2Pix. It measures how well different defenses preserve images after adversarial attacks.

### Supported Methods

| Method | Description |
|--------|-------------|
| `clean` | No protection (baseline) |
| `rapid` | RAPID restoration-based defense |
| `photoguard` | CLIP-based purification defense |
| `facelock` | Face-specific protection |
| `editshield` | Encoder-based purification |

## Installation

```bash
# Install dependencies
uv sync

# Clone Restormer (required for RAPID)
git clone https://github.com/swz30/Restormer.git Restormer

# Download pretrained weights (see Model Setup below)
```

## Model Setup

Create `config.yaml` based on the provided template. Models are auto-downloaded from HuggingFace if not found locally:

```yaml
models:
  sd_i2p_path: "diffusers/instruct-pix2pix-78"  # Model ID or local path
  clip_path: "openai/clip-vit-large-patch14"
  rapid_weights_path: "restormer.pth"
```

### Manual Download

- **Restormer weights**: Place `restormer.pth` in project root
- **CelebA-HQ dataset**: Download from [Dropbox](https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=1) and extract to `celeba-hq/`

## Dataset Preparation

The evaluation uses CelebA-HQ images (indices 4000+ for test split). Prepare the dataset:

```bash
# Option 1: Manual download
wget "https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=1" -O celeba.zip
unzip celeba.zip -d celeba-hq

# Option 2: HuggingFace (auto-downloads on first use)
# Set repo_id in config.yaml
```

## Usage

### Basic Evaluation

```bash
# Evaluate all methods on 100 images
python main.py --num-images 100 --methods all --output-dir results/

# Evaluate specific method
python main.py --methods rapid --num-images 50
```

### Command-Line Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Config file path | `config.yaml` |
| `--num-images` | Number of images to evaluate | 100 |
| `--start-idx` | Starting image index | 4000 |
| `--methods` | Methods to evaluate | `all` |
| `--output-dir` | Results directory | `results/` |
| `--device` | Device (`cuda` or `cpu`) | `cuda` |
| `--save-images` | Save visualization samples | `false` |
| `--epsilon` | Attack epsilon (L-inf) | 0.047 |
| `--iterations` | Attack iterations | 40 |

### Configuration File

All options can be set in `config.yaml`:

```yaml
general:
  seed: 42
  device: "cuda"

dataset:
  path: "celeba-hq"
  num_eval_images: 100

methods:
  - "all"

attack:
  epsilon: 0.047
  num_iter: 40

diffusion:
  edit_steps: 50
  guidance_scale: 7.5
```

### Environment Variables

Override paths via environment variables:

```bash
export RAPID_SD_PATH="/path/to/stable-diffusion"
export RAPID_CLIP_PATH="/path/to/clip"
export RAPID_DATASET_PATH="/path/to/dataset"
export RAPID_OUTPUT_DIR="my-results"
```

## Output

Results are saved to the output directory with metrics:

```
results/
├── metrics.csv          # Numerical results
└── samples/             # Visualization (if --save-images)
```

### Metrics

- **LPIPS**: Perceptual similarity (lower = more similar to original)
- **SSIM**: Structural similarity (higher = better)
- **PSNR**: Peak signal-to-noise ratio (higher = better)
- **CLIP Score**: Semantic similarity to prompt

## Quick Start Example

```bash
# Full evaluation with all methods
python main.py \
  --num-images 200 \
  --methods all \
  --output-dir results/ \
  --save-images
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

## Project Structure

```
Rapid/
├── main.py              # Entry point
├── config.yaml          # Configuration template
├── src/
│   ├── config.py        # CLI and config parsing
│   ├── evaluate.py      # Evaluation loop
│   ├── data/            # Dataset handling
│   ├── models/          # Protection methods
│   ├── metrics/         # Evaluation metrics
│   └── utils/           # Utilities
└── Restormer/           # Cloned Restormer repo
```