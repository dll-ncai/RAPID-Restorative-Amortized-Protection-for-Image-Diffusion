# RAPID Evaluation Pipeline

## Installation

```bash
# Clone the repository
git clone https://github.com/dll-ncai/RAPID-Restorative-Amortized-Protection-for-Image-Diffusion.git
cd RAPID-Restorative-Amortized-Protection-for-Image-Diffusion

# Install dependencies
uv sync

# Copy example config
cp config.example.yaml config.yaml
```

## Model Setup

### 1. Restormer Weights (Required for RAPID)

Download pretrained weights from the Restormer release:

```bash
# Download from https://github.com/swz30/Restormer/releases
# Place restormer.pth in the project root
```

### 2. Dataset

Download CelebA-HQ dataset:

```bash
# Option 1: Download from Dropbox
wget "https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=1" -O celeba.zip
unzip celeba.zip -d celeba-hq
```

The evaluation uses images starting from index 4000 (test split).

## Running Evaluation

### Quick Start

```bash
# Evaluate all methods on 100 images
python main.py --num-images 100 --methods all

# Evaluate specific method
python main.py --methods rapid --num-images 50
```

### Configuration

All options can be configured in `config.yaml`:

```yaml
general:
  seed: 42
  device: "cuda"
  verbose: true

dataset:
  path: "celeba-hq"
  num_eval_images: 100
  start_index: 4000

methods:
  - "all"

attack:
  epsilon: 0.047
  num_iter: 40

diffusion:
  edit_steps: 50
  guidance_scale: 7.5
  image_guidance_scale: 1.5
```

### CLI Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Config file path | `config.yaml` |
| `--num-images` | Number of images | 100 |
| `--start-idx` | Starting image index | 4000 |
| `--methods` | Methods to evaluate | `all` |
| `--output-dir` | Results directory | `results/` |
| `--device` | Device (`cuda`/`cpu`) | `cuda` |
| `--epsilon` | Attack epsilon | 0.047 |
| `--iterations` | Attack iterations | 40 |

### Environment Variables

Override paths via environment variables:

```bash
export RAPID_SD_PATH="/path/to/model"
export RAPID_CLIP_PATH="/path/to/clip"
export RAPID_DATASET_PATH="/path/to/dataset"
export RAPID_OUTPUT_DIR="results"
```

## Output

Results are saved to `results/results.csv` with metrics:

- **LPIPS**: Perceptual similarity (lower = more similar to original)
- **SSIM**: Structural similarity (higher = better)
- **PSNR**: Peak signal-to-noise ratio (higher = better)
- **CLIP Score**: Semantic similarity to prompt (lower = better protection)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 16GB+ RAM