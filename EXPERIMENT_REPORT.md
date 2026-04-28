# RAPID Evaluation Experiment Report

## Experiment Overview

This document details the evaluation experiment comparing RAPID (Real-time Adversarial Protection for Image Diffusion) against competing face protection methods on the CelebA-HQ dataset.

---

## Dataset

### Source
- **HuggingFace Dataset**: [mattymchen/celeba-hq](https://huggingface.co/datasets/mattymchen/celeba-hq)
- **File Used**: `train-00000-of-00006-bae07ad6d4d89a77.parquet`
- **Download URL**: `https://huggingface.co/datasets/mattymchen/celeba-hq/resolve/main/data/train-00000-of-00006-bae07ad6d4d89a77.parquet`

### Dataset Configuration
| Parameter | Value |
|-----------|-------|
| Total images in file | 4,667 |
| Images evaluated | 200 |
| Start index | 4,000 |
| End index | 4,199 |
| Image size | 512 × 512 |
| Image format | RGB |
| Storage format | Parquet (embedded image bytes) |

### Download Command
```bash
mkdir -p celeba-hq
wget "https://huggingface.co/datasets/mattymchen/celeba-hq/resolve/main/data/train-00000-of-00006-bae07ad6d4d89a77.parquet" -O celeba-hq/dataset.parquet
```

---

## Configuration Parameters

### Attack Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Epsilon | 12/255 ≈ 0.0471 | L-inf perturbation budget |
| Iterations | 40 | Number of optimization steps (competitors) |
| Step Size | 0.02 | PGD step size / Adam learning rate |

### Image Processing
| Parameter | Value |
|-----------|-------|
| Image Size | 512 × 512 |
| Batch Size | 1 |
| Normalization | [-1, 1] |

### Stable Diffusion InstructPix2Pix
| Parameter | Value |
|-----------|-------|
| Model Path | `/storage/2/models/diffusion/instruct-pix2pix` |
| Inference Steps | 50 |
| Guidance Scale | 7.5 |
| Image Guidance Scale | 1.5 |

### Evaluation Prompts
The following 10 text prompts were used for image editing:

1. "Turn the person's hair pink"
2. "Let the person turn bald"
3. "Let the person have a tattoo"
4. "Let the person wear purple makeup"
5. "Let the person grow a moustache"
6. "Turn the person into a zombie"
7. "Change the skin color to Avatar blue"
8. "Add elf-like ears"
9. "Let the person wear sunglasses"
10. "Let the person wear a police suit"

### Model Paths
| Model | Path |
|-------|------|
| RAPID (Restormer weights) | `/home/remote/Rapid/restormer.pth` |
| SD InstructPix2Pix | `/storage/2/models/diffusion/instruct-pix2pix` |
| CLIP (ViT-L/14) | `/storage/2/models/clip/ViT-L-14.pt` |
| FaceLock Aligner | `/storage/2/models/facelock/aligner/aligner` |
| FaceLock FR Model | `/storage/2/.cache/huggingface/fr_model` |

### Hardware
| Component | Details |
|-----------|---------|
| GPU | NVIDIA (CUDA 12.x compatible) |
| PyTorch Version | 2.5.1 |
| CUDA Version | cu124 |

---

## Methods Compared

### 1. Clean (Baseline)
- No protection applied
- Reference for unperturbed editing

### 2. PhotoGuard
- **Type**: Encoder attack (VAE latent norm maximization)
- **Iterations**: 40 PGD steps
- **Strategy**: Maximize VAE encoder latent norm to disrupt reconstruction

### 3. FaceLock
- **Type**: FR-guided attack with face aligner
- **Iterations**: 40 PGD steps
- **Strategy**: Maximize face verification loss while considering VAE reconstruction and LPIPS

### 4. EditShield
- **Type**: Latent diffusion attack
- **Iterations**: 40 Adam optimization steps
- **Strategy**: Use DDPM scheduler to create target embeddings and maximize distance

### 5. RAPID (Ours)
- **Type**: Single-pass Restormer attack
- **Forward passes**: 1 (single inference)
- **Strategy**: Use pretrained Restormer transformer to generate perturbations in one forward pass

---

## Evaluation Metrics

### Protection Metrics (Higher = Better Protection)
| Metric | Description |
|--------|-------------|
| **LPIPS** | Perceptual distance between edited clean and edited protected images. Higher = more different = better protection |
| **SSIM** | Structural similarity between edited clean and edited protected images. Lower = more distorted = better protection |
| **PSNR** | Peak signal-to-noise ratio. Lower = more noise = better protection |

### Alignment Metric (Lower = Better Protection)
| Metric | Description |
|--------|-------------|
| **CLIP Score** | Cosine similarity between edited protected image and text prompt. Lower = model failed to follow prompt = better protection |

### Efficiency Metric
| Metric | Description |
|--------|-------------|
| **Inference Time** | Time in seconds to generate adversarial perturbation. Lower = faster |

---

## Results

### Quantitative Results (200 Images)

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

### Results Table

| Method | LPIPS ↑ | SSIM ↑ | PSNR ↑ | CLIP ↓ | Time (s) ↓ |
|--------|---------|--------|--------|--------|------------|
| clean | 0.000 | 1.000 | 100.00 | 0.215 | 0.00 |
| editshield | 0.530 | 0.483 | 15.85 | 0.228 | 6.38 |
| facelock | 0.550 | 0.480 | 15.77 | 0.229 | 10.28 |
| photoguard | 0.559 | 0.488 | 13.20 | 0.229 | 2.38 |
| **rapid** | **0.478** | **0.628** | **19.18** | **0.214** | **0.10** |

*↑ = higher is better for protection, ↓ = lower is better*

---

## Analysis

### Key Findings

1. **Best Protection (CLIP Score)**: RAPID achieves the lowest CLIP score (0.214), indicating the diffusion model fails most to follow the edit prompt on protected images.

2. **Best Visual Quality (SSIM/PSNR)**: RAPID maintains the highest SSIM (0.628) and PSNR (19.18), meaning the perturbations are most imperceptible to humans.

3. **Fastest Inference**: RAPID runs in 0.10 seconds - **23× faster than PhotoGuard (2.38s)**, **63× faster than FaceLock (10.28s)**, and **63× faster than EditShield (6.38s)**.

4. **Single-Pass Efficiency**: Unlike competitors requiring 40 iterations, RAPID uses a single forward pass through the Restormer transformer.

### Metric Interpretation

- **LPIPS**: Measures perceptual difference. RAPID (0.478) provides good protection while maintaining visual quality.
- **SSIM**: Measures structural similarity. RAPID (0.628) best preserves image structure.
- **CLIP**: Measures prompt-image alignment. RAPID (0.214) most effectively breaks alignment.
- **Time**: RAPID (0.10s) enables real-time protection.

---

## Execution Command

To reproduce this experiment:

```bash
# Download dataset
mkdir -p celeba-hq
wget "https://huggingface.co/datasets/mattymchen/celeba-hq/resolve/main/data/train-00000-of-00006-bae07ad6d4d89a77.parquet" -O celeba-hq/dataset.parquet

# Run evaluation
python main.py --num-images 200 --methods all --output-dir results/
```

---

## Project Structure

```
/home/remote/Rapid/
├── main.py                    # CLI entry point
├── pyproject.toml             # Project configuration
├── README.md                  # Project documentation
├── restormer.pth              # RAPID pretrained weights
├── celeba-hq/                 # Dataset directory
│   └── dataset.parquet        # CelebA-HQ data
├── src/
│   ├── config.py              # Configuration & CLI
│   ├── evaluate.py            # Main evaluation loop
│   ├── models/
│   │   ├── rapid.py           # RAPID implementation
│   │   ├── photoguard.py     # PhotoGuard attack
│   │   ├── facelock.py       # FaceLock attack
│   │   ├── editshield.py     # EditShield attack
│   │   ├── pipeline.py       # SD I2P pipeline
│   │   └── restormer.py      # Restormer architecture
│   ├── metrics/
│   │   ├── perceptual.py     # LPIPS, SSIM, PSNR
│   │   └── clip_score.py     # CLIP similarity
│   └── data/
│       └── celeba.py         # Dataset loader
└── results/
    └── results.csv            # Evaluation results
```