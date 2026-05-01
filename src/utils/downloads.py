"""Model downloading utilities for FaceLock components."""

import os
import shutil
from typing import Optional, Any

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel


def download_model_files(repo_id: str, save_path: str, token: Optional[str] = None) -> None:
    """Download all model files from a HuggingFace repository.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., 'username/model')
        save_path: Local directory to save files
        token: HuggingFace access token
    """
    os.makedirs(save_path, exist_ok=True)
    
    files_path = os.path.join(save_path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=token, local_dir=save_path, local_dir_use_symlinks=False)
    
    with open(files_path, 'r') as f:
        files = f.read().split('\n')
    
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(save_path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=token, local_dir=save_path, local_dir_use_symlinks=False)


def load_model_from_local_path(model_path: str, token: Optional[str] = None):
    """Load a model from local path, handling path changes properly.
    
    Args:
        model_path: Local path to model
        token: HuggingFace access token
        
    Returns:
        Loaded model
    """
    import sys
    cwd = os.getcwd()
    os.chdir(model_path)
    sys.path.insert(0, model_path)
    
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, token=token)
    
    os.chdir(cwd)
    sys.path.pop(0)
    return model


def download_face_models(
    aligner_path: str = "/storage/2/models/facelock/aligner/aligner",
    fr_model_path: str = "/storage/2/.cache/huggingface/fr_model",
    aligner_repo_id: str = "Klingener/FaceLock-Aligner",
    fr_repo_id: str = "Klingener/FaceLock-FR",
) -> dict:
    """Load FaceLock models from local paths or auto-download from HuggingFace.
    
    Args:
        aligner_path: Local path to aligner model
        fr_model_path: Local path to FR model
        aligner_repo_id: HuggingFace repo ID for aligner (for auto-download)
        fr_repo_id: HuggingFace repo ID for FR model (for auto-download)
        
    Returns:
        Dict with 'aligner' and 'fr_model' keys
    """
    # Load aligner model - try local first, then auto-download
    if os.path.exists(aligner_path):
        aligner = load_model_from_local_path(aligner_path)
    else:
        # Auto-download from HuggingFace
        aligner = AutoModel.from_pretrained(aligner_repo_id, trust_remote_code=True)
    
    # Load FR model - try local first, then auto-download
    if os.path.exists(fr_model_path):
        fr_model = load_model_from_local_path(fr_model_path)
    else:
        # Auto-download from HuggingFace
        fr_model = AutoModel.from_pretrained(fr_repo_id, trust_remote_code=True)
    
    return {
        'aligner': aligner.model,  # Use .model, not .aligner
        'fr_model': fr_model,
    }


def download_vae_from_pipeline(pipeline) -> torch.nn.Module:
    """Extract VAE from a diffusers pipeline.
    
    Args:
        pipeline: Diffusers pipeline with vae attribute
        
    Returns:
        VAE model
    """
    return pipeline.vae


def download_scheduler_from_pipeline(pipeline) -> Any:
    """Extract scheduler from a diffusers pipeline.
    
    Args:
        pipeline: Diffusers pipeline with scheduler attribute
        
    Returns:
        Scheduler
    """
    return pipeline.scheduler