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
    save_dir: str = "models/face",
    force_download: bool = False,
    token: Optional[str] = None,
) -> dict:
    """Download FaceLock required models (aligner and FR model).
    
    Args:
        save_dir: Directory to save models
        force_download: Force re-download even if exists
        token: HuggingFace access token
        
    Returns:
        Dict with 'aligner' and 'fr_model' keys
    """
    # MobyFace repository (FaceLock models)
    repo_id = "MobyFace/MobyFace"
    
    if force_download and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    if not os.path.exists(os.path.join(save_dir, 'files.txt')):
        download_model_files(repo_id, save_dir, token)
    
    model = load_model_from_local_path(save_dir, token)
    
    # Model components are accessed via .aligner and .net attributes
    return {
        'aligner': model.aligner,
        'fr_model': model,
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