import os
import random
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Any
import yaml

import numpy as np
import torch


def setup_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config_file(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_env_or_default(env_var: str, default: Any) -> Any:
    """Get value from environment variable or return default."""
    return os.environ.get(env_var, default)


class ModelPaths:
    """Model path configuration with auto-download support."""
    
    def __init__(self, config: dict = None):
        config = config or {}
        models = config.get("models", {})
        
        # Stable Diffusion InstructPix2Pix
        self.sd_i2p_path = get_env_or_default(
            "RAPID_SD_PATH", 
            models.get("sd_i2p_path", "/storage/2/models/diffusion/instruct-pix2pix")
        )
        self.sd_i2p_repo_id = models.get("sd_i2p_repo_id", "diffusers/instruct-pix2pix-78")
        
        # CLIP
        self.clip_path = get_env_or_default(
            "RAPID_CLIP_PATH",
            models.get("clip_path", "/storage/2/models/clip/ViT-L-14.pt")
        )
        self.clip_repo_id = models.get("clip_repo_id", "openai/clip-vit-large-patch14")
        
        # FaceLock
        self.facelock_aligner_path = get_env_or_default(
            "RAPID_FACELOCK_ALIGNER_PATH",
            models.get("facelock_aligner_path", "/storage/2/models/facelock/aligner/aligner")
        )
        self.facelock_fr_path = get_env_or_default(
            "RAPID_FACELOCK_FR_PATH",
            models.get("facelock_fr_path", "/storage/2/.cache/huggingface/fr_model")
        )
        self.facelock_aligner_repo_id = models.get("facelock_aligner_repo_id", "Klingener/FaceLock-Aligner")
        self.facelock_fr_repo_id = models.get("facelock_fr_repo_id", "Klingener/FaceLock-FR")
        
        # RAPID/Restormer
        self.rapid_weights_path = get_env_or_default(
            "RAPID_WEIGHTS_PATH",
            models.get("rapid_weights_path", "restormer.pth")
        )
        self.restormer_repo_path = models.get("restormer_repo_path", "Restormer")


class GeneralConfig:
    """General evaluation configuration."""
    
    def __init__(self, config: dict = None):
        config = config or {}
        general = config.get("general", {})
        
        self.seed = general.get("seed", 42)
        self.device = general.get("device", "cuda")
        self.verbose = general.get("verbose", True)
        self.save_images = general.get("save_images", False)


class DatasetConfig:
    """Dataset configuration."""
    
    def __init__(self, config: dict = None):
        config = config or {}
        dataset = config.get("dataset", {})
        
        self.path = get_env_or_default(
            "RAPID_DATASET_PATH",
            dataset.get("path", "celeba-hq")
        )
        self.file = dataset.get("file", "dataset.parquet")
        self.repo_id = dataset.get("repo_id", None)
        self.num_eval_images = dataset.get("num_eval_images", 100)
        self.start_index = dataset.get("start_index", 4000)
        self.url = dataset.get("url", "https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=1")


class OutputConfig:
    """Output configuration."""
    
    def __init__(self, config: dict = None):
        config = config or {}
        output = config.get("output", {})
        
        self.dir = get_env_or_default(
            "RAPID_OUTPUT_DIR",
            output.get("dir", "results")
        )


class MethodsConfig:
    """Methods configuration."""
    
    def __init__(self, config: dict = None):
        config = config or {}
        self.methods = config.get("methods", ["all"])


class PromptsConfig:
    """Prompts configuration."""
    
    def __init__(self, config: dict = None):
        config = config or {}
        self.prompts = config.get("prompts", DEFAULT_PROMPTS)


@dataclass
class AttackConfig:
    """Attack hyperparameters."""
    epsilon: float = 12 / 255.0
    num_iter: int = 40
    step_size: float = 0.02


@dataclass
class ImageConfig:
    """Image processing configuration."""
    size: int = 512
    batch_size: int = 1


@dataclass
class DataConfig:
    """Dataset configuration."""
    num_eval_images: int = 100
    start_index: int = 4000
    url: str = "https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=1"


@dataclass
class DiffusionConfig:
    """Stable Diffusion InstructPix2Pix configuration."""
    edit_steps: int = 50
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5


DEFAULT_PROMPTS: List[str] = [
    "Turn the person's hair pink",
    "Let the person turn bald",
    "Let the person have a tattoo",
    "Let the person wear purple makeup",
    "Let the person grow a moustache",
    "Turn the person into a zombie",
    "Change the skin color to Avatar blue",
    "Add elf-like ears",
    "Let the person wear sunglasses",
    "Let the person wear a police suit",
]


@dataclass
class EvalConfig:
    """Main evaluation configuration."""
    seed: int = 42
    device: torch.device = field(default_factory=get_device)

    attack: AttackConfig = field(default_factory=AttackConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)

    prompts: List[str] = field(default_factory=lambda: DEFAULT_PROMPTS)

    models: ModelPaths = field(default_factory=ModelPaths)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    methods: List[str] = field(default_factory=lambda: ["all"])

    save_images: bool = False
    verbose: bool = True
    
    config_file: str = "config.yaml"

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        setup_seed(self.seed)


def parse_args() -> EvalConfig:
    """Parse command line arguments into EvalConfig."""
    parser = argparse.ArgumentParser(
        description="RAPID Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config YAML file",
    )
    
    # General options
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    
    # Model paths (override config)
    parser.add_argument(
        "--sd-path", type=str, default=None,
        help="Stable Diffusion InstructPix2Pix model path (or repo ID for auto-download)",
    )
    parser.add_argument(
        "--clip-path", type=str, default=None,
        help="CLIP model path (or repo ID for auto-download)",
    )
    parser.add_argument(
        "--facelock-aligner-path", type=str, default=None,
        help="FaceLock aligner model path (or repo ID for auto-download)",
    )
    parser.add_argument(
        "--facelock-fr-path", type=str, default=None,
        help="FaceLock FR model path (or repo ID for auto-download)",
    )
    parser.add_argument(
        "--rapid-weights", type=str, default=None,
        help="RAPID/Restormer weights path",
    )
    
    # Dataset paths
    parser.add_argument(
        "--dataset-path", type=str, default=None,
        help="Dataset directory path",
    )
    
    # Evaluation options
    parser.add_argument(
        "--num-images", type=int, default=None,
        help="Number of test images to evaluate",
    )
    parser.add_argument(
        "--start-idx", type=int, default=None,
        help="Starting index in dataset (after training split)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to evaluate: photoguard, facelock, editshield, rapid, or all",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--save-images", action="store_true",
        help="Save visualization samples",
    )
    parser.add_argument(
        "--no-verbose", action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--epsilon", type=float, default=None,
        help="Perturbation budget (L-inf)",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Number of attack iterations",
    )

    args = parser.parse_args()

    yaml_config = load_config_file(args.config)
    
    general_cfg = yaml_config.get("general", {}) if yaml_config else {}
    dataset_cfg = yaml_config.get("dataset", {}) if yaml_config else {}
    methods_cfg = yaml_config.get("methods", ["all"]) if yaml_config else ["all"]
    prompts_cfg = yaml_config.get("prompts", DEFAULT_PROMPTS) if yaml_config else DEFAULT_PROMPTS
    
    model_paths = ModelPaths(yaml_config)
    dataset_config = DatasetConfig(yaml_config)
    output_config = OutputConfig(yaml_config)
    
    if args.sd_path is not None:
        model_paths.sd_i2p_path = args.sd_path
    if args.clip_path is not None:
        model_paths.clip_path = args.clip_path
    if args.facelock_aligner_path is not None:
        model_paths.facelock_aligner_path = args.facelock_aligner_path
    if args.facelock_fr_path is not None:
        model_paths.facelock_fr_path = args.facelock_fr_path
    if args.rapid_weights is not None:
        model_paths.rapid_weights_path = args.rapid_weights
    if args.dataset_path is not None:
        dataset_config.path = args.dataset_path
    if args.output_dir is not None:
        output_config.dir = args.output_dir
    
    attack_cfg = yaml_config.get("attack", {}) if yaml_config else {}
    attack = AttackConfig(
        epsilon=args.epsilon if args.epsilon is not None else attack_cfg.get("epsilon", 12 / 255.0),
        num_iter=args.iterations if args.iterations is not None else attack_cfg.get("num_iter", 40),
        step_size=attack_cfg.get("step_size", 0.02),
    )
    
    img_cfg = yaml_config.get("image", {}) if yaml_config else {}
    image = ImageConfig(
        size=img_cfg.get("size", 512),
        batch_size=img_cfg.get("batch_size", 1),
    )
    
    diff_cfg = yaml_config.get("diffusion", {}) if yaml_config else {}
    diffusion = DiffusionConfig(
        edit_steps=diff_cfg.get("edit_steps", 50),
        guidance_scale=diff_cfg.get("guidance_scale", 7.5),
        image_guidance_scale=diff_cfg.get("image_guidance_scale", 1.5),
    )

    config = EvalConfig(
        seed=args.seed if args.seed is not None else general_cfg.get("seed", 42),
        device=torch.device(args.device),
        data=DataConfig(
            num_eval_images=args.num_images if args.num_images is not None else dataset_cfg.get("num_eval_images", 100),
            start_index=args.start_idx if args.start_idx is not None else dataset_cfg.get("start_index", 4000),
        ),
        attack=attack,
        image=image,
        diffusion=diffusion,
        models=model_paths,
        dataset=dataset_config,
        output=output_config,
        methods=args.methods if args.methods is not None else methods_cfg,
        prompts=prompts_cfg,
        save_images=args.save_images,
        verbose=not args.no_verbose,
        config_file=args.config,
    )

    return config


# Backwards compatibility - module-level constants
DEVICE = get_device()
SEED = 42
EPSILON = 12 / 255.0
NUM_ITER = 40
STEP_SIZE = 0.02
IMAGE_SIZE = 512
BATCH_SIZE = 1
NUM_EVAL_IMAGES = 100
START_INDEX = 4000
PROMPTS = DEFAULT_PROMPTS
EDIT_STEPS = 50
GUIDANCE_SCALE = 7.5
IMAGE_GUIDANCE_SCALE = 1.5

setup_seed(SEED)