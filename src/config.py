import os
import random
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

EPSILON = 12 / 255.0
NUM_ITER = 40
STEP_SIZE = 0.02

IMAGE_SIZE = 512
BATCH_SIZE = 1

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CELEBA_URL = "https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=1"

RAPID_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "restormer.pth")
RESTORMER_REPO_PATH = os.path.join(os.path.dirname(__file__), "..", "Restormer")

NUM_EVAL_IMAGES = 100
START_INDEX = 5000

PROMPTS = [
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

EDIT_STEPS = 50
GUIDANCE_SCALE = 7.5
IMAGE_GUIDANCE_SCALE = 1.5