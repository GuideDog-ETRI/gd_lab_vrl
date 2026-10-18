"""pytest setup for gast_nav CPU fixtures: repo src on sys.path, single CPU thread, no simulator import."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
torch.set_num_threads(1)
