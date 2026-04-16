#!/usr/bin/env python3
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture

"""Configuration for running pretrained GPT-2 XL with the MESO attention path.

This adds a parallel config for decoder-only causal LM inference.
"""

import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Hugging Face model identifier
MODEL_NAME = "openai-community/gpt2-xl"

# Prompt / generation parameters
PROMPT = "In-memory computing for transformers can"
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.8
TOP_K = 50
DO_SAMPLE = False
USE_CACHE = False  # keep False for the custom MESO attention path

# MESO / attention simulation parameters
MODE = 1  # 1 = MESO attention path, 0 = native PyTorch attention path
BIT_WIDTH = torch.tensor([8, 8])
CYCLE_RES = torch.tensor([4, 4])
VAR = 0.0

# Optional: evaluate on a small text dataset after download
RUN_DATASET_EVAL = False
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test[:32]"
BLOCK_SIZE = 128

# Hugging Face cache directory
CACHE_DIR = "./hf_cache"
