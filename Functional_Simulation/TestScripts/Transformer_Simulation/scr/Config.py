
#!/usr/bin/env python3
"""
###############################################################################
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture
# Module:        Config.py
# Description:   Centralized configuration file for the Small Language Model (SLM) 
#		 architecture for German-English translation task.
#
# Synopsis:      This module serves as the authoritative source providing all 
#                necessary model, optimization, and hardware simulation parameters 
#                for initializing and executing the SLM architecture experiments.
#
# Created:       2025-11-11
# Last Modified: 2026-04-10
###############################################################################

Usage and Interface:

    This module serves as the **central configuration source** for the main transformer execution script.

Execution Parameters (Configuration Constants):

    Model Configuration:
        batch_size (int):   Number of samples processed per iteration (Default: 64).
        max_len (int):      Maximum sequence length for inputs (Default: 512).
        d_model (int):      Dimensionality of the model embeddings (Default: 512).
        n_heads (int):      Number of attention heads (Default: 8).
        n_layers (int):     Number of transformer layers (Default: 6).
        ffn_hidden (int):   Size of the feed-forward network's inner layer (Default: 2048 = 4*d_model).

    Optimization Configuration:
        init_lr (float):    Initial learning rate (Default: 1e-4).
        epoch (int):        Maximum training epochs (Default: 100).
        weight_decay (float): L2 regularization factor (Default: 0.0001).
	label_smoothing (float) : Regularization factor to prevent overconfidence (Default: 0.1)
    
    Hardware/IMC Configuration:
        bit_width (int):    Mantissa bit-width configured for the MESO IMC architecture (Default: 8, valid: [4,32]).
	cycle_res (int):    Cycle resolution used during time-multiplexing (Default: 4, valid: [1,7])
	var:		    device variability used to modulate the spin current of the MESO device (Default:0, valid :[0,1]).
        mode (int):         Execution mode selector. 1: In-Memory Computing (IMC) simulation; 0: CMOS GPU reference (Default: 0).

Note: This script is designed to be **sourced** by a main execution module (TransformerTop.py) to load 
      these predefined configuration constants. 
"""


import torch

# This code was tested extensively on mps.
#uncomment the "cuda" line if you wish to experiment. 
device = (
	"mps" if torch.backends.mps.is_available()
	#else "cuda" if torch.cuda.is_available()
	else "cpu"
	)
DEVICE = device

# model parameter setting
batch_size = 64
max_len = 128
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 4*d_model
drop_prob = 0.1

# optimizer parameter setting
init_lr = 1e-4
factor = 0.1
adam_eps = 1e-9
patience = 3
warmup = 4000
epoch = 50
clip = 1.0
weight_decay = 0.0001
label_smoothing=0.1
inf = float('inf')
specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
resume_training=True
strict_config=True

mode=1
bit_width=torch.tensor([8, 8])
cycle_res=torch.tensor([4, 4])
var=0.0

torch.set_default_dtype(torch.float32)
