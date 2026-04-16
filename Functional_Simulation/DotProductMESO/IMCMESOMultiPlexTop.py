
#!/usr/bin/env python3
"""
###############################################################################
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture
# Module:        MESODSTop.py
# Description:   Top-level module for MESO In-Memory Computing (IMC) operations.
#
# Synopsis:      This module handles pre-processing for IMC architectures. It 
#                accepts quantized input tensors, manages data partitioning, 
#                and applies padding to ensure proper dimensionality for 
#                crossbar array processing.
#
# Limitation:    This module is designed exclusively for single-head, multi-batch configurations.
# Reference:     For multi-head execution capability, please utilize the dedicated 
#                module: MESOTRDPTopTimePlex.py.
#
# Created:       2025-11-11
# Last Modified: 2026-03-10
###############################################################################

Usage and Interface:

    The primary entry point is the 'Top' function:
    
    Out = Top(Weights, Weight_Shifts, Weight_Sign, Inputs, Input_Shift, Input_Sign, Weight_Matrix, Bit_Width)

Parameters:

    Weights:         Quantized weight tensor (e.g., Fully Connected Cell (FCC) weights, or query/key/softmax components).
    Inputs:          Quantized input tensor (e.g., FCC inputs, or key/query/value components).
    XXX_Shifts:      Exponent of the largest element relative to the defined bit-width (calculated via ShiftsConv_batch utility).
    XXX_Sign:        Sign indicator of the minimum element within the input matrix.
    Weight_Matrix:   Weight matrix with floating-point elements.
    Bit_Width:       The quantization bit-width used for the tensors.

Refer to TRDP_MESO.py for comprehensive implementation examples and usage context.
"""


import math
import sys
sys.path.append('../')
import numpy as np
import cmath
import cProfile
import torch
import torch.nn.functional as F

from DotProductMESO.IMCMESOMultiPlex_Process import VectorizedProg_torch

def Alter(B, InRes, cycles, BW=8):
	mult=2**torch.arange(InRes, dtype=B.dtype, device=B.device)
	pad_size=cycles*InRes-BW
	B_padded = F.pad(B, (0, pad_size))
	B_out=B_padded.view(*B.shape[:-1], cycles, InRes)
	return (B_out * mult).sum(dim=-1)


def Top(Weight, ShiftW, SignW, B, ShiftB, SignB, Ax, BW=8, Var=0.0, InRes=4):

	# --------------------------------------------------------------------------------------------------------------------------
	# 1) Basic shape checks
	# --------------------------------------------------------------------------------------------------------------------------
	batchW,rowsW, colsW, bitsW = Weight.shape
	assert bitsW == BW, "Issues with the Weight tensor in Top module"

	batchB,rowsB, colsB, bitsB = B.shape
	assert bitsB == BW, "Issues with the Values tensor in Top module"

	assert batchW==batchB, "minibatch sizes for the inputs are different"


	# --------------------------------------------------------------------------------------------------------------------------
	# 2) Crossbar tiling parameters
	# --------------------------------------------------------------------------------------------------------------------------
	# These are the fixed crossbar dimensions used by the underlying MESO/IMC core.
	#   - `rows`: number of wordlines / rows in a crossbar
	#   - `cols`: number of bitlines / columns in a crossbar
	# The 256×64 array/tile size was selected to satisfy IR-drop constraints in a 45 nm technology node.
	# Each output element is produced by aggregating contributions from 64 MESO devices.

	cols = 16
	rows = 128

	# Current magnitude used when mapping (0/1) weight bits to +/- current.
	#A '0' maps to -I and a '1' maps to +I. 
	I=float(10E-6)

	# --------------------------------------------------------------------------------------------------------------------------
	# 3) Reorder and map weights to +/-I
	# --------------------------------------------------------------------------------------------------------------------------
	# Move bit-slices forward so Weight becomes: (batch, BW, rowsW, colsW)
	# The +/- I values are subsequently divided onto the finite sized crossbars. 

	Weight=Weight.permute(0,3,1,2)

	# Map stored bits/levels to a sign/current model expected by the MESO core.
	# Here: 0 -> -I, non-zero -> +I.
	Weight=torch.where(Weight==0,-I,I)


	# --------------------------------------------------------------------------------------------------------------------------
	# 4) Flatten B's (colsB, BW) into a single feature dimension
	# --------------------------------------------------------------------------------------------------------------------------
	# (batch, rowsB, colsB, BW) -> (batch, rowsB, colsB, cycles) -> (batch, rowsB, colsB*cycles)
	# Each row now carries all bit-slices of the original features.
	#Reorder the input activations into time-multiplexed slices. 

	cycles = math.ceil(BW/InRes)
	BNew =Alter(B, InRes, cycles, BW)	
	B = BNew.reshape(batchB,rowsB,-1)

	# --------------------------------------------------------------------------------------------------------------------------
	# 5) Tile Weight into (D1 x D2) crossbar blocks and pad to crossbar multiples - emulating crossbar programming
	# --------------------------------------------------------------------------------------------------------------------------
	# Number of crossbars needed along Weight's two spatial dimensions.

	D1 = math.ceil(rowsW / rows)
	D2 = math.ceil(colsW / cols)
    
	# Amount of zero/constant padding needed to reach exact crossbar multiples.
	padX = int(rows * D1) - rowsW #pad along Weight row dimension
	padY = int(cols * D2) - colsW #pad along Weight col dimension
    
	# F.pad uses the reverse order of dimensions; for 4D tensors we supply 8
	# values (pad_left, pad_right) for each of the last 4 dimensions.
	# We only pad (rowsW, colsW) and keep (batch, BW) unchanged
	# The entire dot-product operation gets performed over (batchW x BW x D1 x D2) crossbars. 
	# The crossbars are of size rows x cols
	padding = (0, padY, 0, padX, 0, 0, 0, 0) 
    
	# Pad with -I (the "zero" weight current in this representation) to make the 
	#new dimension of W_padded a multiple of the crossbar size. 
	W_padded = F.pad(Weight, padding, mode='constant', value=-I) 
    
	#Reshape into explicit tiles:
	# (batch, BW, rowsW_pad, colsW_pad) -> (batch, BW, D1, rows, D2, cols)
	WeightN = W_padded.reshape(batchW, BW, D1, rows, D2, cols)
    
	#Permute to group crossbars as (D1, D2) with the per-crossbar (rows, cols) last: (batch, BW, D1, D2, rows, cols)
	WeightN = WeightN.permute(0, 1, 2, 4, 3, 5)

	WeightN = F.pad(WeightN, (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0), mode='constant', value=-I)

	assert WeightN.shape[-2]==rows+1

	low, high = 1-Var, 1+Var
	scale = (high - low) * torch.rand_like(WeightN) + low
	print(f"Expected Variability in MESO current is between : 0% and {Var*100}%")
	print(f"Simulated Variability in MESO current is between : 0% and {max(1-torch.min(scale).item(), torch.max(scale).item()-1)*100}%")
	WeightN = WeightN * scale


	# --------------------------------------------------------------------------------------------------------------------------
	# 6) Tile B in blocks of 16 rows to match Weight column tiles
	# --------------------------------------------------------------------------------------------------------------------------
	# B is blocked by `cols` (16) because it is given as gate voltage to the transistors along each column of the crossbar.
	num_blocks_B = math.ceil(rowsB / cols)
	pad_rows_B = int(num_blocks_B * cols) - rowsB
    
	# Pad only the row dimension of B (second-to-last dim for a 3D tensor).
	padding_B = (0, 0, 0, pad_rows_B, 0, 0)
	B_padded = F.pad(B, padding_B, mode='constant', value=0.0)

	#Block rows into (num_blocks_B, cols) chunks: (batch, rowsB_pad, colsB*cycles) -> (batch, num_blocks_B, cols, colsB*cycles)
	#This is done to increase the ease of the CIM-based dot-product execution. 
	BN = B_padded.reshape(batchB, num_blocks_B, cols, colsB*cycles)

	# Assertion based Sanity check: Weight column blocks must match B row blocks.
	assert D2==num_blocks_B

	# --------------------------------------------------------------------------------------------------------------------------
	# 7) Call the MESO vectorized core that performs the in-memory processing, ADC conversion and post-processing.
	# --------------------------------------------------------------------------------------------------------------------------
	Output = VectorizedProg_torch(WeightN, ShiftW, SignW, BN, ShiftB, SignB, Ax, InRes, cycles, BW)

	return Output



