

#!/usr/bin/env python3
"""
###############################################################################
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture
# Module:        MESOTRDPTop.py
# Description:   Top-level module for MESO In-Memory Computing (IMC) operations.
#
# Synopsis:      This module handles pre-processing for IMC architectures. It 
#                accepts quantized input tensors, manages data partitioning, 
#                and applies padding to ensure proper dimensionality for 
#                crossbar array processing.
#		 This module is designed for multi-head, multi-batch configurations.
#
# Created:       2025-11-11
# Last Modified: 2026-03-27
###############################################################################

Usage and Interface:

    The primary entry point is the 'Top' function:
    
    Out = Top(Weights, Weight_Shifts, Weight_Sign, Inputs, Input_Shift, Input_Sign, Weight_Matrix, Bit_Width)

Parameters:

    Weights:         Quantized weight tensor (e.g., Fully Connected Cell (FCC) weights, or query/key/softmax components).
    Inputs:          Quantized input tensor (e.g., FCC inputs, or key/query/value components).
    XXX_Shifts:      Exponent of the largest element relative to the defined bit-width (calculated via ShiftsConv_batch utility).
    XXX_Sign:        Sign indicator of the minimum element within the weight matrix.
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


from DotProductMESO.AttProcessTimePlex import VectorizedProg_torch


def Alter(B, InRes, cycles, BW=8):
	mult=2**torch.arange(InRes, dtype=B.dtype, device=B.device)
	pad_size=cycles*InRes-BW
	B_padded = F.pad(B, (0, pad_size))
	B_out=B_padded.view(*B.shape[:-1], cycles, InRes)
	return (B_out * mult).sum(dim=-1)

def Top(Weight, ShiftW, SignW, B, ShiftB, SignB, Ax, BW=8, InRes=1, Var=0.0):
    
	batchW, headsW, rowsW, colsW, bitsW = Weight.shape
	assert bitsW == BW, "Issues with the Weight tensor in Top module"

	batchB, headsB, rowsB, colsB, bitsB = B.shape
	assert bitsB == BW, "Issues with the Values tensor in Top module"

	assert batchW==batchB, "minibatch sizes for the inputs are different"
	assert headsW==headsB, "heads mismatch between inputs - Top"

	cols = 16
	rows = 128

	I=10E-6 #Output current through MESO device

	Weight=Weight.permute(0,1,4,2,3)

	Weight=torch.where(Weight==0,-I,I)

	cycles = math.ceil(BW/InRes)
	BNew =Alter(B, InRes, cycles, BW)

	B=BNew.reshape(batchB, headsB, rowsB, -1)

	D1 = math.ceil(rowsW / rows)
	D2 = math.ceil(colsW / cols)
    
	padX = int(rows * D1) - rowsW
	padY = int(cols * D2) - colsW
    
	padding = (0, padY, 0, padX, 0, 0, 0, 0, 0, 0) 
    
	W_padded = F.pad(Weight, padding, mode='constant', value=-I) 
    
	WeightN = W_padded.reshape(batchW, headsW, BW, D1, rows, D2, cols)
    
	WeightN = WeightN.permute(0, 1, 2, 3, 5, 4, 6)

	WeigntN=F.pad(WeightN, (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), mode='constant', value=-I)

	low, high = 1-Var, 1+Var
	scale = (high - low) * torch.rand_like(WeightN) + low
	WeightN = WeightN * scale


	num_blocks_B = math.ceil(rowsB / cols)
	pad_rows_B = int(num_blocks_B * cols) - rowsB
    
	padding_B = (0, 0, 0, pad_rows_B, 0, 0, 0, 0)

	B_padded = F.pad(B, padding_B, mode='constant', value=0.0)
    
	BN = B_padded.reshape(batchB, headsB, num_blocks_B, cols, colsB*cycles)


	assert D2==num_blocks_B

	Output = VectorizedProg_torch(WeightN, ShiftW, SignW, BN, ShiftB, SignB, Ax, cycles, InRes, BW)

	return Output



