
#!/usr/bin/env python3
"""
###############################################################################
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture
# Module:        QuantizationPrep.py
# Description:   Preprocessing module for signed quantization and tensor convergence in IMC architectures.
#
# Synopsis:      This module provides the necessary functions to handle signed 
#                (positive/negative) floating-point inputs for In-Memory Computing (IMC) 
#                simulation. It performs **sign-aware quantization** (Shifts_torch3) 
#                and generates the minimum value metadata. Crucially, it includes 
#                functions (converge/converge4) to **pad** and **concatenate** the 
#                quantized tensors with the minimum value metadata, which is often 
#                required for two's complement or sign-reconstruction in specialized 
#                hardware array layouts.
#
# Created:       2025-11-11
# Last Modified: 2025-12-08
###############################################################################

Functions Provided:

Shifts_torch3(Input, Bit_Width):
    Performs sign-aware quantization on input tensor A. It calculates the minimum value 
    offset, subtracts the offset to ensure all values are non-negative, 
    quantizes the resulting positive tensor, and quantizes the absolute minimum 
    value for later sign reconstruction.
    
    Parameters:
        Input (torch.Tensor): The input floating-point tensor (weights or activations) to be quantized.
        Bit_Width (int):      The target output **bit-width** (BW) for quantization (Default: 16).
    
    Returns: Quantized positive tensor (A1), Quantized minimum value (minAbin), 
             Shift (ShiftA), and overall Sign (SignA).

Converged=converge(Weights_Bin, Weights_Min, Inputs_Bin, Inputs_min):
    Handles 3D tensor convergence (Batch x Rows x Cols x Bit_Width) for single-head/multi-batch 
    operations. It replicates and concatenates the quantized minimum values 
    along the appropriate dimensions.
    
    Parameters:
        Weights_Bin (torch.Tensor): The quantized weight tensor (e.g., Q, K, V components).
        Weights_Min (torch.Tensor): The quantized minimum value tensor derived from the weights (W).
        Inputs_Bin (torch.Tensor):  The quantized input/activation tensor (e.g., K, V, or input data).
        Inputs_Min (torch.Tensor):  The quantized minimum value tensor derived from the input (BT).

Converged=converge4(Weights_Bin, Weights_Min, Inputs_Bin, Inputs_min):
    Handles 5D tensor convergence (Batch x Heads x Rows x Cols x BW) for multi-head/multi-batch 
    operations. It replicates and concatenates the quantized minimum values along the row/column dimensions.
    
    Parameters:
        Weights_Bin (torch.Tensor): The 5D quantized weight tensor.
        Weights_Min (torch.Tensor): The quantized minimum value tensor derived from the weights (W).
        Inputs_Bin (torch.Tensor): The 5D quantized input/activation tensor.
        Inputs_min (torch.Tensor): The quantized minimum value tensor derived from the input (BT).
"""


import math
import sys
sys.path.append('../')
import torch

# Fixed-point / "fraction" conversion helpers used by the quantization flow.
# These are assumed to implement:
#   - ShiftCalc: choose (DecVal, Shift) to represent the dynamic range with n bits
#   - vectorized_fractions: fast tensor quantization using the chosen scaling
#   - fractions: scalar (Python) version, used here for the minimum-value term

from GeneralScripts.FracConv_batch import ShiftCalc
from GeneralScripts.FracConv_batch import fractions
from GeneralScripts.FracConv_batch import vectorized_fractions
from GeneralScripts.FracConv_batch import ShiftCalc_torch_vectorized


def Shifts_torch3(A,n=16):

	"""Sign-aware quantization by shifting the tensor into the non-negative range.

	This helper is used when the downstream IMC/CIM datapath expects **non-negative**
	inputs (e.g., conductances / device states), while the original tensor may have
	negative values.

	Steps:
	  1) Compute a scalar offset minA_val = min(min(A), 0). This ensures the offset
	     is never positive, and is exactly 0 if A is already non-negative.
	  2) Shift: AN = A - minA_val, so AN >= 0 elementwise.
	  3) Determine a scaling (DecValA, ShiftA) based on the max magnitude between
	     the shifted tensor's max and the magnitude of the minimum offset.
	  4) Quantize the shifted tensor AN -> A1 (binary representation).
	  5) Quantize the magnitude of the minimum value |minA_val| -> minAbin.

	Args:
	    A: Input tensor (weights/activations) of arbitrary shape.
	    n: Target bit-width (default 16).

	Returns:
	    A1:       Shifted, Quantized and aligned representation of the shifted tensor AN (non-negative).
	    minAbin:  Quantized and aligned representation of |minA_val| (the shift compensation term).
	    ShiftA:   Scalar shift/exponent metadata chosen by ShiftCalc.
	    SignA:    Sign metadata for the offset term: -1 if minA_val < 0, else 0.
	              (When minA_val == 0, SignA is set to 0.)
	"""


	# Preserve the original device placement for any tensors created here.
	device=A.device

	# Scalar minimum of A, clamped to be <= 0 (so we only ever shift "up", to maintain the accuracy).
	# minA_val is a Python float (from .item()) used as a scalar offset.
	#Todo: Shift minA_val calculation to a per-batch basis to prevent global statistics from washing out local variance.
	minA_val = min(torch.min(A).item(), 0)

	# Shift the tensor so all elements become non-negative (AN >= 0).
	AN = A - minA_val
	
	#Sign metadata for the offset: typically -1 when minA_val is negative, else 0.
	#This is used later to reconstruct the signed contribution.
	SignA=minA_val/abs(minA_val) if minA_val!=0 else 0
	
	# Magnitude of the offset term (always non-negative).
	minA_val_abs = abs(minA_val)

	# Dynamic-range estimate for selecting scaling parameters.
	# max_A_val is computed on the shifted tensor AN (>= 0).
	max_A_val = torch.max(AN).item()

	# Choose scaling/shift such that both the shifted tensor and the offset term
	# are representable with n bits.
	DecValA, ShiftA = ShiftCalc(max(max_A_val, minA_val_abs), n)

	# Quantize and align the shifted tensor using the chosen scaling.
	A1=vectorized_fractions(AN,ShiftA,DecValA,n)

	# Quantize and align the scalar offset magnitude into the same representation.
	minAbin = torch.tensor(fractions(minA_val_abs, ShiftA, DecValA, n), dtype=A.dtype, device=device)

	# Keep outputs on the same device as the input.
	return A1.to(device),minAbin,ShiftA,SignA


def converge(Wbin, minW, BTBin, minBT):

	"""Converge/pad 4D tensors by appending their quantized minimum-value terms.

	This function is intended for batched matmul-style flows where:
	  - Wbin represents a weight-like operand with shape [B, Rw, Cw, BW]
	  - BTBin represents a (possibly transposed) activation-like operand with
	    shape [B, Rbt, Cbt, BW]

	The minimum-value metadata (minW, minBT) is appended as:
	  - an extra "row" plane for Wbin (concatenate along dim=1)
	  - an extra "column" plane for BTBin (concatenate along dim=2)

	This produces a structured layout where the offset/shift compensation can be
	accounted for by downstream IMC/CIM kernels.

	Args:
	    Wbin:   Quantized weights (4D): [batch, rowsW, colsW, BW]
	    minW:   Quantized |min(W)| term (typically shape [BW] )
	    BTBin:  Quantized inputs (4D):  [batch, rowsBT, colsBT, BW]
	    minBT:  Quantized |min(BT)| term (typically shape [BW] )

	Returns:
	    Wcat:   Wbin with an additional row containing minW replicated across cols.
	    BTcat:  BTBin with an additional column containing minBT replicated across rows.
	"""

	# Basic matmul alignment sanity checks:
	#   - inner dimensions must match (W columns == BT rows)
	#   - batch dimension must match
	batchW, rowsW, colsW, BW=Wbin.shape
	batchBT, rowsBT, colsBT, BW=BTBin.shape

	assert colsW==rowsBT
	assert batchBT==batchW

	# Expand minW so it can be concatenated as an extra row:
	#   minW: [BW]                (typical)
	#   -> repeat(colsW, 1): [colsW, BW]
	#   -> add batch:        [B, colsW, BW]
	#   -> add row axis:     [B, 1, colsW, BW]
	minW=minW.repeat(colsW,1)
	minW=minW.unsqueeze(0).repeat(batchW, 1, 1)
	minW = minW.unsqueeze(1)

	# Expand minBT so it can be concatenated as an extra column:
	#   minBT: [BW]
	#   -> repeat(rowsBT, 1): [rowsBT, BW]
	#   -> add batch:         [B, rowsBT, BW]
	#   -> add col axis:      [B, rowsBT, 1, BW]
	minBT=minBT.repeat(rowsBT,1)
	minBT=minBT.unsqueeze(0).repeat(batchBT,1,1)
	minBT=minBT.unsqueeze(2)

	# Concatenate:
	#   - Wbin:  [B, rowsW, colsW, BW] + [B, 1, colsW, BW] -> [B, rowsW+1, colsW, BW]
	#   - BTBin: [B, rowsBT, colsBT, BW] + [B, rowsBT, 1, BW] -> [B, rowsBT, colsBT+1, BW]
	return torch.cat((Wbin,minW),dim=1),torch.cat((BTBin,minBT), dim=2)



def converge4(Wbin, minW, BTBin, minBT):

	"""Converge/pad 5D multi-head tensors by appending minimum-value terms.

	Multi-head variant of :func:`converge`, operating on:
	  - Wbin:  [B, H, Rw, Cw, BW]
	  - BTBin: [B, H, Rbt, Cbt, BW]

	minW is broadcast/repeated to a layer of shape [B, H, 1, Cw, BW] and appended
	along the rows dimension (dim=2). minBT is broadcast to [B, H, Rbt, 1, BW]
	and appended along the cols dimension (dim=3).

	Args:
	    Wbin:   Quantized weights (5D): [batch, heads, rowsW, colsW, BW]
	    minW:   Quantized |min(W)| term (typically shape [BW])
	    BTBin:  Quantized inputs (5D):  [batch, heads, rowsBT, colsBT, BW]
	    minBT:  Quantized |min(BT)| term (typically shape [BW])

	Returns:
	    Wcat:   [B, H, rowsW+1, colsW, BW]
	    BTcat:  [B, H, rowsBT, colsBT+1, BW]
	"""

	batchW, multi_headW, rowsW, colsW, BW=Wbin.shape
	batchBT, multi_headBT, rowsBT, colsBT, BW=BTBin.shape

	# Alignment checks for batched multi-head matmul.
	assert colsW==rowsBT
	assert batchBT==batchW
	assert multi_headBT==multi_headW


	# Build a [B, H, 1, colsW, BW] plane containing minW replicated across columns.
	# minW is treated as a per-(model) scalar vector (BW bits), not per-row/per-col.
	minW_base = minW.view(1, 1, 1, 1, BW)
	minW_layer = minW_base.repeat(batchW, multi_headW, 1, colsW, 1)

	# Build a [B, H, rowsBT, 1, BW] plane containing minBT replicated across rows.
	minBT_base = minBT.view(1, 1, 1, 1, BW)
	minBT_layer = minBT_base.repeat(batchBT, multi_headBT, rowsBT, 1, 1)

	# Append the min planes:
	#   - W:  add one extra row (dim=2)
	#   - BT: add one extra column (dim=3)
	return torch.cat((Wbin,minW_layer),dim=2),torch.cat((BTBin,minBT_layer), dim=3)