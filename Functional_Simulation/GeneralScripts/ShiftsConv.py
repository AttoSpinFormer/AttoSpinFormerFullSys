
#!/usr/bin/env python3
"""
###############################################################################
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture
# Module:        ShiftsConv.py
# Description:   Core preprocessing routines for sign-aware quantization and tensor alignment.
#
# Synopsis:      This module implements various functions to prepare two input tensors 
#                (A and B, typically weights and inputs) for hardware simulation. 
#                Key steps include calculating the necessary **Shift metadata** (exponent 
#                for alignment), converting floating-point values to **fixed-point binary** 
#                using helper scripts, and isolating the sign information. It includes 
#                a specialized function (converge) for **padding/concatenating** 
#                the quantized tensors with their minimum value metadata.
#
# Created:       2025-11-11
# Last Modified: 2025-12-08
###############################################################################

Functions Provided:

Shifts_torch(A, B, n=16):
    Quantizes two generic signed input tensors (A and B). Tensor A uses an **offset quantization** scheme 
   (subtracting the minimum value) to handle its sign, while Tensor B uses **absolute value** quantization 
   with a separate sign tensor.
    
    Parameters:
        A (torch.Tensor): The first input tensor (e.g., weights), which may contain negative values.
        B (torch.Tensor): The second input tensor (e.g., activations), which may contain negative values.
        n (int): The target output **bit-width** (BW) for quantization (Default: 16).
    
    Returns: A1, minAbin, ShiftA, SignA, B1, ShiftB, SignB (quantized tensors and metadata).

Shifts_torch2(A, B, n=16):
    Quantizes two input tensors where A is assumed to be **non-negative** (e.g., Softmax output), 
    and B is a generic signed tensor quantized via absolute value and sign separation.
    
    Parameters:
        A (torch.Tensor): The first input tensor, assumed to be $\geq 0$.
        B (torch.Tensor): The second input tensor, which may contain negative values.
        n (int): The target output bit-width (BW) for quantization (Default: 16).

Shifts_torch3(A, n=16):
    Quantizes a single signed input tensor (A) using the **offset quantization** scheme.
    
    Parameters:
        A (torch.Tensor): The input floating-point tensor to be quantized.
        n (int): The target output bit-width (BW) for quantization (Default: 16).

converge(Wbin, minW, BTBin, minBT):
    Handles 3D tensor convergence (Rows x Cols x BW) for single-batch/single-head operations. 
    It concatenates the minimum value metadata (minW, minBT) onto the respective quantized 
    tensors (Wbin, BTBin) along the row and column dimensions.
    
    Parameters:
        Wbin (torch.Tensor): The quantized weight tensor (3D).
        minW (torch.Tensor): The quantized minimum value derived from the weights (W).
        BTBin (torch.Tensor): The quantized input/activation tensor (3D).
        minBT (torch.Tensor): The quantized minimum value derived from the input (BT).
"""


import math
import sys
sys.path.append('../')
import torch
from GeneralScripts.FracConv import ShiftCalc
from GeneralScripts.FracConv import fractions
from GeneralScripts.FracConv import vectorized_fractions
from GeneralScripts.FracConv import ShiftCalc_torch_vectorized


def Shifts_torch(A,B,n=16):

	minA_val = torch.min(A).item()
	AN = A - minA_val
	SignA=minA_val/abs(minA_val) if minA_val!=0 else 0
	minA_val_abs = abs(minA_val)
	max_A_val = torch.max(AN).item()
	DecValA, ShiftA = ShiftCalc(max(max_A_val, minA_val_abs), n)
	A1=vectorized_fractions(AN,ShiftA,DecValA,n)
	minAbin = torch.tensor(fractions(minA_val_abs, ShiftA, DecValA, n), dtype=A.dtype)

	max_B_val = abs(torch.max(B).item())
	min_B_abs_val = abs(torch.min(B).item())
	DecValB, ShiftB = ShiftCalc(max(max_B_val, min_B_abs_val), n)
	SignB = torch.sign(B)
	BN = torch.abs(B)
	B1=vectorized_fractions(BN,ShiftB,DecValB,n)
	return A1,minAbin,ShiftA,SignA,B1,ShiftB,SignB


def Shifts_torch2(A,B,n=16):

	assert torch.min(A).item()>=0, "Softmax matrix has some negative values"

	max_A_val = torch.max(A).item()
	DecValA, ShiftA = ShiftCalc(max_A_val, n)
	A1=vectorized_fractions(A,ShiftA,DecValA,n)

	max_B_val = abs(torch.max(B).item())
	min_B_abs_val = abs(torch.min(B).item())
	DecValB, ShiftB = ShiftCalc(max(max_B_val, min_B_abs_val), n)
	SignB = torch.sign(B)
	BN = torch.abs(B)
	B1=vectorized_fractions(BN,ShiftB,DecValB,n)
	return A1,ShiftA,B1,ShiftB,SignB


def Shifts_torch3(A,n=16):

	device=A.device
	minA_val = torch.min(A).item()
	AN = A - minA_val
	SignA=minA_val/abs(minA_val) if minA_val!=0 else 0
	minA_val_abs = abs(minA_val)
	max_A_val = torch.max(AN).item()
	DecValA, ShiftA = ShiftCalc(max(max_A_val, minA_val_abs), n)
	A1=vectorized_fractions(AN,ShiftA,DecValA,n)
	minAbin = torch.tensor(fractions(minA_val_abs, ShiftA, DecValA, n), dtype=A.dtype, device=device)

	return A1.to(device),minAbin,ShiftA,SignA



def converge(Wbin,minW,BTBin,minBT):

	rowsW,colsW,BW=Wbin.shape
	rowsBT,colsBT,BW=BTBin.shape

	assert colsW==rowsBT

	minW=minW.repeat(colsW,1)
	minW=minW.unsqueeze(0)
	#Wbin=torch.cat((Wbin,minW),dim=0).to(Wbin.device)

	minBT=minBT.repeat(rowsBT,1)
	minBT=minBT.unsqueeze(1)
	#BTBin=torch.cat((BTBin,minBT), dim=1).to(Wbin.device)

	return torch.cat((Wbin,minW),dim=0),torch.cat((BTBin,minBT), dim=1)