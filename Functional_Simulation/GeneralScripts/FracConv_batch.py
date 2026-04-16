

#!/usr/bin/env python3
"""
###############################################################################
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture
# Module:        FracConv_batch.py
# Description:   Utility functions for converting floating-point values to a fixed-point binary representation.
#
# Synopsis:      This module contains the mathematical and vectorized PyTorch 
#                routines required to implement custom quantization logic. It calculates 
#                the **binary equivalent** of a fractional number, determines the 
#                necessary **Exponent/Shift** (P) for alignment based on the maximum 
#                element magnitude, and performs the final conversion to a 
#                **Bit-Width (BW)** constrained integer representation. Can work on multi-batch, multi-head inputs.
#
#
# Created:       2025-11-11
# Last Modified: 2025-12-08
###############################################################################

Main Functions:

DecVal, Shift_Weight = ShiftCalc(Input_Max, Bit_Width): 
    Vectorized PyTorch implementation of the shift calculation (P) for an entire tensor, 
    based on the binary magnitude of the elements.
    
    Parameters:
        Input_Max (int)       The maximum element of the input floating-point tensor used to determine the necessary shift alignment.
        Bit_Width (int):      The target output bit-width used in the final shift calculation.

Out=vectorized_fractions(Input, FracVal, DecVal, Bit_Width)
    The **Primary Vectorized Conversion Utility**: Efficiently converts an entire 
    tensor of floating-point values (A) into a fixed-point binary representation 
    of length BW.
    
    Parameters:
        Input (torch.Tensor): The input floating-point tensor to be quantized.
        FracVal (int):        The total number of fractional bits required for precision (based on largest element).
        DecVal (int):         The number of bits required for the integer part of the largest element.
        Bit_Width (int):      The target output bit-width.

"""


import numpy as np
import math
import sys
import random
import time
import cmath
import torch

"""converts a fractional number into binary equivalent. support functions"""


# -----------------------------------------------------------------------------
# Scalar (Python list-based) reference helpers
# -----------------------------------------------------------------------------

def dec2bin(A):

	"""Return (DecVal, P) for a scalar magnitude A.

	DecVal:
		Number of bits needed for the integer part of A.
		- If A >= 1: DecVal = floor(log2(A)) + 1
		- If 0 < A < 1: DecVal = 0

	P:
		A shift/exponent-like value used by the quantizer.
		- If A >= 1: P = -floor(log2(A))
		- If 0 < A < 1: P = ceil(|log2(A)|)

	This helper is primarily used by ShiftCalc() to derive the scaling/shift.
	"""

	At=bin(int(A))[2:]

	# Special-case: A in (0,1) (i.e., int(A)==0) but non-zero magnitude.
	# The loop finds the smallest u such that A * 2^u >= 1.
	# Similar to a floating-point representation.
	if len(At)==1 and At[0]=='0' and A!=0:
		remainder=int(A)
		u=0
		while remainder<1:
			x1=A*2
			add=int(x1)
			remainder=add
			A=x1-add
			u+=1
		return 0,u
	else:
		# General case: A >= 1 (or A == 0).
		# - DecVal is the number of integer bits.
		# - The returned P is an exponent.
		return len(At),-(len(At)-1)


def ShiftCalc(B,BW):

	"""Compute integer-bit count and a BW-adjusted shift for scalar B.

	Args:
		B:  Scalar magnitude used to size the quantization range (often max(|X|)).
		BW: Target bit-width.

	Returns:
		DecVal: Integer-bit count needed for B.
		P:      Shift value used by downstream quantization.
			Our implementation uses: P = dec2bin(B)[1] + BW

	In practice, this P is often treated as a *fractional-bit budget* used when
	building the bit-vector representation.
	"""

	DecVal,P=dec2bin(B)
	P+=BW
	return DecVal,P


# -----------------------------------------------------------------------------
# Torch helpers (used by the vectorized quantization path)
# -----------------------------------------------------------------------------

def ShiftCalc_torch_vectorized(B, BW):
	"""Torch implementation of ShiftCalc().

	Args:
		B:  Scalar magnitude (torch.Tensor or Python scalar). Should be >= 0.
		BW: Target bit-width (Python int or scalar-like).

	Returns:
		(DecVal, P) as torch scalars (dtype: double).
	"""

	if not isinstance(B, torch.Tensor):
		B = torch.tensor(B).double()
	BW = torch.tensor(BW).double()

	# If max magnitude is 0, no integer bits are required; shift is just BW.
	if B.item() == 0:
		return torch.tensor(0.0).double(), BW


	# log2(B) is used to determine exponent/shift.
	# B must be positive. If B is negative, use |B| with this function.
	log_val = torch.log2(B)

	is_greater_than_one = B >= 1
    
	# For B >= 1:
	#   DecVal = floor(log2(B)) + 1
	#   P      = -floor(log2(B))
	DecVal_ge1 = torch.floor(log_val) + 1
	P_ge1 = -torch.floor(log_val)

	# For 0 < B < 1:
	#   DecVal = 0
	#   P      = ceil(|log2(B)|)  (how many left shifts to reach >= 1)
	DecVal_lt1 = torch.tensor(0.0).double()
	P_lt1 = torch.ceil(torch.abs(log_val))
    
	DecVal = torch.where(is_greater_than_one, DecVal_ge1, DecVal_lt1)
	P = torch.where(is_greater_than_one, P_ge1, P_lt1)

	# BW adjustment (consistent with ShiftCalc()).
	P += BW

	return DecVal, P


def frac2bin(A,P):

	"""Convert the fractional part of A into a length-P bit list (LSB-first).

	Args:
		A: Fractional value in [0,1).
		P: Number of fractional bits to generate.

	Returns:
		List[int] of length P with LSB-first ordering.
		(The final reverse() matches the LSB-first convention used elsewhere.)
	"""

	B,remainder=[],A
	while(len(B)<P):
		x1=remainder*2
		add=int(x1)
		B.append(add)
		remainder=x1-add

	# Reverse to produce LSB-first ordering.
	B.reverse()
	return(B)	
    
def fractions(B,M,DecVal,BW):

	"""Scalar reference conversion of a float B to a BW-bit vector (LSB-first).

	Args:
		B:      Scalar float value (typically non-negative).
		M:      Number of fractional bits to generate.
		DecVal: Number of integer bits used for the integer part.
		BW:     Output bit-width.

	Returns:
		List[int] of length BW, LSB-first.
	"""

	inB=int(B)

	# Integer bits: generate fixed-width binary string, then reverse for LSB-first.
	am=[int(b) for b in f"{inB:0{DecVal}b}"[::-1]]

	# Fractional bits + integer bits, both LSB-first.
	tot=frac2bin(B-inB,M)+am

	# Case handling when M > BW and there is no integer part => purely fractional element.
	Ap=0
	if M>BW and DecVal==0:
		Ap=M-BW

	# Slice out the BW bits (still LSB-first).
	return tot[len(tot)-BW-Ap:len(tot)-Ap]


# -----------------------------------------------------------------------------
# Vectorized conversion (PyTorch)
# -----------------------------------------------------------------------------

def vectorized_fractions(A, M, DecVal, BW):

	"""Vectorized conversion of a tensor A into BW-bit (LSB-first) representation.

	Args:
		A (torch.Tensor): Input floating-point tensor to convert. Commonly non-negative and
				  already shifted/normalized for CIM execution.
		M (int or scalar-like): Number of fractional bits to generate (often derived from ShiftCalc()).
		DecVal (int or scalar-like): Number of integer bits to allocate for the integer part.
		BW (int or scalar-like): Final bit-width of the output vector.

	Returns:
		torch.Tensor: Bit tensor of shape `A.shape + (BW,)`, containing 0/1 values. Bits are LSB-first along the last dimension.
	"""

	dtype = A.dtype
	device = A.device

	# Treat M/DecVal/BW as scalar tensors for device placement; later we call .item()
	# when we need Python integers for loop bounds/slicing.
	#M = torch.tensor(M, dtype=dtype, device=device)

	if torch.is_tensor(M):
		M = M.to(dtype=dtype, device=device)
	else:
		M = torch.tensor(M, dtype=dtype, device=device)

	DecVal = torch.tensor(DecVal, dtype=dtype, device=device)
	#BW = torch.tensor(BW, dtype=dtype, device=device)

	if torch.is_tensor(BW):
		BW = BW.to(dtype=dtype, device=device)
	else:
		BW = torch.tensor(BW, dtype=dtype, device=device)
    
	# Split into integer and fractional components.
	integer_part = torch.floor(A)
	fractional_part = A - integer_part

	# Build integer bits (LSB-first).
	# int_indices: [0, 1, ..., DecVal-1]
	int_indices = torch.arange(DecVal, device=device)

	# powers_of_2: [2^(DecVal-1), ..., 2^0] for MSB-first extraction.
	powers_of_2 = torch.pow(2.0, int_indices.flip(dims=[0]))

	# Expand integer_part to broadcast against powers_of_2.
	expanded_int = integer_part.unsqueeze(-1)
    
	# Extract integer bits (MSB-first), then flip to LSB-first.
	integer_bits_msb_first = torch.remainder(torch.div(expanded_int, powers_of_2, rounding_mode='floor'), 2).to(dtype)
    
	integer_bits = integer_bits_msb_first.flip(dims=[-1])

	# Generate fractional bits:
	#   - Loop index i corresponds to the i-th bit after the binary point (2^-1, 2^-2, ...).
	#   - We flip at the end to match the LSB-first convention (2^-M at index 0).
	frac_bits = torch.empty(A.shape + (int(M.item()),), device=device, dtype=dtype)
	remainder = fractional_part

	for i in range(int(M.item())):
		x1 = remainder * 2
		add = torch.floor(x1)
        
		frac_bits[..., i] = add.to(dtype)
		remainder = x1 - add
        
	fractional_bits = frac_bits.flip(dims=[-1])

	# Concatenate fractional and integer parts:
	# Layout (LSB-first overall):
	#   [2^-M ... 2^-1 | 2^0 ... 2^(DecVal-1)]
	tot = torch.cat((fractional_bits, integer_bits), dim=-1)


	# Ap is an additional offset used when the value is purely fractional and we have
	# more fractional bits than the target BW. This behavior mirrors the scalar
	# `fractions()` helper.
	Ap = torch.tensor(0.0, device=device,dtype=dtype)
	if (M.item() > BW.item()) and (DecVal.item() == 0):
		Ap = M - BW-1

	tot_len = tot.shape[-1]
	Ap_int = int(Ap.item())
	BW_int = int(BW.item())

	# Select BW bits from the concatenated vector.
	# Because bits are LSB-first, taking a tail slice drops the least-significant
	# (low-precision) bits when tot is longer than BW.    
	output = tot[..., tot_len - BW_int - Ap_int : tot_len - Ap_int]

	return output