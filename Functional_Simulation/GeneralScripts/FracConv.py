

#!/usr/bin/env python3
"""
###############################################################################
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture
# Module:        FracConv.py
# Description:   Utility functions for converting floating-point values to a fixed-point binary representation.
#
# Synopsis:      This module contains the mathematical and vectorized PyTorch 
#                routines required to implement custom quantization logic. It calculates 
#                the **binary equivalent** of a fractional number, determines the 
#                necessary **Exponent/Shift** (P) for alignment based on the maximum 
#                element magnitude, and performs the final conversion to a 
#                **Bit-Width (BW)** constrained integer representation.
#
# Limitation:    This module is designed exclusively for single-batch configurations.
# Reference:     For multi-head execution capability, please utilize the dedicated 
#                module: FracConv_batch.py.
#
# Created:       2025-11-11
# Last Modified: 2025-12-08
###############################################################################

Functions Provided:

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

---

Supporting Functions (Scalar/Legacy):

def dec2bin(A):
    Calculates the integer part length (DecVal) and the exponent shift (P) for a single scalar value.

def ShiftCalc_torch_vectorized(B, BW):
    Calculates the final Shift value (P) required for alignment based on the bit-width (BW).

def frac2bin(A, P):
    Converts the fractional part of a number into a list of binary digits up to precision P.

def fractions(B, M, DecVal, BW):
    The main scalar conversion function: combines integer and fractional bits and truncates 
    the result to fit the target Bit-Width (BW) based on the calculated alignment (Ap).
"""


import numpy as np
import math
import sys
import random
import time
import cmath
import torch

"""converts a fractional number into binary equivalent. support functions"""

def dec2bin(A):
	At=bin(int(A))[2:]
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
		return len(At),-(len(At)-1)


def ShiftCalc(B,BW):
	DecVal,P=dec2bin(B)
	P+=BW
	return DecVal,P


def ShiftCalc_torch_vectorized(B, BW):
	if not isinstance(B, torch.Tensor):
		B = torch.tensor(B).double()
	BW = torch.tensor(BW).double()

	if B.item() == 0:
		return torch.tensor(0.0).double(), BW

	log_val = torch.log2(B)

	is_greater_than_one = B >= 1
    
	DecVal_ge1 = torch.floor(log_val) + 1
	P_ge1 = -torch.floor(log_val)

	DecVal_lt1 = torch.tensor(0.0).double()
	P_lt1 = torch.ceil(torch.abs(log_val))
    
	DecVal = torch.where(is_greater_than_one, DecVal_ge1, DecVal_lt1)
	P = torch.where(is_greater_than_one, P_ge1, P_lt1)

	P += BW

	return DecVal, P


def frac2bin(A,P):
	B,remainder=[],A
	while(len(B)<P):
		x1=remainder*2
		add=int(x1)
		B.append(add)
		remainder=x1-add
	B.reverse()
	return(B)	
    
def fractions(B,M,DecVal,BW):
	inB=int(B)
	am=[int(b) for b in f"{inB:0{DecVal}b}"[::-1]]
	tot=frac2bin(B-inB,M)+am
	Ap=0
	if M>BW and DecVal==0:
		Ap=M-BW
	return tot[len(tot)-BW-Ap:len(tot)-Ap]


def vectorized_fractions(A, M, DecVal, BW):

	dtype = A.dtype
	device = A.device

	M = torch.tensor(M, dtype=A.dtype).to(device)
	DecVal = torch.tensor(DecVal, dtype=A.dtype).to(device)
	BW = torch.tensor(BW, dtype=A.dtype).to(device)
    
	integer_part = torch.floor(A)
	fractional_part = A - integer_part

	int_indices = torch.arange(DecVal, device=device,dtype=torch.int16)
	powers_of_2 = torch.pow(2.0, int_indices.flip(dims=[0]))
    
	expanded_int = integer_part.unsqueeze(-1)
    
	integer_bits_msb_first = torch.remainder(torch.div(expanded_int, powers_of_2, rounding_mode='floor'), 2).to(dtype)
    
	integer_bits = integer_bits_msb_first.flip(dims=[-1])

	frac_bits = torch.empty(A.shape + (int(M.item()),), device=device, dtype=dtype)
	remainder = fractional_part

	for i in range(int(M.item())):
		x1 = remainder * 2
		add = torch.floor(x1)
        
		frac_bits[..., i] = add.to(dtype)
		remainder = x1 - add
        
	fractional_bits = frac_bits.flip(dims=[-1])

	tot = torch.cat((fractional_bits, integer_bits), dim=-1)

	Ap = torch.tensor(0.0, device=device,dtype=dtype)
	if (M.item() > BW.item()) and (DecVal.item() == 0):
		Ap = M - BW-1

	tot_len = tot.shape[-1]
	Ap_int = int(Ap.item())
	BW_int = int(BW.item())
    
	output = tot[..., tot_len - BW_int - Ap_int : tot_len - Ap_int]
    
	return output.to(dtype)