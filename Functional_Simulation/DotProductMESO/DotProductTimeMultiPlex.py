
#!/usr/bin/env python3
"""
###############################################################################
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture
# Module:        DotProductTimeMultiPlex.py
# Description:   Top-level module for interfacing MESO IMC with test scripts 
#		 Time multiplexing of data through gate within the MESO IMC module. 
#
# Synopsis:      This module processes inputs received from test scripts,
#		 converting floating-point values into the required quantized binary format,
#		 determines the associated shift and sign metadata for the inputs, and
#		 subsequently submits these processed inputs and metadata to the core IMC module (IMCMESOMultiPlexTop.py)
#
# Limitation:    This implementation is restricted solely to single-head, multi-batch configurations.
# Reference:     For multi-head execution capability, clients must utilize the dedicated
#                module: AttMESOTimePlex.py.
#
# Created:       2026-03-05
# Last Modified: 2026-03-10
###############################################################################

Usage and Interface:

    The primary interface is provided by the 'FPMatMulMESO' class:

    constructor: FPMatMulMESO(Bit_Width)
    Method Call Inputs: Queries (torch.Tensor), Keys (torch.Tensor).
    
Parameters:

    Bit_Width :              Defines the quantization bit-width applied to the tensors.
    Input Dimension :        Specifies the input tensor shape as Batch x Rows x Columns.

Consult TestScripts/MESODotProductCheck.py for comprehensive implementation examples and operational context.
"""



import math
import sys
sys.path.append('../')
import cmath
import torch
import torch.nn as nn
import torch.nn.functional as F
import cProfile


# MESO core: performs the actual IMC dot-product / matmul given binary operands
from DotProductMESO.IMCMESOMultiPlexTop import Top


# Quantization + metadata helpers:
#   * Shifts_torch3  : quantizes a tensor into a binary representation and emits
#                      1 min/1 shift/1 sign metadata per tensor.
#   * converge       : aligns/merges two quantized tensors.
from GeneralScripts.ShiftsConv_batch import Shifts_torch3
from GeneralScripts.ShiftsConv_batch import converge

from torch.autograd import Function


# In the test-script naming used by this project:
#   - Queries  are treated as the "weights" matrix (W)
#   - Keys     are treated as the "inputs" matrix (K)
# The MESO core appears to expect the two tensors to be aligned for dot-product. Weight: (b x s x d) and BT: (b x d x s)

class FPMatMulAutogradMESO(Function):

	"""Custom autograd Function wrapper around MESO fixed-point matmul.

	Forward:
		 1. Transpose `Keys` into `BT` (B‑transpose) as expected by the MESO datapath,
		    ensuring the operands are correctly aligned for the dot‑product computation.
		 2. Decompose each input tensor into two constituent tensors: one capturing the
		    fine‑grained (residual) values shifted by the input tensor's minimum value 
		    and the other representing the coarse/minimum.
		    component, enabling CIM‑friendly processing. The constituent tensors differ in size.
		 3. Quantize the constituent tensors of the two input tensors independently and align their
		    mantissas/exponents as required by the CIM numeric format.
		 4. Concatenate the quantized constituents to form packed tensors for
		    simplified CIM execution.
		 5. Invoke the MESO top‑level core (`Top`) to compute and return the matmul result.

	Backward:
		Uses a floating-point batched-matmul gradient (surrogate). This does not
		backprop through quantization or the IMC execution path.
	"""


	@staticmethod
	def forward(ctx, Queries, Keys, module_instance):
		# Save tensors for backward() and keep a reference to the module instance.
		ctx.save_for_backward(Queries, Keys)
		ctx.module_instance = module_instance 

		# BT is the transpose of Keys across the last two dimensions.
		# If Keys is shaped [B, S, D], BT becomes [B, D, S].
		BT= Keys.permute(0,2,1)
		Weight=Queries

		# Basic dimensionality sanity check for matmul:
		#Weight: [B, rowsW, colsW]
		#BT: 	 [B, rowsBT, colsBT]
		#Requirement: colsW==rowsBT
		batchW, rowsW, colsW=Weight.shape
		batchBT, rowsBT, colsBT=BT.shape
		if colsW!=rowsBT:
			raise ValueError(
					f"Dimensions of the Queries and Keys don't match. "
					f"Weight shape: {Weight.shape}, BT shape: {BT.shape}"
					)
		
		# Bit width used by quantization/encoding stage.	
		BW = module_instance.BW 
		Var = module_instance.var
		InRes = module_instance.InRes

		# Quantize Weight and BT.
		# Shifts_torch3 returns:
		#   BinTensor, Bin_min_val, ShiftMeta, SignMeta
		# Tensor decomposition outputs (CIM/MESO-ready):
		#   BinTensor    : Quantized, mantissa-aligned binary representation of the shifted tensor’s
		#                  non-negative (all-positive) elements. This encodes the fine-grained component
		#                  after shifting by the original tensor’s minimum value.
		#   Bin_min_val  : Quantized, mantissa-aligned binary representation of |min(original_tensor)|,
		#                  i.e., the offset term used to undo/track the shift applied above.
		#
		# Metadata:
		#   ShiftMeta / SignMeta : Metadata values that capture the scaling (shift) and sign
		#                          handling needed by the MESO core during computation/reconstruction.
		WBin,minW,ShiftW,SignW = Shifts_torch3(Weight, BW)
		BTBin,minBT,ShiftBT,SignBT = Shifts_torch3(BT, BW)


		#concatenate the constituent tensors.
		WBinMerge,BTBinMerge=converge(WBin,minW,BTBin,minBT)

		# Execute the MESO core.
		Output = Top(WBinMerge, ShiftW, SignW, BTBinMerge, ShiftBT, SignBT, Weight, BW, Var, InRes)

		# Preserve the input dtype (e.g., float16/float32) for downstream ops.
		return Output.to(Queries.dtype)

	@staticmethod
	def backward(ctx, grad_output):

		"""Surrogate backward pass.

		The MESO forward path is non-differentiable due to quantization and
		hardware-like execution. Here we return a *floating-point* gradient that
		matches a corresponding batched matmul.

		IMPORTANT:
			The exact transpose convention depends on how callers interpret
			(Queries, Keys). Verify this matches your forward semantics.
		"""

		Queries, Keys = ctx.saved_tensors

		# Compute gradients via floating-point batched matrix multiplies.
		# NOTE: The original code used Keys.transpose(1, 2) for grad_queries.
		# Depending on how Keys is shaped at call sites, you may need Keys (not
		# transposed) here to match `Y = Queries @ Keys^T`.
		grad_queries = torch.bmm(grad_output, Keys.transpose(1, 2))
		grad_keys = torch.bmm(grad_output.transpose(1, 2), Queries)

		# The third return corresponds to module_instance (non-tensor), so None.
		return grad_queries, grad_keys, None


##The below classes are used by Fully-connected layers of neural networks.
class FPMatMulTimePlexMESO(nn.Module):

	"""nn.Module wrapper around the custom autograd Function.

	Usage:
		mm = FPMatMulMESO(bit_width=bit_width)
		y  = mm(Queries, Keys)
	"""

	def __init__(self,bit_width=16, var=0, cycle_res=4):
		super().__init__()
		# Bit width used during quantization/encoding.
		self.BW=bit_width
		self.var=var
		self.InRes=cycle_res

	def forward(self,Queries,Keys):
		# Delegate to the custom autograd function.
		return FPMatMulAutogradMESO.apply(Queries,Keys,self)


class FPLinearMESO(nn.Module):

	"""Example linear layer that uses MESO matmul.

	This is a convenience wrapper for experiments. It defines parameters:
		* weight: [out_features, in_features]
		* bias  : [out_features]

	and uses FPMatMulMESO to compute the affine transform.
	"""
	def __init__(self, in_features, out_features, bit_width=8):
		super(FPLinearMESO, self).__init__()
		# Weight init: scaled normal initialization.
		self.weight = nn.Parameter(torch.randn(out_features,in_features)*2/torch.sqrt(torch.tensor(in_features)))
		self.bias = nn.Parameter(torch.randn(out_features))

		# MESO-backed matmul module (quantizes at the specified bit-width).
		self.fixed_point_mm = FPMatMulMESO(bit_width=bit_width)

	def forward(self, input):
		# Compute matmul and add bias.
		output = self.fixed_point_mm(input, self.weight).T+ self.bias
		return output


