
#!/usr/bin/env python3
"""
###############################################################################
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture
# Module:        Process_pytorch_MESO.py
# Description:   Core PyTorch module for MESO In-Memory Computing (IMC) kernel execution.
#
# Synopsis:      This module orchestrates the IMC pipeline: accepting split
#                quantized weight and time-multiplexed input matrices, simulating crossbar 
#                computation, and converting analog outputs (Current -> Voltage) 
#                into digital floating-point results. It incorporates the
#                required compensation term before reorganizing the outputs 
#                to their final, correct tensor dimensions.
#
# Limitation:    This module is designed exclusively for single-head configurations.
# Reference:     For multi-head execution capability, please utilize the dedicated 
#                module: AttProcessTimePlex.py.
#
# Created:       2025-11-11
# Last Modified: 2026-03-10
###############################################################################

Usage and Interface:

    The primary entry point is the 'VectorizedProg_torch' function, which operates on PyTorch tensors:
    
    Out = VectorizedProg_torch(Weights_Split, Weight_Shifts, Weight_Sign, Inputs_Split, 
                               Input_Shift, Input_Sign, Weight_Matrix, Bit_Width)

Parameters:

    Weights_Split (torch.Tensor): Quantized weight tensor, partitioned into fixed 256x64 blocks for parallel crossbar execution 
                                  (e.g., FCC weights, attention queries/keys/softmax).
    Inputs_Split (torch.Tensor):  Quantized input tensor, partitioned for effective IMC parallelization.
    XXX_Shifts (torch.Tensor):    Exponent of the largest element relative to the defined Bit_Width (calculated via ShiftsConv_batch utility).
    XXX_Sign (torch.Tensor):      Sign indicator of the minimum element within the weight matrix used for sign reconstruction.
    Weight_Matrix (torch.Tensor): The original floating-point weight matrix used for reference and compensation.
    Bit_Width (int):              The quantization bit-width applied to the input and weight tensors.

Dependencies:
    Requires torch.Tensor inputs for all relevant parameters.

Refer to MESODSTop.py for upstream data handling and comprehensive implementation examples.
"""


import math
import sys
sys.path.append('../')

import torch

def VectorizedProg_torch(A1, P1, SignA, B1, P2, SignB, Ax, InRes, cycles, n=8):

	"""
	Run the processing and post-processing cycles of the MESO/CIM vectorized dot-product (matmul) 
	pipeline on quantized/split tensors.

	This function models the IMC datapath at a high level:
	  1. Computes per-tile analog dot-products and compensation term (CompMESO) through time-multiplexing using a crossbar-style einsum.
	  2. Converts Output current to voltage through integration and biasing (Output voltage: [0,1]). 
	  3. Applies an ADC quantization model (discretization).
	  4. Adds compensation terms to account for the negative current of MESO for a logical '0' (check the AttoSpinFormer paper). 
	  5. Recombines bit-slices using exponent/mantissa alignment metadata (P1/P2).
	  6. Adds the compensation terms required by the "shift-by-min" tensor decomposition
	    (used to make operands non-negative for CIM).

	Expected tensor shapes
	---------------------
	A1: (batch, bit_width, row_tiles_A, col_tiles_A, crossbar_rows, crossbar_cols)
	B1: (batch, row_tiles_B, crossbar_cols, colsB_total)
	where col_tiles_A == row_tiles_B and crossbar_cols matches between A1/B1 for matmul.

	Args:
		A1: Shifted + Quantized + split weight tensor (bit-sliced and tiled).
		P1: Weight shift/exponent metadata (from mantissa alignment).
		SignA: Sign of the minimum/offset term for the weight decomposition (+/- 1).
		B1: Shifted + Quantized + split activation/input tensor + time-multiplexed (tiled to match A1).
		P2: Input shift/exponent metadata (from mantissa alignment).
		SignB: Sign of the minimum/offset term for the input decomposition (±1).
		Ax: Reference tensor used to recover the intended output size. 
		n:  bit_width.

	Returns:
		torch.Tensor: Reconstructed matmul result with compensation terms applied.
	"""

	torch.set_printoptions(threshold=float('inf'))
	
	dtype=A1.dtype
	device = A1.device

	I=torch.tensor(10e-6, device=device, dtype=dtype)
	# Reference MESO output current.
	
	# Unpack tensor shapes used throughout the pipeline.
	# B1 is a 4-D tiled activation tensor; 
	#A1 is a 6-D bit-sliced + tiled weight tensor.
	#Here, final dimension is colsB*cycles
	batchB,rowbatchB, rowsB, colsB = B1.shape
	batchA,bitsA, rowbatchA, colsbatchA, rowsA, colsA =A1.shape

	# ADC resolution model: chooses enough bits to represent a sum of 'colsA' terms, based on MESO and input pulse resolution.
	vals=2**InRes-1
	ADCres=torch.round(torch.log2(torch.tensor(colsA*2*vals+1).float())).item() #=round(log2(1+2*colsA*(2^InRes-1)))

	print(f"The ADC resolution for the cycle resolution of {InRes} and crossbar cols of {colsA} is {ADCres}")

	#Convert ADC resolution into a quantization scale.
	Res=pow(2,ADCres)-1

	# R models the load resistance connected to the sense amplifier, which converts the
	# accumulated crossbar output current into a voltage.
	# It is derived by normalizing against the maximum possible accumulated current
	# (I * colsA), ensuring the resulting analog accumulation remains comparable
	# across different dot‑product lengths.
	R=float(1)/(I*colsA)


	#R,C are the resistors and capacitors used for integrating the crossbar output current. 
	#we make sure that RC>>>time period. Here, time period is 100ns. 
	#This results in the equation that Vout=sigma(-Iout*t/Cap). 
	#If the RC condition doesnt hold, then this entire dot-product module will fail. 
	timePeriod = torch.tensor(100e-9, device=device, dtype=dtype) #Overall time period. 
	Cap=float((timePeriod*I*colsA)/0.5)
	multFactor=timePeriod/(Cap*vals)

	#Assertion-based Sanity checks: A1/B1 must agree on the number of tiles along the reduction axis,
	#and batch dimensions must match.
	assert colsbatchA==rowbatchB, "dimensionality mismatch"
	assert batchA==batchB, "minibatch mismatch"

	# Columns are grouped by bit-slices. 'bCols' is the effective number of output column
	# groups once bit-sliced packing is accounted for.
	bCols= colsB// cycles
	

	# Crossbar-style multiply-accumulate (vectorized):
	#   A1 carries (bitsA) bit-slices and tiling in both row/col;
	#   B1 is tiled to match the A1 column-tiles (reduction axis).
	# The einsum produces per-crossbar dot-products with explicit bit-slice dimension preserved.
	# Multiplying with "multFactor converts the output current to integrator output voltage (bipolar). 
	out=torch.einsum('btxjlm,bjmu->btxjlu', A1, B1)*multFactor

	# Biasing circuit. Shifts the output voltage to unipolar. 
	out=0.5+out

	# Analog -> digital model: apply an ADC transfer/quantization model.
	# Convert analog voltage values to discrete digital integers.
	# Map normalized values into ADC code space.
	out=out*Res

	# Discretize to integer ADC codes.
	out=torch.round(out) 

	#Subtraction of the compensation term (CompMESO) to negate the effects of negative MESO currents and bias shifting (check the AttoSpinFormer paper). 
	outN=out[...,:-1,:]-out[...,-1:,:]

	# Sum across the A1 'col-tiles' dimension (axis=3 in the einsum output) to accumulate
	# partial dot-products into full dot-products for each row-tile and output column.
	out=outN.sum(dim=3).to(dtype)


	# Rescale after correction (integer right shift). This effectively divides by 2 and is
	# also a compensation term to account for the negative MESO current flow (check the AttoSpinFormer paper)
	out=torch.round(out / 2.0)

	batchF,rowsF,colsF=Ax.shape


	# Bit-slice recombination factor. Each bit-slice is weighted by its significance and
	# the mantissa-alignment shifts (P1, P2).
	# Note: P1 and P2 are typically per-(batch,row/col tile) metadata; broadcasting relies
	# on PyTorch's implicit expansion rules.
	# P1 : Shift term for the weight tensor
	# P2: Shift term for the activations tensor.
	factor=torch.pow(2.0, torch.arange(bitsA, dtype=dtype, device=device)-P1-P2+2).to(dtype)

	# Combine the bit-slice dimension (dim=1) into a single fixed-point value per tile.
	outNew=torch.sum(out * factor.reshape(1,-1, 1, 1, 1), dim=1, dtype=dtype)

	# Repack the column dimension into (bCols, bitsA) so we can apply a second-stage
	# recombination across the packed bit-slices per output column group.
	outNew_reshaped=outNew.reshape(batchA, rowbatchA, rowsA-1, bCols, cycles)
	
	# Final bit-weighted accumulation across the packed time-multiplexed bit dimension.
	base = vals+1
	powers = torch.pow(base , torch.arange(cycles, device=device, dtype=dtype)).to(dtype)
	out3=torch.sum(outNew_reshaped * powers, dim=-1, dtype=dtype)

	# Collapse tiled rows into a single contiguous row dimension.
	out3_reshaped=out3.reshape(out3.shape[0], -1, out3.shape[-1])

	# Trim any padding introduced by upstream tiling. Upstream typically appends an extra
	# row/column that store the decomposition offset terms needed for reconstruction.
	out4_optimized = out3_reshaped[:, :(rowsF+1), :]

	# Reconstruction for shift-by-min decomposition:
	#   Upstream decomposes each operand as:
	#       X' = X - SignX * ABX,   where SignX = sign(min(X)) and ABX = |min(X)|
	#       Y' = Y - SignY * ABY,   where SignY = sign(min(Y)) and ABY = |min(Y)|
	#
	#   Expanding the original product gives:
	#       X * Y = X'Y' + (SignX*ABX)*Y' + (SignY*ABY)*X' + (SignX*SignY*ABX*ABY)
	#
	#   `out4_optimized` reserves its last row/column to carry the correction terms required to
	#   reconstruct X*Y from the shifted products: (SignX*ABX)*Y', (SignY*ABY)*X', and
	#   SignX*SignY*ABX*ABY.
	
	weighted_row_term = SignA * out4_optimized[:, rowsF, :bCols-1].unsqueeze(1)
	weighted_col_term = SignB * out4_optimized[:, :rowsF, bCols-1].unsqueeze(2)

	base_slice = out4_optimized[:, :rowsF, :bCols - 1]
	out4x = base_slice + weighted_row_term + weighted_col_term

	# Scalar offset term (min(X)*min(Y)) applied uniformly across all outputs in this tile.
	scalar_term = SignB * SignA * out4_optimized[:, rowsF, bCols-1]

	# Final reconstructed result (base + row term + col term + scalar term).
	out4x_Final=out4x+scalar_term.unsqueeze(1).unsqueeze(2)

	return out4x_Final.to(dtype)

