
#!/usr/bin/env python3
"""
###############################################################################
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture
# Module:        MESODotProductCheck.py
# Description:   Verification script for the MESO In-Memory Computing (IMC) module.
#
# Synopsis:      This module performs functional validation and quantitative error
#                analysis of the MESO IMC pipeline. It generates synthetic input 
#                tensors (Batch x Rows x Columns), executes both the IMC module 
#                and a reference CMOS GPU implementation, and computes the output 
#                error metric based on their divergence.
#
# Created:       2025-11-11
# Last Modified: 2026-03-27
###############################################################################

Usage and Interface:

    The primary execution interface is provided by the module's 'main' entry point.

Execution Parameters:

    bit_width_input (int): Defines the quantization bit-width applied to all processed tensors 
                           during test case generation.

Execution Command:
    
    This script is executed directly from the terminal: 
    python3 MESODotProductCheck.py 

Note: The output error metric provides a measure of fidelity between the simulated IMC hardware 
      and the standard floating-point CMOS implementation.

Practical considerations:
    - Large sequence lengths / depths can exhaust device memory (especially on GPU).
    - The module import below assumes the repository layout is unchanged and that this script is executed from a directory where '../' resolves correctly.
"""


# -----------------------------------------------------------------------------
# Standard library imports
# -----------------------------------------------------------------------------
import math
import sys
import cmath
import cProfile

# -----------------------------------------------------------------------------
# Third Party imports
# -----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Local / Project imports
# -----------------------------------------------------------------------------
sys.path.append('../')
from DotProductMESO.DotProductTimeMultiPlex import FPMatMulTimePlexMESO


# -----------------------------------------------------------------------------
# Device selection
# -----------------------------------------------------------------------------
# This code was tested extensively on mps.
#uncomment the "cuda" line if you wish to experiment. 
device="cpu"
#device = (
#	"mps" if torch.backends.mps.is_available()
#	#else "cuda" if torch.cuda.is_available()
#	else "cpu"
#	)


torch.set_printoptions(threshold=50000)

# -----------------------------------------------------------------------------
# User input for both USER and PRE modes: quantization bit-width
#   The MESO model is configured with this bit width. Valid range: 2-32.
# -----------------------------------------------------------------------------
print("Dot-product simulation using MESO-based CiM. Input1 represents Queries/softmax while Input2 represents Keys/Values.")
print("The default bit width is 8-bits and feature depth is 64.")
print("--------------------------------------------------")

while True:
	try:
		bit_width_input = input("Enter the required bit width (2<= Input <=32): ")
        
		bit_width = int(bit_width_input)
        
		if 1 < bit_width < 33:
			break
		else:
			print("Please enter a bit width between 2 and 32.")
            
	except ValueError:
		print("Invalid input. Please enter an integer.")


print(f"Bit-width selected: {bit_width}\n")
print("\n" + "-"*60 + "\n")


while True:
	try:
		var_input = input("Enter the required variability (0<= Input <=1): ")
        
		var = float(var_input)
        
		if 0 <= var <= 1:
			break
		else:
			print("Please enter a variability between 0 and 1.")
            
	except ValueError:
		print("Invalid input for variability.")


print(f"Variability selected: {var}\n")
print("\n" + "-"*60 + "\n")

while True:
	try:
		res_input = input("Enter per cycle resolution for the input pulse (1<= resolution <=7): ")
        
		cycle_res = int(res_input)
        
		if 1 <= cycle_res <= 7:
			break
		else:
			print("Please enter a per cycle resolution between 1 and 7.")
            
	except ValueError:
		print("Invalid input for cycle resolution.")


print(f"Cycle resolution selected: {cycle_res}\n")
print("\n" + "-"*60 + "\n")


# -----------------------------------------------------------------------------
# User input for both USER and PRE modes: input value distribution / sign mode
# This determines whether random inputs remain in [0,1) or are shifted to [-1,1).
# ALL_NEG range: 
#	Weight: [-1,1), BT: [-1,1)
# W_NEG_BT_POS range: 
#	Weight: [-1,1),  BT: [0,1)
# ALL_POS range: 
#	Weight: [0,1),  BT: [0,1)
# -----------------------------------------------------------------------------

print("\n--- Select Matrix Parameters ---")
print("1: Both the input matrices have positive and negative elements. Input1: [-1,1); Input2: [-1,1)")
print("2: Elements of Input1 are Positive, and elements of Input2 are mixed. Input1: [-1,1); Input2: [0,1).")
print("3: Elements of both the matrices are Positive. Input1: [0,1); Input2: [0,1)")
    
mode_input = input("Enter selection (1, 2, or 3): ")
    
# Map the input to a key/string for easier condition checking
if mode_input == '1':
	mode = "ALL_NEG"
elif mode_input == '2':
	mode = "W_NEG_BT_POS"
elif mode_input == '3':
	mode = "ALL_POS"
else:
	print("Invalid input. Defaulting to Both Positive (Mode 3).")
	mode = "ALL_POS"
    
print(f"Matrix Mode selected: {mode}\n")

print("\n" + "-"*60 + "\n")


# -----------------------------------------------------------------------------
# User input: run configuration
#   USER mode: one (sequence length, depth) configuration.
#	       User defined inputs: sequence length, depth. Batch=1; num_tests=10.
#   PRE  mode: sweep sequence length over powers of two (depth fixed at 64)
#	       depth=64; batch=1; num_tests=100; sequence length: 64-2048.
# -----------------------------------------------------------------------------

print("\n--- Select Simulation Mode ---")
print("1: User defined sequence length and depth.")
print("2: Pre-defined run case (Depth: 64, Range: 64-2048).")

while True:
	try:
		sim_input_init = input("Enter simulation mode choice (1 or 2): ")
        
		sim_input = int(sim_input_init)
        
		if sim_input==1 or sim_input == 2:
			break
		else:
			print("Please enter a simulation mode choice between 1 and 2.")
            
	except ValueError:
		print("Invalid input for simulation mode.")


#sim_input = input("Enter simulation mode choice (1 or 2): ")

modes='USER' if sim_input==1 else 'PRE'
print(f"Simulation Mode selected: {modes}\n")
print("\n" + "-"*60 + "\n")

if sim_input==1:
	sim_mode = "USER"
	print("Note: Larger depths and sequence lengths can cause simulations to fail due to memory exhaustion or operand/output tensors growing beyond practical system limits, especially at higher bit-widths.")

	depth=64

	try:
		depth = int(input("Enter depth (2-256): "))

		if depth<1:
			print("Depth too small. Defaulting to 64.")
	except ValueError: 
		print("Invalid input for Depth. Expecting an integer. Defaulting to 64.")

	seqL=128
	try:
		seqL= int(input("Enter sequence length (2-8192): "))
		if seqL<1:
			print("Sequence length too small. Defaulting to 128.")

	except ValueError:
		print("Invalid input for sequence length. Expecting an integer. Defaulting to 128.")

elif sim_input == 2:
	sim_mode = "PRE"

else:
	print("Invalid input. Defaulting to Pre-defined run case.")
	sim_mode = "PRE"


# -----------------------------------------------------------------------------
# Derived experiment parameters
#   num_tests: number of random trials per configuration. PRE: 100, USER: 10.
#   Depth:     feature dimension D (fixed at 64 in PRE mode)
#   sizeL:     sequence length L (fixed in USER mode; swept in PRE mode)
#   r1/r2:     exponent range for sweep: sizeL = 2^l1 for l1 in [r1, r2). PRE mode.
# -----------------------------------------------------------------------------

num_tests=100 if sim_mode=="PRE" else 100
Depth = 64 if sim_mode=="PRE" else depth
sizeL = 128 if sim_mode=="PRE" else seqL
batch = 1
r1, r2= 6, 12


print(f"Simulation Mode: {sim_mode}, Depth: {Depth}, Sequence Length: {sizeL if sim_mode=='USER' else 'Variable'}, batch size: {batch}, #tests: {num_tests}")	
print("\n" + "-"*60 + "\n")

def run(sizeL,Depth,batch,num_tests,sim_mode,mode,r1,r2,bit_width):

	"""Run MESO dot-product tests and report relative error.

	This function compares the MESO simulated matrix multiplication against
	a floating-point reference (torch.matmul) for randomly generated inputs.

	Shapes
	------
	Weight: (batch, sizeL, Depth)
	BT:     (batch, sizeL, Depth)
	Output: (batch, sizeL, sizeL)  where Output = Weight @ BT^T

	Error Metric
	------------
	Relative L1 error (percentage):
		100 * |ref - meso| / |ref|
	summed over output matrix dims and averaged over the batch.
	"""

	# Instantiate the custom MESO simulated matmul module.
	model=FPMatMulTimePlexMESO(bit_width=bit_width, var=var, cycle_res=cycle_res).to(device)

	ErrorTot=0
	ErrorC=0

	if sim_mode == "USER":
		for test in range(num_tests):
			# Generate random inputs in [0,1) for the requested (batch, L, D) shape.
			Weight = torch.rand(batch, sizeL, Depth, dtype=torch.float32, device=device)
			BT = torch.rand(batch, sizeL, Depth, dtype=torch.float32, device=device)


			# Optionally shift inputs to introduce negative values depending on the selected mode.
			#   (2*x - 1) maps [0,1) -> [-1,1).
			if mode == "ALL_NEG":
				# Make both inputs's range: [-1,1)
				Weight = (2*Weight)-1
				BT = (2*BT)-1
			elif mode == "W_NEG_BT_POS": 
				#Weight: [-1,1), BT: [0,1)
				Weight = (2*Weight)-1

			# MESO model output (simulated in-memory dot product / CiM pipeline).
			Output=model(Weight,BT)

			# Reference output computed in floating point (GPU/CPU torch.matmul).
			# Permute BT from (B,L,D) to (B,D,L) to form BT^T for batched matmul.
			OutNew = torch.matmul(Weight, BT.permute(0,2,1))


			#print(OutNew, Output)
			#Compute relative L1 error (%) for each batch element. 
			ErrorC = (torch.abs(OutNew - Output).sum(dim=(1,2)) * 100) / torch.abs(OutNew).sum(dim=(1,2))
	
			print(f"Test batch: {test}, mode: {mode}, bit-width: {bit_width}, sequence len: {sizeL}, depth: {Depth}, ErrorC: {ErrorC.mean():.4f}%")	
			
			#Accumulate the per-test error into a running total and compute the average error over all tests for a user-defined sequence length and depth..
			ErrorTot+=ErrorC.mean()

		TotalTests=batch*num_tests
		print("--------------------------------------------------")
		print(f"#Tests: {TotalTests}, simulation mode: {sim_mode}, mode: {mode}, bit-width: {bit_width}, sequence len: {sizeL}, depth: {Depth}, ErrorC: {ErrorTot/num_tests:.4f}%")	
		print("--------------------------------------------------")

	else:
		#"PRE" mode: sweep sequence length across powers of two: sizeL=2^l1
		#depth=64.
		for l1 in range(r1, r2, 1):
			sizeL = pow(2, l1)
			ErrorC = 0.0

			print("--------------------------------------------------")
			ErrorTot=0
			for test in range(num_tests):
            
				# Generate random inputs in [0,1) for the requested (batch, L, D) shape.
				Weight = torch.rand(batch, sizeL, Depth, dtype=torch.float32, device=device)
				BT = torch.rand(batch, sizeL, Depth, dtype=torch.float32, device=device)


				# Optionally shift inputs to introduce negative values depending on the selected mode.
				#   (2*x - 1) maps [0,1) -> [-1,1).
				if mode == "ALL_NEG":
					# Make both inputs's range: [-1,1)
					Weight = (2*Weight)-1
					BT = (2*BT)-1
            
				elif mode == "W_NEG_BT_POS":
					#Weight: [-1,1), BT: [0,1)
					Weight = (2*Weight)-1

				# MESO model output (simulated in-memory dot product / CiM pipeline).
				Output=model(Weight,BT)
				# Reference output computed in floating point (GPU/CPU torch.matmul).
				# Permute BT from (B,L,D) to (B,D,L) to form BT^T for batched matmul.
				OutNew = torch.matmul(Weight, BT.permute(0,2,1))

				#Compute relative L1 error (%) for each batch element.
				ErrorC = (torch.abs(OutNew - Output).sum(dim=(1,2)) * 100) / torch.abs(OutNew).sum(dim=(1,2))
			
				print(f"Test batch: {test}, mode: {mode}, bit-width: {bit_width}, sequence len: {sizeL}, depth: {Depth}, ErrorC: {ErrorC.mean():.4f}%")	

				#Accumulate the per-test error into a running total and compute the average error over all tests for the current sequence length and depth.
				ErrorTot+=ErrorC.mean()

			TotalTests=batch*num_tests
			print("--------------------------------------------------")
			print(f"#Tests: {TotalTests}, simulation mode: {sim_mode}, mode: {mode}, bit-width: {bit_width}, sequence len: {sizeL}, depth: {Depth}, ErrorC: {ErrorTot/num_tests:.4f}%")	
			print("--------------------------------------------------")


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main():
	# Ensure float32 everywhere: the MESO pipeline/reference are validated for FP32.
	torch.set_default_dtype(torch.float32) #DONOT touch this
	run(sizeL,Depth,batch,num_tests,sim_mode,mode,r1,r2,bit_width)


# -----------------------------------------------------------------------------
# Script entry: set seed for reproducibility and profile the execution.
# -----------------------------------------------------------------------------
if __name__ =="__main__":
	torch.manual_seed(0)
	#cProfile.run('main()')
	main()
