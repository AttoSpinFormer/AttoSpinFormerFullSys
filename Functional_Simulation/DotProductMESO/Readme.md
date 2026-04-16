# AttoSpinFormer - **Floating-Point Matrix Multiplication Kernel using MESO(TRDP-MESO)**

**AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture**

## **Overview**

This project provides a custom, low-bitwidth implementation of a Multi-Head Floating-Point Matrix Multiplication (Dot Product) in PyTorch. The core functionality is designed to execute the matrix multiplication using highly optimized, vectorized floating-point arithmetic.

The implementation utilizes a custom torch.autograd.Function to define a floating-point forward pass (using the custom kernel) while leveraging PyTorch's native floating-point operations for the backward pass (standard gradient calculation).

## **File Descriptions**

This module is divided into three interconnected Python files:

### **TRDP\_MESO.py**

This is the main module that exposes the fixed-point matrix multiplication layers to a PyTorch user.

| Class/Function | Description |
| :---- | :---- |
| **FPMatMulAutogradMESO** | A custom torch.autograd.Function that executes the custom queries@Keys.T using MESO-based CiM and the standard floating-point backward pass. |
| **FPMatMulAutogradMESO2** | A custom torch.autograd.Function that executes the custom softmax@values using MESO-based CiM and the standard floating-point backward pass. |
| **FPMatMulMESO4D** | An nn.Module wrapper that calls FPMatMulAutogradMESO, assuming a typical 4D tensor structure (Batch, Heads, Sequence Length, Feature Dimension). |
| **FPMatMulMESO4D2** | An nn.Module wrapper that calls FPMatMulAutogradMESO2, assuming 4D tensor structures (Batch, Heads, Sequence Length, Sequence Length) and (Batch, Heads, Sequence Length, Feature Dimension) |
| **FPLinearMESO** | A conceptual wrapper for a custom MESO-based CiM for linear layer's core weight-input product. |

### **MESOTRDPTop.py**

This file contains the top-level orchestration logic for preparing tensors before they enter the core fixed-point kernel.

| Function | Description |
| :---- | :---- |
| **Top** | Handles preprocessing of the weight and input tensors. This includes: 1\. Asserting correct bit width (BW). 2\. Padding the tensors to align with the MESO crossbar size (e.g., $256 \\times 64$). 3\. Reshaping and permuting the tensors into the specialized block structure required by the vectorized kernel. |

### **Process\_TR\_MESO.py**

This file implements the highly optimized, vectorized fixed-point multiplication logic.

| Function | Description |
| :---- | :---- |
| **VectorizedProg\_torch** | The core floating-point kernel. |

## **Dependencies**

This project requires:

* **PyTorch (torch)**  
* **Python 3+**

*Note: The code also imports custom modules like ShiftsConv\_batch and converge4 which are present in the file ../GeneralScripts/ShiftsConv_batch.py.*

## **Conceptual Usage**

The primary module is designed to replace a standard torch.matmul or dot product within a low-bitwidth Transformer block, particularly in the Multi-Head Attention mechanism.

import torch  
import torch.nn as nn  
from DotProductMESO.TRDP\_MESO import FPMatMulMESO4D
from DotProductMESO.TRDP\_MESO import FPMatMulMESO4D2

\# Example dimensions:  
\# B \= Batch Size, H \= Heads, L \= Sequence Length, D \= Feature Dimension  
B, H, L, D \= 4, 8, 128, 64 

\# 1\. Instantiate the Fixed-Point MatMul layer  
\# Set the desired bit-width (e.g., 8-bit fixed point)  
q\_kt \= FPMatMulMESO4D(bit\_width=8) 
sm\_va \= FPMatMulMESO4D2(bit\_width=8) 

\# 2\. Prepare sample tensors (Queries and Keys in a typical Attention setup)  
Queries \= torch.randn(B, H, L, D)   
Keys \= torch.randn(B, H, L, D) 
Values \=torch.randn(B, H, L, D)

\# 3\. Execute the Floating-Point Forward Pass  
\# Output shape will be (B, H, L\_Q, L\_K)  
output\_floating\_point \= q\_kt(Queries, Keys)
attention\_scores \=sm\_va(softmax(output\_floating\_point),Values)

print(f"Input Q/K/Va shape: {Queries.shape}")  
print(f"Floating-Point Queries x Keys.T Output shape: {output\_floating\_point.shape}")

print(f"Floating-Point Softmax x Values Output shape: {attention\_scores.shape}")

\# The backward pass will use the standard PyTorch implementation  
loss \= output\_floating\_point.sum()  + attention\_scores.sum()  
loss.backward()

print("Backward pass completed using standard gradients.")  

---

## Project Context

This repository contains code and simulation files associated with the manuscript titled:

**AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture**
