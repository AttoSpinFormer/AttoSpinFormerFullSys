# AttoSpinFormer

**AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture**

## Overview

This repository contains the simulation and evaluation framework for **AttoSpinFormer**, a compute-in-memory transformer architecture based on **magneto-electric spin-orbit logic**. The project brings together device-level modeling, circuit-level pulse-generation support, functional building blocks, and higher-level evaluation scripts used to study the architecture and its components.

At a high level, the repository is organized so that different folders correspond to different abstraction levels:

- **device/circuit-level modeling**
- **functional simulation of architectural building blocks**
- **evaluation/test scripts**
- **training-related simulation workflows**

Each major folder contains its own README with more detailed instructions for running the corresponding scripts and simulations.

---

## Repository Structure

### `Functional_Simulation/`

This folder contains the main functional simulation flow for key computational blocks used in the AttoSpinFormer architecture.

It includes:

- `DotProductMESO/`  
  Scripts related to the MESO-based dot-product module.

- `GeneralScripts/`  
  General-purpose support scripts, including the **FracConv** and **ShiftsConv** scripts.

This folder is a good starting point for readers interested in the functional behavior of the building blocks that support the overall architecture.

---

### `TestScripts/`

This folder contains evaluation-oriented scripts used to test and assess MESO-related behavior at a higher level.

It includes the **MESO evaluation script** and serves as a top-level location for test-oriented workflows associated with the project.

This folder sits alongside other project folders such as the transformer training simulations and the GPT2-XL evaluation scripts.

---

### `MESO_VerilogA_DACFreeSys/`

This folder contains the lower-level simulation setup for MESO device and circuit support.

It includes:

- **MESO Verilog-A models and Xyce simulation files**
- **DAC-free pulse generation system files**

These files support device-level or circuit-level analysis and are separate from the higher-level functional simulation scripts. Refer to the folder-specific README for simulator requirements, model dependencies, and usage details.

---

### Transformer training simulation folder

This folder contains the **transformer training simulations** used in the broader AttoSpinFormer workflow. 

Refer to the local README in that folder for details on setup, dependencies, and execution.

---

### GPT2_XL scripts folder

This folder contains the **GPT2-XL-related scripts** used within the project. It is maintained separately from the training and device-level simulation flows.

Refer to the local README in that folder for specific script descriptions and usage instructions.

---

## How to Navigate the Repository

A simple way to approach the repository is:

1. Start with this top-level README to understand the project structure.
2. Go to the folder most relevant to your goal:
   - For **device/circuit simulations**, use `MESO_VerilogA_DACFreeSys/`
   - For **functional module simulations**, use `Functional_Simulation/`
   - For **evaluation/test workflows**, use `Functional_Simulation/TestScripts/`
   - For **training-related studies**, use the transformer training simulation folder or GPT2-XL folder within the TestScripts folder
3. README inside that folder provides detailed instructions about scripts, dependencies, and expected outputs.


---

## Notes

- The repository contains multiple simulation flows at different abstraction levels.
- Folder-specific READMEs provide the detailed instructions for individual scripts.
- This top-level README is intended only as a roadmap to help users understand the overall organization of the project.

---

## Project Context

This repository contains code and simulation files associated with the manuscript titled:

**AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture**