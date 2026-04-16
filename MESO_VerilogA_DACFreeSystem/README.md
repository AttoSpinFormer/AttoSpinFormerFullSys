# AttoSpinFormer - Spice Simulations

**AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture**

## Overview

This directory contains two separate simulation flows:

1. **MESO / CMOS-MESO primitive simulation** using **Xyce** and **Verilog-A**
2. **DAC-free pulse generation simulation** using **LTspice**

These flows should be treated as distinct setups because they use different simulators, model files, and supporting libraries.

---

## 1. MESO / CMOS-MESO Primitive Simulation

The MESO / CMOS-MESO primitive can be tested using the provided Xyce netlists together with the supplied Verilog-A models.

### Files

- `MESO_Characterization_Transient.sp`  
  Primary transient testbench for the CMOS-MESO primitive. This file is intended to be used with the **MESO2YN** Verilog-A model.

- `MESO2YN_xyce_WORKING.va`  
  Current Verilog-A model used for transient characterization and functional simulation.

- `MESO2Y_xyce_scaled_leakic.va`  
  Physics-based Verilog-A model containing the **LLG equations** and other device equations that more closely match the intended physical behavior.

- `TestingNewMESOModel.sp`  
  Alternate standalone Xyce testbench for simulating the MESO device.

### Recommended Usage

#### Functional / working transient test

Use:

- `MESO_Characterization_Transient.sp`
- `MESO2YN_xyce_WORKING.va`

This setup is recommended for validating the simulation flow and testing the CMOS-MESO primitive using the currently working model.

#### Physics-based device simulation

Use either `.sp` testbench together with:

- `MESO2Y_xyce_scaled_leakic.va`

This Verilog-A file includes the **LLG equations** along with spin–orbit coupling and spin-to-charge conversion mechanisms. It further incorporates equivalent circuit components, including the magnetoelectric capacitor, interconnect parasitics, and spin injection and readout paths.

### Notes

- `MESO_Characterization_Transient.sp` is the preferred starting point for testing the CMOS-MESO primitive with the current Verilog-A model.
- `MESO2Y_xyce_scaled_leakic.va` is the more physics-complete model.
- Either `.sp` file may be used to simulate the device as a standalone block, depending on the desired setup.

### Suggested Starting Point

If the goal is simply to run a stable MESO simulation, start with:

- `MESO_Characterization_Transient.sp`
- `MESO2YN_xyce_WORKING.va`

If the goal is to use the model that best reflects the device physics, use:

- `MESO2Y_xyce_scaled_leakic.va`

---

## 2. DAC-Free Pulse Generation Simulation

This directory also includes an Spice simulation for a DAC-free pulse generation scheme implemented in 45 nm CMOS.

### File

- `DACFreePulseSystem_45nm_SpiceSim.sp`  
  Spice transistor-level netlist for simulating the DAC-free pulse generation circuit.

### Purpose

This simulation is intended to evaluate the DAC-free pulse generation scheme independently of the MESO Verilog-A/Xyce device flow. It may be useful for studying pulse generation, timing behavior, or driver/interface circuitry that could be used with a MESO-based system.

### Simulator

- **LTspice/HSpice**

### Library / Model Dependencies

The Spice netlist includes the following external files:

- `standard.mos`
- `UniversalOpAmp1.lib`
- `LTC.lib`

Based on the present netlist structure:

- `standard.mos` contains the required MOS transistor models.
- `UniversalOpAmp1.lib` is required for the instantiated op-amp block.

### Notes

- This LTspice simulation is separate from the MESO Xyce/Verilog-A flow.
- It should be documented and run as a standalone circuit simulation.

### Key Signal Nodes

The following node voltages are useful for interpreting the DAC-free pulse generation simulation:

- `V(n011)`  
  Output voltage of the DAC-free pulse generation circuit.

- `V(n007)`  
  Program control signal that enables the DAC-free pulse application system.

- `V(n009)`  
  Bit-value input to be stored in the MESO device. This signal controls the direction of the output voltage.


---

## 3. Workflow Summary

### For MESO device simulation

Use:
- Xyce
- One of the `.sp` testbenches
- One of the Verilog-A model files

Recommended first run:
- `MESO_Characterization_Transient.sp`
- `MESO2YN_xyce_WORKING.va`

### For DAC-free pulse generation simulation

Use:
- LTspice/HSpice
- `DACFreePulseSystem_45nm_SpiceSim.sp`
- Required LTspice/HSpice model/library includes

---

## 4. Important Distinction

Although both sets of files may relate to the broader MESO system, they represent **different levels of simulation**:

- The **MESO files** focus on the device / primitive model and its behavior.
- The **DAC-free pulse file** focuses on the transistor-level pulse-generation circuitry used at the interface.

---

## Project Context

This repository contains code and simulation files associated with the manuscript titled:

**AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture**