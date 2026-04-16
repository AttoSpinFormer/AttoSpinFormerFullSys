# AttoSpinFormer- MESO Dot-Product Check

**AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture**


`MESODotProductCheck.py` is a verification script for the MESO in-memory computing (IMC) dot-product pipeline.

It compares the output of the MESO simulated matrix multiplication against a standard floating-point reference computed with `torch.matmul`, and reports the relative output error. This is intended as a functional and numerical fidelity check for the MESO dot-product path.

---

## What it does

The script:

- generates synthetic random input tensors
- runs the MESO dot-product implementation
- runs a floating-point reference dot-product
- computes relative L1 error between the two outputs
- prints per-test and averaged error statistics

The tested operation is:

```text
Output = Weight @ BT^T
```

with tensor shapes:

- `Weight`: `(batch, sequence_length, depth)`
- `BT`: `(batch, sequence_length, depth)`
- `Output`: `(batch, sequence_length, sequence_length)`

---

## Running the script

Run directly from the terminal:

```bash
python3 MESODotProductCheck.py
```

The script will prompt for:

- **bit width** (`2-32`)
- **variability** (`0-1`)
- **cycle resolution** (`1-7`)
- **matrix sign mode**
- **simulation mode**:
  - user-defined sequence length / depth
  - predefined sweep over sequence lengths

---

## Matrix input modes

The script supports three random input distributions:

1. **ALL_NEG**  
   Input1 and Input2 both in `[-1, 1)`

2. **W_NEG_BT_POS**  
   Input1 in `[-1, 1)`, Input2 in `[0, 1)`

3. **ALL_POS**  
   Input1 and Input2 both in `[0, 1)`

---

## Simulation modes

### User mode
Runs a single user-defined configuration:

- custom sequence length
- custom depth

### Predefined mode
Runs a sweep over sequence lengths:

- depth fixed at `64`
- sequence length swept from `64` to `2048` as powers of two

---

## Error metric

The reported metric is **relative L1 error (%)** between the MESO output and the floating-point reference:

```text
100 * |reference - meso| / |reference|
```

summed over output elements and averaged across tests.

---

## Notes

- The script uses synthetic random inputs only.
- Large sequence lengths or depths can exhaust memory.
- The import path assumes the repository layout is unchanged.

---

## Project Context

This repository contains code and simulation files associated with the manuscript titled:

**AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture**
