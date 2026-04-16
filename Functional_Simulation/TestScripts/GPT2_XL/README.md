# AttoSpinFormer- GPT-2 XL MESO Benchmarking

**AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture**

Run **pretrained GPT-2 XL** with either:

- **GPU mode**: standard Hugging Face / PyTorch attention
- **MESO mode**: GPT-2 XL attention patched so the attention-layer matrix multiplies run through the MESO simulation path

This setup is meant to answer a focused question:

> What happens to GPT-2 XL benchmark quality (inference) when the attention dot-product path is replaced by CMOS-MESO-based Compute-in-memory compute blocks?

---

## Quick start

### Core benchmark suite: PPL + HellaSwag + PIQA

Run both GPU and MESO:
```bash
python GPT2XL_Benchmarks.py --mode both
```

Run MESO only:
```bash
python GPT2XL_Benchmarks.py --mode meso
```

Run GPU only:
```bash
python GPT2XL_Benchmarks.py --mode gpu
```

Run only perplexity in MESO mode:
```bash
python GPT2XL_Benchmarks.py --mode meso --skip-hellaswag --skip-piqa
```

Quick smoke test:
```bash
python GPT2XL_Benchmarks.py --mode meso --max-samples 32
```

Save results to JSON:
```bash
python GPT2XL_Benchmarks.py --mode both --output-json results.json
```

This script supports `gpu`, `meso`, and `both` modes and can selectively skip PPL, HellaSwag, or PIQA.

---

### Additional benchmark suite: LAMBADA + ARC-Challenge

Run both GPU and MESO:
```bash
python GPT2XL_LAMBADA_ARC_Benchmarks.py --mode both
```

Run MESO only:
```bash
python GPT2XL_LAMBADA_ARC_Benchmarks.py --mode meso
```

Run GPU only:
```bash
python GPT2XL_LAMBADA_ARC_Benchmarks.py --mode gpu
```

Run only LAMBADA:
```bash
python GPT2XL_LAMBADA_ARC_Benchmarks.py --mode meso --skip-arc
```

Run only ARC-Challenge:
```bash
python GPT2XL_LAMBADA_ARC_Benchmarks.py --mode meso --skip-lambada
```

Quick smoke test:
```bash
python GPT2XL_LAMBADA_ARC_Benchmarks.py --mode meso --max-samples 32
```

Save results to JSON:
```bash
python GPT2XL_LAMBADA_ARC_Benchmarks.py --mode both --output-json lambada_arc_results.json
```

This second benchmark script is intentionally separate so you can add LAMBADA and ARC-Challenge without changing the original PPL/HellaSwag/PIQA path.

---

## What this sim does

The code loads **Hugging Face GPT-2 XL** and keeps the model structure intact, but in **MESO mode** it monkey-patches each GPT-2 attention block so that:

- **Q × Kᵀ** runs through a MESO kernel
- **softmax(scores) × V** runs through a MESO kernel

Masking and softmax still run in normal PyTorch. Embeddings, MLPs, layer norms, residuals, and LM head stay standard Hugging Face GPT-2 XL.

So this sim changes **only the attention matmul path**, not the full transformer implementation.

---

## Files used by this simulation

Only these files are required for the GPT-2 XL MESO benchmark flow shown here.

### `GPT2XL_Benchmarks.py`
Main benchmark driver for:

- **Perplexity (PPL)**
- **HellaSwag**
- **PIQA**

It:
- parses CLI arguments
- loads GPT-2 XL in `gpu`, `meso`, or `both` mode
- runs the selected benchmarks
- prints a summary
- optionally writes JSON results

For perplexity, it uses sliding-window language modeling evaluation. For HellaSwag and PIQA, it scores answer choices with conditional log-likelihood.

---

### `GPT2XL_LAMBADA_ARC_Benchmarks.py`
Additional benchmark driver for:

- **LAMBADA last-token exact-match accuracy**
- **ARC-Challenge multiple-choice accuracy**

It:
- parses CLI arguments
- loads GPT-2 XL in `gpu`, `meso`, or `both` mode
- runs LAMBADA and/or ARC-Challenge
- prints a summary
- optionally writes JSON results

LAMBADA evaluation:
- tokenizes each example
- uses the full sequence except the last token as context
- predicts the final token
- checks exact match with the gold last token

ARC-Challenge evaluation:
- builds a prompt from the question stem
- scores each answer option by conditional log-likelihood
- normalizes by token count
- chooses the highest-scoring option
- compares with the gold answer key

The LAMBADA script also includes dataset fallback behavior. If no override is supplied, it tries a small list of candidate LAMBADA datasets until one loads successfully.

---

### `GPT2XL_HF_MESO.py`
Loads the pretrained GPT-2 XL model and patches the attention path in MESO mode. It:

- downloads GPT-2 XL weights and tokenizer from Hugging Face
- optionally patches every `GPT2Attention` block
- provides helper routines for generation and optional dataset evaluation

In MESO mode, each attention block is modified so that:

- `Q × Kᵀ` uses `FPMatMulMESO4DTimePlex`
- `softmax(attention) × V` uses `FPMatMulMESO4D2TimePlex`

---

### `GPT2XL_Config.py`
Central config for GPT-2 XL runs. It defines:

- device
- dtype
- model name
- prompt / generation settings
- MESO settings
- optional small dataset evaluation settings
- Hugging Face cache directory

Important defaults include:

- `MODEL_NAME = "openai-community/gpt2-xl"`
- `MODE = 1` for MESO
- `BIT_WIDTH = [7, 7]` (Mantissa bit-width for the Q@KT and softmax@Value modules)
- `CYCLE_RES = [4, 4]` (Cycle resolution for the Q@KT and softmax@Value modules)
- `VAR = 0.0` (Variability used for MESO devices for both modules)
- `USE_CACHE = False`

---

### `AttMESOTimePlex.py`
Defines the two MESO attention matmul modules used by the GPT-2 attention patch:

- `FPMatMulMESO4DTimePlex` for **Queries × Keysᵀ**
- `FPMatMulMESO4D2TimePlex` for **Softmax × Values**

This file is the bridge between GPT-2 attention tensors and the MESO compute pipeline. It:

- quantizes tensors with `Shifts_torch3`
- merges the compensation terms with `converge4`
- calls the top-level MESO datapath via `Top(...)`

---

### `MESOTRDPTopTimePlex.py`
Top-level tensor preparation and orchestration for the MESO dot-product flow. It:

- repacks tensors into array-friendly blocks
- pads tensors to hardware-style dimensions
- applies input-resolution packing
- injects variation when enabled
- calls the core compute kernel `VectorizedProg_torch(...)`

This is where the simulation converts model tensors into the MESO execution layout.

---

### `AttProcessTimePlex.py`
Core MESO execution kernel. It performs the simulated dot-product pipeline, including:

- analog-style accumulation flow
- bit-slice and cycle reconstruction
- compensation-term reconstruction
- return of the final floating-point output tensor

This is the lowest-level compute stage in the MESO path.

---

### `ShiftsConv_batch.py`
Batch / multi-head quantization prep used by the attention MESO kernels. It provides:

- `Shifts_torch3(...)` for sign-aware quantization
- `converge4(...)` for appending minimum-value compensation tensors to multi-head inputs

This is called directly from `AttMESOTimePlex.py`.

---

### `FracConv_batch.py`
Batch / multi-head fixed-point conversion utilities used by `ShiftsConv_batch.py`. It provides:

- `ShiftCalc(...)`
- `fractions(...)`
- `vectorized_fractions(...)`
- `ShiftCalc_torch_vectorized(...)`

These helpers generate the fixed-point bit representations expected by the MESO simulation path.

---

## File dependency flow

```text
Benchmark entrypoints
├── GPT2XL_Benchmarks.py
└── GPT2XL_LAMBADA_ARC_Benchmarks.py
     ├── GPT2XL_HF_MESO.py
     │    ├── GPT2XL_Config.py
     │    └── AttMESOTimePlex.py
     │         ├── MESOTRDPTopTimePlex.py
     │         │    └── AttProcessTimePlex.py
     │         └── ShiftsConv_batch.py
     │              └── FracConv_batch.py
     └── GPT2XL_Config.py
```

Both benchmark entrypoints share the same GPT-2 XL loader, attention patch, config, and MESO compute stack.

---

## Attention flow in MESO mode

Inside each patched GPT-2 attention block, the flow is:

1. compute `Q × Kᵀ` with MESO
2. divide by `sqrt(d_k)`
3. apply causal mask
4. apply attention mask if present
5. apply `softmax`
6. compute `softmax(scores) × V` with MESO

So both core attention matrix multiplies are routed through MESO when `mode=meso`.

---

## Benchmarks

### `GPT2XL_Benchmarks.py`

#### Perplexity
Default PPL settings are:

- dataset: `wikitext`
- config: `wikitext-2-raw-v1`
- split: `test`
- text field: `text`
- max length: `1024`
- stride: `512`

The script:
- loads the dataset
- removes empty text rows
- concatenates text
- tokenizes
- evaluates with a sliding window
- accumulates negative log-likelihood
- returns perplexity as `exp(total_nll / total_tokens)`

#### HellaSwag / PIQA
For each example, the script:
- constructs a prompt
- scores each answer choice with conditional log-probability
- normalizes by answer length
- picks the best-scoring choice
- computes accuracy

---

### `GPT2XL_LAMBADA_ARC_Benchmarks.py`

#### LAMBADA
Default behavior:
- uses the user override dataset if provided
- otherwise tries several built-in LAMBADA dataset candidates
- uses the specified split and text field
- predicts the final token from the preceding context

Metric:
- **last-token exact-match accuracy**

#### ARC-Challenge
Default settings:
- dataset: `allenai/ai2_arc`
- config: `ARC-Challenge`
- split: `validation`

Metric:
- **multiple-choice accuracy** via conditional log-likelihood over answer options

---

## Standalone sanity check

You can also run the model/patch script directly:

```bash
python GPT2XL_HF_MESO.py
```

This:
- loads GPT-2 XL according to `GPT2XL_Config.py`
- prints the model structure
- generates text from the configured prompt
- optionally runs a small dataset eval if enabled in config

---

## Installation

You need at least:

- Python 3
- `torch`
- `transformers`
- `datasets`
- `huggingface_hub`
- `tqdm`

### `install_gpt2_xl_deps.sh`
Helper install script for the GPT-2 XL benchmark environment.

This script:
- upgrades `pip`
- installs the core Python dependencies required by the GPT-2 XL benchmarking flow:
  - `transformers`
  - `datasets`
  - `huggingface_hub`
  - `sentencepiece`
  - `tqdm` :contentReference[oaicite:0]{index=0}

Example usage:

```bash
bash install_gpt2_xl_deps.sh
This is a convenience script for setting up the Hugging Face / dataset stack before running the benchmark scripts. 

Install with:

```bash
pip install torch transformers datasets huggingface_hub tqdm
```

You also need the local module layout expected by the imports in the scripts, including:

- `DotProductMESO.AttMESOTimePlex`
- `DotProductMESO.MESOTRDPTopTimePlex`
- `DotProductMESO.AttProcessTimePlex`
- `GeneralScripts.ShiftsConv_batch`
- `GeneralScripts.FracConv_batch`

---

## Notes

- **Only attention matmuls are replaced in MESO mode.** The rest of GPT-2 XL remains standard Hugging Face code.
- `USE_CACHE = False` is the intended setting for the patched attention path.
- `GPT2XL_Benchmarks.py` accepts `--cache-dir` for dataset loading, but model/tokenizer loading in `GPT2XL_HF_MESO.py` uses `CACHE_DIR` from `GPT2XL_Config.py`, so model-cache and dataset-cache handling are not fully unified.
- `GPT2XL_LAMBADA_ARC_Benchmarks.py` follows the same shared model-loading path, so the same MESO patching behavior applies there too.

---

## Summary

This simulation benchmarks **pretrained GPT-2 XL** while swapping the attention-layer dot-product path between:

- standard PyTorch / GPU attention
- MESO-based IMC attention simulation

`GPT2XL_Benchmarks.py` covers PPL, HellaSwag, and PIQA.  
`GPT2XL_LAMBADA_ARC_Benchmarks.py` adds LAMBADA and ARC-Challenge.  
Both scripts share the same GPT-2 XL loader, attention patch, config, and MESO compute path.

---

## Project Context

This repository contains code and simulation files associated with the manuscript titled:

**AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture**
