# Transformer Training + MESO Attention README

This repository section is used for **Transformer training** on a German-to-English machine translation task, with the option to run the attention-layer dot products in either:

- **CMOS / GPU reference mode** (`mode = 0`)
- **MESO IMC mode** (`mode = 1`)

The training stack is built around an **encoder-decoder Transformer** trained on the **Multi30k** translation dataset, with evaluation via **validation/test perplexity** and **BLEU score**. The MESO path is used specifically inside the attention mechanism so the training code can compare a conventional reference execution path against an IMC-style attention path under the same training pipeline. 

---

## What this code is for

This code is intended for **Transformer training experiments**, not just inference or benchmarking. In particular, it:

- validates and loads experiment configuration
- loads and tokenizes translation data
- builds or restores vocabularies
- constructs the encoder-decoder Transformer
- optionally resumes from a compatible checkpoint
- trains the model across epochs
- evaluates with loss, perplexity, and BLEU
- saves checkpoints, optimizer state, scheduler state, and vocabularies for later resume or analysis 

The key architectural feature is that the **attention matrix multiplies** can run through a MESO simulation path during training when `mode = 1`, while the rest of the Transformer remains standard PyTorch.

---

## Quick start

Run training:

```bash
python TransformerTop.py
```

This launches the full pipeline:

1. validate `Config.py`
2. set up vocabularies, model, optimizer, schedulers, and iterators
3. resume from checkpoint if configured
4. train for the requested number of epochs
5. evaluate on validation during training
6. save best checkpoints
7. run final test-set evaluation at the end filecite turn2file4 filecite turn2file3 filecite turn2file1

---

## Training modes

The training flow supports two execution modes from `Config.py`:

- `mode = 0`: **Typical CMOS execution / GPU reference**
- `mode = 1`: **MESO IMC execution**

In MESO mode, the model still trains as a normal encoder-decoder Transformer, but the attention-layer dot products are routed through MESO kernels. `bit_width`, `cycle_res`, and `var` control quantization / temporal resolution / device variability for that path. filecite turn2file2 filecite turn2file4 filecite turn2file12

---

## Files used by this training flow

Only the files below are part of this Transformer training pipeline.

### `TransformerTop.py`

This is the **main training entrypoint**.

It:

- imports the validated configuration
- calls the centralized setup pipeline
- defines the training loop
- defines validation/test inference and BLEU evaluation
- periodically prints sample translations
- saves checkpoints when validation loss improves
- runs final test-set evaluation after training completes filecite turn2file4

Important behaviors in this script:

- training loss is computed with teacher forcing over `trg[:, :-1]` vs `trg[:, 1:]`
- gradient clipping is applied
- a Transformer-style warmup scheduler is stepped during early training
- validation uses autoregressive decoding for sequence generation
- BLEU is computed on detokenized generated sentences
- checkpoint cleanup keeps only the best N checkpoints over time filecite turn2file4 filecite turn2file8 filecite turn2file0

This file is the top-level script you run for paper experiments involving **Transformer training with MESO-aware attention**. filecite turn2file4

---

### `Config.py`

This is the **central experiment configuration** file.

It defines:

- device selection
- model size parameters
- optimization parameters
- training length
- regularization parameters
- resume behavior
- MESO execution controls

Key parameters include:

- `batch_size`
- `max_len`
- `d_model`
- `n_layers`
- `n_heads`
- `ffn_hidden`
- `drop_prob`
- `init_lr`
- `warmup`
- `epoch`
- `clip`
- `label_smoothing`
- `resume_training`
- `strict_config`
- `mode`
- `bit_width`
- `cycle_res`
- `var` filecite turn2file2

For the current uploaded config, training is set up with:

- `d_model = 512`
- `n_layers = 6`
- `n_heads = 8`
- `ffn_hidden = 2048`
- `epoch = 50`
- `mode = 1` (MESO)
- `bit_width = [8, 8]`
- `cycle_res = [4, 4]`
- `var = 0.0` filecite turn2file2

---

### `ConfigSanityCheck.py`

This module validates and auto-corrects configuration values before training starts.

`TransformerTop.py` imports `get_validated_config()` and uses its output instead of trusting raw `Config.py` values directly. The validator checks:

- model dimensions
- optimizer hyperparameters
- dropout / smoothing ranges
- MESO mode validity
- bit width and cycle resolution ranges
- training flags such as `resume_training` and `strict_config` filecite turn2file3

This makes the training pipeline more robust and prevents invalid experiment settings from silently propagating into the run. filecite turn2file3

---

### `ModelSetupManager.py`

This module centralizes the **training setup pipeline**.

Its main function, `setup_model_and_data(config)`, performs the following steps:

1. search for a matching checkpoint
2. load vocabularies from checkpoint if available
3. otherwise build new vocabularies
4. create the Transformer model
5. create optimizer, criterion, and schedulers
6. load checkpoint weights/state if resuming
7. create train/validation/test iterators filecite turn2file1

It also defines `TransformerLRScheduler`, which implements the standard Transformer warmup schedule based on model dimension and warmup steps. filecite turn2file1

This file is effectively the experiment bootstrap layer for **Transformer training**. filecite turn2file1

---

### `CheckPointManager.py`

This module handles checkpoint and training-history management.

It provides utilities for:

- finding the best checkpoint by validation loss
- validating checkpoint config against current config
- loading model / optimizer / scheduler state
- saving full checkpoints with training history and vocabularies
- cleaning up older checkpoints
- printing checkpoint summaries
- reading and writing legacy text history files filecite turn2file0

Notable training-relevant behavior:

- checkpoints store `model_state_dict`, optimizer state, scheduler state, train/val losses, BLEU history, config, and source/target vocabularies
- config compatibility can be enforced via `strict_config`
- checkpoint selection prefers the lowest validation loss among config-compatible checkpoints filecite turn2file0

This is what makes resumed **Transformer training** reproducible and safe across architecture changes. filecite turn2file0

---

### `DatasetLoads.py`

This module defines the **training data pipeline** for the translation task.

It:

- loads SpaCy tokenizers for German and English
- defines torchtext `Field` objects for source and target text
- loads the **Multi30k** dataset
- builds vocabularies or restores them from checkpoint
- creates padded `BucketIterator` objects for train/validation/test
- exports vocabulary metadata needed by the model setup path filecite turn2file10

This file is central to the **Transformer training** experiments because it controls tokenization, vocabulary generation, and iterator construction for the German→English task. filecite turn2file10

---

### `TransformerBasic.py`

This file defines the top-level **encoder-decoder Transformer** module.

It:

- stores source/target padding indices and SOS index
- instantiates the encoder and decoder stacks
- builds source padding masks
- builds target causal + padding masks
- runs the encoder and decoder forward passes filecite turn2file6

This is the core model container used during training. filecite turn2file6

---

### `EncDecLayer.py`

This file defines the internal Transformer building blocks:

- `EncoderLayer`
- `DecoderLayer`
- `Encoder`
- `Decoder`

It manages:

- self-attention
- cross-attention in the decoder
- residual connections
- dropout
- layer normalization
- feed-forward blocks
- token/position embeddings for encoder and decoder inputs filecite turn2file5

Because these layers instantiate `MultiHeadAttention(..., mode, bit_width, cycle_res, var)`, this is one of the places where MESO-aware execution is threaded into the **training-time Transformer**. filecite turn2file5 filecite turn2file12

---

### `Attention.py`

This file implements the training-time **multi-head attention** layer and is the key switch between CMOS and MESO execution.

Important structure:

- linear projections for Q, K, V
- head split / concat logic
- scale-dot-product attention
- output projection

The critical mode switch is in `ScaleDotProductAttention.forward(...)`:

- if `mode == 1`, it uses MESO kernels for:
  - `Q × Kᵀ`
  - `softmax(scores) × V`
- otherwise it uses standard PyTorch matmul (`@`) filecite turn2file12

So for **Transformer training**, this file is where the attention datapath changes between reference execution and MESO execution, while the surrounding training loop remains unchanged. filecite turn2file12

---

### `Embeddings.py`

This module defines the embedding stack used by the encoder and decoder:

- token embeddings
- sinusoidal positional encoding
- dropout over the combined embedding representation filecite turn2file9

This is standard Transformer embedding logic and is used at training and inference time inside both encoder and decoder stacks. filecite turn2file9

---

### `LayerNorm.py`

This module implements custom **layer normalization** over the last feature dimension, using learnable scale and bias parameters. It is used inside the encoder and decoder layers around attention and feed-forward sublayers. filecite turn2file7

---

### `bleu.py`

This module contains BLEU-related helper functions used during evaluation.

In practice, `TransformerTop.py` imports:

- `idx_to_word`
- `get_bleu`

The script also uses `sacrebleu` directly for corpus BLEU during inference evaluation, but `bleu.py` still provides the vocabulary-index to text conversion utilities used for readable sequence reconstruction and debugging. filecite turn2file8 filecite turn2file4

---

## Files not part of this training flow

### `DatasetClass.py`

This file implements an ImageNet dataset wrapper and is **not part of the Transformer training pipeline described here**. It is unrelated to the German-English translation training flow driven by `TransformerTop.py`. filecite turn2file11

---

## Dependency flow

```text
TransformerTop.py
├── ConfigSanityCheck.py
│    └── Config.py
├── ModelSetupManager.py
│    ├── DatasetLoads.py
│    │    └── Config.py
│    ├── TransformerBasic.py
│    │    └── EncDecLayer.py
│    │         ├── Attention.py
│    │         ├── LayerNorm.py
│    │         └── Embeddings.py
│    └── CheckPointManager.py
├── bleu.py
└── sacrebleu (external package)
```

This is the main dependency path for the **Transformer training** simulation shown in the uploaded files. filecite turn2file4 filecite turn2file1 filecite turn2file10 filecite turn2file6 filecite turn2file5 filecite turn2file12

---

## Training pipeline in detail

### 1. Configuration validation

Training starts by calling `get_validated_config()` from `ConfigSanityCheck.py`. This validates and, if needed, corrects parameter values before they are used. filecite turn2file3 filecite turn2file4

### 2. Setup and resume logic

`setup_model_and_data(config)` then:

- searches for config-compatible checkpoints
- restores vocabularies if available
- builds the Transformer model
- creates optimizer and schedulers
- loads checkpoint weights/state if resuming
- builds iterators for train/valid/test filecite turn2file1 filecite turn2file0

### 3. Training loop

For each batch:

- source and target tensors are loaded
- the model is run as `model(src, trg[:, :-1])`
- loss is computed against `trg[:, 1:]`
- gradients are backpropagated
- gradients are clipped
- optimizer is stepped
- warmup LR schedule is advanced during early training filecite turn2file4

### 4. Validation / inference

Validation uses autoregressive generation:

- initialize each target with `<sos>`
- iteratively decode one token at a time
- compute validation loss
- convert predicted indices back to text
- compute BLEU over generated translations filecite turn2file4 filecite turn2file8

### 5. Checkpointing

When validation loss improves:

- full training state is saved
- current config is saved
- source and target vocabularies are saved
- training history is updated
- periodic cleanup keeps only the best checkpoints filecite turn2file4 filecite turn2file0

---

## MESO usage during training

This repository is not only for Transformer architecture definition; it is explicitly used for **Transformer training with a switchable attention backend**.

In the training-time attention implementation:

- `mode = 0` uses regular matrix multiplication for attention scores and value aggregation
- `mode = 1` uses MESO kernels for the same two operations filecite turn2file12

That means the model can be trained and evaluated under the same task/dataset/setup, while changing only the attention dot-product implementation. This makes it suitable for paper experiments comparing:

- reference training behavior
- MESO-attention training behavior
- effects of bit width, cycle resolution, and variability on translation quality and optimization behavior filecite turn2file12 filecite turn2file2

---

## Metrics reported by the training stack

The training/evaluation code reports:

- **training loss**
- **validation loss**
- **training perplexity** (`exp(train_loss)`)
- **validation perplexity** (`exp(valid_loss)`)
- **BLEU score** on validation and test sets filecite turn2file4

These are the main paper-facing metrics for this training flow. filecite turn2file4

---

## Installation / environment notes

Based on the imported code, you will need at least:

- Python 3
- `torch`
- `torchtext`
- `spacy`
- `sacrebleu`

You will also need SpaCy language models:

- `en_core_web_sm`
- `de_core_news_sm` filecite turn2file10

In addition, the attention module expects the MESO dot-product packages to be importable through the local project layout, including `DotProductMESO.*` modules. filecite turn2file12

---

## Summary

This codebase is used for **Transformer training** on a machine-translation task with a configurable attention backend.

- `TransformerTop.py` runs the end-to-end training experiment
- `Config.py` and `ConfigSanityCheck.py` control experiment settings safely
- `ModelSetupManager.py` prepares the model, vocabularies, iterators, optimizer, and resume state
- `TransformerBasic.py`, `EncDecLayer.py`, `Attention.py`, `Embeddings.py`, and `LayerNorm.py` define the trainable Transformer model
- `DatasetLoads.py` provides the translation dataset pipeline
- `CheckPointManager.py` manages reproducible checkpointing
- `Attention.py` is the key switch that enables MESO-backed attention during training filecite turn2file4 filecite turn2file1 filecite turn2file12

For paper evaluation, this makes the stack suitable for controlled comparisons between standard Transformer training and Transformer training with MESO-mapped attention operations. filecite turn2file4 filecite turn2file12
