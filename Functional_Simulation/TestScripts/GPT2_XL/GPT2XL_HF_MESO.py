#!/usr/bin/env python3
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture
"""Run pretrained GPT-2 XL through the existing MESO attention kernels.

What this script does:
1. Downloads GPT-2 XL weights + tokenizer from Hugging Face.
2. Optionally downloads a small evaluation text split.
3. Monkey-patches each GPT-2 attention block so QK^T and softmax@V can flow
   through the existing MESO simulation modules from AttMESOTimePlex.py.
4. Runs generation and optionally computes perplexity on a small dataset.

"""

import sys
import math
import types
from dataclasses import dataclass

sys.path.append('../../')

import torch
import torch.nn as nn

from GPT2XL_Config import (
	DEVICE,
	DTYPE,
	MODEL_NAME,
	PROMPT,
	MAX_NEW_TOKENS,
	TEMPERATURE,
	TOP_K,
	DO_SAMPLE,
	USE_CACHE,
	MODE,
	BIT_WIDTH,
	CYCLE_RES,
	VAR,
	RUN_DATASET_EVAL,
	DATASET_NAME,
	DATASET_CONFIG,
	DATASET_SPLIT,
	BLOCK_SIZE,
	CACHE_DIR,
)

from DotProductMESO.AttMESOTimePlex import FPMatMulMESO4DTimePlex, FPMatMulMESO4D2TimePlex


@dataclass
class RunStats:
	prompt_tokens: int
	generated_tokens: int
	total_tokens: int


def _prepare_attention_mask_for_scores(attention_mask: torch.Tensor, query_length: int, key_length: int) -> torch.Tensor:
	"""Convert [batch, seq] mask into broadcastable [batch, 1, q, k] bool mask."""
	if attention_mask.dim() != 2:
		raise ValueError(f"Expected 2D attention_mask, got {attention_mask.shape}")
	mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
	if query_length != 1:
		mask = mask.expand(-1, 1, query_length, -1)
	return mask[:, :, :, :key_length]


class MESOGPT2AttentionPatch(nn.Module):
	"""Callable helper attached to each GPT-2 attention block."""

	def __init__(self, bit_width: torch.Tensor, cycle_res: torch.Tensor, var: float):
		super().__init__()
		self.qktdp = FPMatMulMESO4DTimePlex(
			bit_width=int(bit_width[0].item()),
			cycle_res=int(cycle_res[0].item()),
			var=float(var),
		)
		self.smvadp = FPMatMulMESO4D2TimePlex(
			bit_width=int(bit_width[1].item()),
			cycle_res=int(cycle_res[1].item()),
			var=float(var),
		)

	def forward(self, module, query, key, value, attention_mask=None, head_mask=None):
		d_k = query.size(-1)
		attn_weights = self.qktdp(query, key) / math.sqrt(d_k)

		q_len = query.size(-2)
		k_len = key.size(-2)

		# Causal mask from the GPT-2 module buffer when available.
		if hasattr(module, "bias") and module.bias is not None:
			causal_mask = module.bias[:, :, k_len - q_len : k_len, :k_len].to(torch.bool)
			mask_value = torch.finfo(attn_weights.dtype).min
			attn_weights = torch.where(causal_mask, attn_weights, mask_value)

		if attention_mask is not None:
			if attention_mask.dtype != torch.bool:
				attention_mask = attention_mask.to(dtype=torch.bool)
			mask_value = torch.finfo(attn_weights.dtype).min
			attn_weights = torch.where(attention_mask, attn_weights, mask_value)

		attn_weights = torch.softmax(attn_weights, dim=-1)
		attn_weights = attn_weights.type_as(value)

		if head_mask is not None:
			attn_weights = attn_weights * head_mask

		attn_output = self.smvadp(attn_weights, value)
		return attn_output, attn_weights


def patch_gpt2_attention_with_meso(model: nn.Module, bit_width: torch.Tensor, cycle_res: torch.Tensor, var: float) -> nn.Module:
	"""Patch Hugging Face GPT-2 attention blocks to use MESO matmul kernels."""
	try:
		from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
	except Exception as exc:
		raise RuntimeError(
			"transformers is required. Install it with: pip install transformers datasets huggingface_hub"
		) from exc

	patched = 0

	for module in model.modules():
		if isinstance(module, GPT2Attention):
			helper = MESOGPT2AttentionPatch(bit_width=bit_width, cycle_res=cycle_res, var=var).to(next(model.parameters()).device)
			module.meso_patch = helper

			def _attn(self, query, key, value, attention_mask=None, head_mask=None):
				return self.meso_patch(self, query, key, value, attention_mask=attention_mask, head_mask=head_mask)

			module._attn = types.MethodType(_attn, module)
			patched += 1

	if patched == 0:
		raise RuntimeError("No GPT2Attention modules were patched.")

	print(f"Patched {patched} GPT-2 attention blocks with MESO kernels.")
	return model


def load_model_and_tokenizer(model_name: str, device: str, dtype: torch.dtype, use_meso: bool):
	try:
		from transformers import AutoModelForCausalLM, AutoTokenizer
	except Exception as exc:
		raise RuntimeError(
			"transformers is required. Install it with: pip install transformers datasets huggingface_hub"
		) from exc

	tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR, torch_dtype=dtype)
	model.to(device)
	model.eval()

	if use_meso:
		model = patch_gpt2_attention_with_meso(model, BIT_WIDTH, CYCLE_RES, VAR)

	return model, tokenizer


def run_generation(model, tokenizer, prompt: str) -> RunStats:
	encoded = tokenizer(prompt, return_tensors="pt")
	input_ids = encoded["input_ids"].to(DEVICE)
	attention_mask = encoded["attention_mask"].to(DEVICE)

	with torch.no_grad():
		output_ids = model.generate(
			input_ids=input_ids,
			attention_mask=attention_mask,
			max_new_tokens=MAX_NEW_TOKENS,
			temperature=TEMPERATURE,
			top_k=TOP_K,
			do_sample=DO_SAMPLE,
			use_cache=USE_CACHE,
			pad_token_id=tokenizer.eos_token_id,
		)

	text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
	print("\n=== GENERATED TEXT ===")
	print(text)
	print("======================\n")

	return RunStats(
		prompt_tokens=int(input_ids.shape[1]),
		generated_tokens=int(output_ids.shape[1] - input_ids.shape[1]),
		total_tokens=int(output_ids.shape[1]),
	)


def evaluate_dataset_perplexity(model, tokenizer):
	try:
		from datasets import load_dataset
	except Exception as exc:
		raise RuntimeError(
			"datasets is required for dataset evaluation. Install it with: pip install datasets"
		) from exc

	dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, cache_dir=CACHE_DIR)
	texts = [x["text"] for x in dataset if x.get("text", "").strip()]
	if not texts:
		print("Dataset split is empty after filtering.")
		return None

	enc = tokenizer("\n\n".join(texts), return_tensors="pt", truncation=True, max_length=BLOCK_SIZE)
	input_ids = enc["input_ids"].to(DEVICE)
	attention_mask = enc["attention_mask"].to(DEVICE)

	with torch.no_grad():
		outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
	loss = float(outputs.loss.item())
	ppl = math.exp(loss)
	print(f"Dataset loss: {loss:.4f}")
	print(f"Dataset perplexity: {ppl:.4f}")
	return loss, ppl


def print_gpt2_xl_structure(model):
	cfg = model.config
	print("\n=== GPT-2 XL STRUCTURE ===")
	print(f"Model name: {MODEL_NAME}")
	print(f"Layers (blocks): {cfg.n_layer}")
	print(f"Attention heads: {cfg.n_head}")
	print(f"Hidden size: {cfg.n_embd}")
	print(f"Head dimension: {cfg.n_embd // cfg.n_head}")
	print(f"Context length: {cfg.n_positions}")
	print(f"Vocabulary size: {cfg.vocab_size}")
	print(f"Approx params: ~1.5B")
	print("Per block: LN -> causal self-attn -> residual -> LN -> MLP(4x hidden) -> residual")
	print("Final: token embedding + positional embedding -> 48 blocks -> final LN -> tied LM head")
	print("==========================\n")


def main():
	print(f"Loading {MODEL_NAME} on {DEVICE} (mode={MODE})")
	model, tokenizer = load_model_and_tokenizer(MODEL_NAME, DEVICE, DTYPE, use_meso=(MODE == 1))
	print_gpt2_xl_structure(model)

	stats = run_generation(model, tokenizer, PROMPT)
	print(f"Prompt tokens: {stats.prompt_tokens}")
	print(f"Generated tokens: {stats.generated_tokens}")
	print(f"Total tokens: {stats.total_tokens}")

	if RUN_DATASET_EVAL:
		evaluate_dataset_perplexity(model, tokenizer)


if __name__ == "__main__":
	main()
