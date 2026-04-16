#!/usr/bin/env python3
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture

"""Benchmark GPT-2 XL in GPU vs MESO attention mode.

Metrics:
- Perplexity on a held-out language modeling dataset using sliding-window evaluation.
- HellaSwag accuracy via multiple-choice conditional log-likelihood.
- PIQA accuracy via multiple-choice conditional log-likelihood.

This script reuses the Hugging Face GPT-2 XL loader and the MESO attention patch
from GPT2XL_HF_MESO.py so the code path stays as close as possible between GPU
and MESO runs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from GPT2XL_HF_MESO import load_model_and_tokenizer
from GPT2XL_Config import MODEL_NAME, DEVICE, DTYPE


DEFAULT_PPL_DATASET = "wikitext"
DEFAULT_PPL_CONFIG = "wikitext-2-raw-v1"
DEFAULT_PPL_SPLIT = "test"
DEFAULT_PPL_TEXT_FIELD = "text"

DEFAULT_HELLASWAG_DATASET = "Rowan/hellaswag"
DEFAULT_HELLASWAG_SPLIT = "validation"
DEFAULT_PIQA_DATASET = "ybisk/piqa"
DEFAULT_PIQA_SPLIT = "validation"


@dataclass
class BenchmarkResult:
	mode: str
	perplexity: Optional[float]
	hellaswag_acc: Optional[float]
	piqa_acc: Optional[float]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run GPT-2 XL benchmarks in GPU and/or MESO mode.")
	parser.add_argument("--mode", choices=["gpu", "meso", "both"], default="both")
	parser.add_argument("--model-name", default=MODEL_NAME)
	parser.add_argument("--device", default=DEVICE)
	parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=str(DTYPE).split(".")[-1])
	parser.add_argument("--cache-dir", default=None)
	parser.add_argument("--batch-size", type=int, default=1, help="Batch size for multiple-choice tasks. Keep at 1 for GPT-2 XL unless memory is ample.")
	parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per benchmark for quick smoke tests.")
	parser.add_argument("--skip-ppl", action="store_true")
	parser.add_argument("--skip-hellaswag", action="store_true")
	parser.add_argument("--skip-piqa", action="store_true")
	parser.add_argument("--ppl-dataset", default=DEFAULT_PPL_DATASET)
	parser.add_argument("--ppl-config", default=DEFAULT_PPL_CONFIG)
	parser.add_argument("--ppl-split", default=DEFAULT_PPL_SPLIT)
	parser.add_argument("--ppl-text-field", default=DEFAULT_PPL_TEXT_FIELD)
	parser.add_argument("--ppl-max-length", type=int, default=1024)
	parser.add_argument("--ppl-stride", type=int, default=512)
	parser.add_argument("--hellaswag-dataset", default=DEFAULT_HELLASWAG_DATASET)
	parser.add_argument("--hellaswag-split", default=DEFAULT_HELLASWAG_SPLIT)
	parser.add_argument("--piqa-dataset", default=DEFAULT_PIQA_DATASET)
	parser.add_argument("--piqa-split", default=DEFAULT_PIQA_SPLIT)
	parser.add_argument("--output-json", default=None)
	return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
	mapping = {
		"float32": torch.float32,
		"float16": torch.float16,
		"bfloat16": torch.bfloat16,
	}
	return mapping[name]


def normalize_whitespace(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def preprocess_hellaswag_text(text: str) -> str:
	text = text.replace(" [title]", ". ")
	text = re.sub(r"\[.*?\]", "", text)
	text = text.replace("  ", " ")
	return normalize_whitespace(text)


def build_hellaswag_prompt_and_choices(example: Dict) -> Tuple[str, List[str], int]:
	ctx_a = preprocess_hellaswag_text(example["ctx_a"])
	ctx_b = preprocess_hellaswag_text(example.get("ctx_b", ""))
	if ctx_b:
		ctx_b = ctx_b[0].upper() + ctx_b[1:] if len(ctx_b) > 1 else ctx_b.upper()
	prompt = normalize_whitespace(f"{ctx_a} {ctx_b}")
	endings = [preprocess_hellaswag_text(x) for x in example["endings"]]
	label = int(example["label"])
	return prompt, endings, label


def build_piqa_prompt_and_choices(example: Dict) -> Tuple[str, List[str], int]:
	prompt = normalize_whitespace(f"Question: {example['goal']}\nAnswer:")
	choices = [normalize_whitespace(example["sol1"]), normalize_whitespace(example["sol2"])]
	label = int(example["label"])
	return prompt, choices, label


@torch.no_grad()
def conditional_logprob(model, tokenizer, prompt: str, completion: str, device: str) -> Tuple[float, int]:
	prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
	full_text = prompt + (" " if prompt and not prompt.endswith((" ", "\n")) else "") + completion
	full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

	if full_ids.shape[1] <= prompt_ids.shape[1]:
		raise ValueError("Completion produced no additional tokens; cannot score candidate.")

	outputs = model(full_ids, use_cache=False)
	logits = outputs.logits[:, :-1, :]
	target_ids = full_ids[:, 1:]
	log_probs = F.log_softmax(logits, dim=-1)
	token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

	completion_start = prompt_ids.shape[1] - 1
	completion_token_log_probs = token_log_probs[:, completion_start:]
	total_logprob = float(completion_token_log_probs.sum().item())
	token_count = int(completion_token_log_probs.numel())
	return total_logprob, token_count


@torch.no_grad()
def evaluate_multiple_choice(
	model,
	tokenizer,
	dataset_name: str,
	split: str,
	builder,
	device: str,
	cache_dir: Optional[str] = None,
	max_samples: Optional[int] = None,
) -> float:
	from datasets import load_dataset

	dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
	if max_samples is not None:
		dataset = dataset.select(range(min(max_samples, len(dataset))))

	correct = 0
	total = 0

	for example in tqdm(dataset, desc=f"Evaluating {dataset_name}:{split}"):
		prompt, choices, label = builder(example)
		scores = []
		for choice in choices:
			total_logprob, token_count = conditional_logprob(model, tokenizer, prompt, choice, device)
			avg_logprob = total_logprob / max(token_count, 1)
			scores.append(avg_logprob)
		pred = int(max(range(len(scores)), key=lambda i: scores[i]))
		correct += int(pred == label)
		total += 1

	return correct / max(total, 1)


@torch.no_grad()
def evaluate_perplexity(
	model,
	tokenizer,
	dataset_name: str,
	dataset_config: Optional[str],
	split: str,
	text_field: str,
	device: str,
	max_length: int,
	stride: int,
	cache_dir: Optional[str] = None,
	max_samples: Optional[int] = None,
) -> float:
	from datasets import load_dataset

	if dataset_config:
		dataset = load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir)
	else:
		dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

	if max_samples is not None:
		dataset = dataset.select(range(min(max_samples, len(dataset))))

	texts: List[str] = []
	for row in dataset:
		text = row.get(text_field, "")
		if isinstance(text, str) and text.strip():
			texts.append(text)

	if not texts:
		raise ValueError(f"No non-empty texts found in field '{text_field}'.")

	encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
	input_ids = encodings.input_ids.to(device)
	seq_len = input_ids.size(1)

	nlls: List[torch.Tensor] = []
	prev_end = 0
	for begin in tqdm(range(0, seq_len, stride), desc=f"Evaluating {dataset_name}:{split} perplexity"):
		end = min(begin + max_length, seq_len)
		trg_len = end - prev_end
		input_chunk = input_ids[:, begin:end]
		target_chunk = input_chunk.clone()
		target_chunk[:, :-trg_len] = -100

		outputs = model(input_ids=input_chunk, labels=target_chunk, use_cache=False)
		neg_log_likelihood = outputs.loss * trg_len
		nlls.append(neg_log_likelihood)

		prev_end = end
		if end == seq_len:
			break

	ppl = torch.exp(torch.stack(nlls).sum() / prev_end)
	return float(ppl.item())


def load_for_mode(mode: str, model_name: str, device: str, dtype: torch.dtype):
	use_meso = mode == "meso"
	model, tokenizer = load_model_and_tokenizer(model_name, device, dtype, use_meso=use_meso)
	return model, tokenizer


def run_mode(mode: str, args: argparse.Namespace) -> BenchmarkResult:
	dtype = resolve_dtype(args.dtype)
	print(f"\n=== Loading {args.model_name} in {mode.upper()} mode on {args.device} ({args.dtype}) ===")
	model, tokenizer = load_for_mode(mode, args.model_name, args.device, dtype)

	perplexity = None
	hellaswag_acc = None
	piqa_acc = None

	if not args.skip_ppl:
		perplexity = evaluate_perplexity(
			model=model,
			tokenizer=tokenizer,
			dataset_name=args.ppl_dataset,
			dataset_config=args.ppl_config,
			split=args.ppl_split,
			text_field=args.ppl_text_field,
			device=args.device,
			max_length=args.ppl_max_length,
			stride=args.ppl_stride,
			cache_dir=args.cache_dir,
			max_samples=args.max_samples,
		)
		print(f"{mode.upper()} perplexity: {perplexity:.4f}")

	if not args.skip_hellaswag:
		hellaswag_acc = evaluate_multiple_choice(
			model=model,
			tokenizer=tokenizer,
			dataset_name=args.hellaswag_dataset,
			split=args.hellaswag_split,
			builder=build_hellaswag_prompt_and_choices,
			device=args.device,
			cache_dir=args.cache_dir,
			max_samples=args.max_samples,
		)
		print(f"{mode.upper()} HellaSwag acc: {hellaswag_acc:.4%}")

	if not args.skip_piqa:
		piqa_acc = evaluate_multiple_choice(
			model=model,
			tokenizer=tokenizer,
			dataset_name=args.piqa_dataset,
			split=args.piqa_split,
			builder=build_piqa_prompt_and_choices,
			device=args.device,
			cache_dir=args.cache_dir,
			max_samples=args.max_samples,
		)		
		print(f"{mode.upper()} PIQA acc: {piqa_acc:.4%}")

	del model
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return BenchmarkResult(
		mode=mode,
		perplexity=perplexity,
		hellaswag_acc=hellaswag_acc,
		piqa_acc=piqa_acc,
	)


def print_summary(results: Sequence[BenchmarkResult]) -> None:
	print("\n=== SUMMARY ===")
	header = f"{'Mode':<8} {'Perplexity':>12} {'HellaSwag':>12} {'PIQA':>12}"
	print(header)
	print("-" * len(header))
	for res in results:
		ppl = f"{res.perplexity:.4f}" if res.perplexity is not None else "-"
		hs = f"{100 * res.hellaswag_acc:.2f}%" if res.hellaswag_acc is not None else "-"
		pq = f"{100 * res.piqa_acc:.2f}%" if res.piqa_acc is not None else "-"
		print(f"{res.mode:<8} {ppl:>12} {hs:>12} {pq:>12}")

	if len(results) == 2:
		a, b = results
		print("\n=== DELTA (second - first) ===")
		if a.perplexity is not None and b.perplexity is not None:
			delta_ppl = b.perplexity - a.perplexity
			rel_ppl = 100.0 * delta_ppl / a.perplexity
			print(f"Perplexity delta: {delta_ppl:+.4f} ({rel_ppl:+.2f}%)")
		if a.hellaswag_acc is not None and b.hellaswag_acc is not None:
			print(f"HellaSwag delta: {(100.0 * (b.hellaswag_acc - a.hellaswag_acc)):+.2f} pts")
		if a.piqa_acc is not None and b.piqa_acc is not None:
			print(f"PIQA delta: {(100.0 * (b.piqa_acc - a.piqa_acc)):+.2f} pts")


def main() -> None:
	args = parse_args()
	modes = [args.mode] if args.mode in {"gpu", "meso"} else ["gpu", "meso"]
	results = [run_mode(mode, args) for mode in modes]
	print_summary(results)

	if args.output_json:
		with open(args.output_json, "w", encoding="utf-8") as f:
			json.dump([asdict(r) for r in results], f, indent=2)
		print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
	main()
