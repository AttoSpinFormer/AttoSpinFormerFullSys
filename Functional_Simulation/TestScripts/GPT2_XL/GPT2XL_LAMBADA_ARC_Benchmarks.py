#!/usr/bin/env python3
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture

"""Benchmark GPT-2 XL in GPU vs MESO mode on LAMBADA and ARC-Challenge.

Metrics:
- LAMBADA last-token exact-match accuracy
- ARC-Challenge multiple-choice accuracy via conditional log-likelihood

This script is intentionally separate from GPT2XL_Benchmarks.py so you can keep
that path unchanged while adding two more benchmarks.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from GPT2XL_HF_MESO import load_model_and_tokenizer
from GPT2XL_Config import MODEL_NAME, DEVICE, DTYPE


DEFAULT_LAMBADA_CANDIDATES = [
    ("EleutherAI/lambada_openai", None, "test", "text"),
    ("craffel/openai_lambada", None, "test", "text"),
    ("cimec/lambada", None, "test", "text"),
]
DEFAULT_ARC_DATASET = "allenai/ai2_arc"
DEFAULT_ARC_CONFIG = "ARC-Challenge"
DEFAULT_ARC_SPLIT = "validation"


@dataclass
class BenchmarkResult:
    mode: str
    lambada_acc: Optional[float]
    arc_challenge_acc: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-2 XL LAMBADA and ARC-Challenge benchmarks in GPU and/or MESO mode.")
    parser.add_argument("--mode", choices=["gpu", "meso", "both"], default="both")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=str(DTYPE).split(".")[-1])
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per benchmark for quick smoke tests.")
    parser.add_argument("--skip-lambada", action="store_true")
    parser.add_argument("--skip-arc", action="store_true")
    parser.add_argument("--lambada-dataset", default=None, help="Override the LAMBADA dataset repo id.")
    parser.add_argument("--lambada-config", default=None, help="Optional config for the overridden LAMBADA dataset.")
    parser.add_argument("--lambada-split", default="test")
    parser.add_argument("--lambada-text-field", default="text")
    parser.add_argument("--arc-dataset", default=DEFAULT_ARC_DATASET)
    parser.add_argument("--arc-config", default=DEFAULT_ARC_CONFIG)
    parser.add_argument("--arc-split", default=DEFAULT_ARC_SPLIT)
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


def load_for_mode(mode: str, model_name: str, device: str, dtype: torch.dtype):
    use_meso = mode == "meso"
    model, tokenizer = load_model_and_tokenizer(model_name, device, dtype, use_meso=use_meso)
    return model, tokenizer


def load_dataset_with_fallback(
    primary_name: Optional[str],
    primary_config: Optional[str],
    primary_split: str,
    primary_text_field: str,
    cache_dir: Optional[str],
):
    from datasets import load_dataset

    candidates = []
    if primary_name:
        candidates.append((primary_name, primary_config, primary_split, primary_text_field))
    candidates.extend(DEFAULT_LAMBADA_CANDIDATES)

    last_err: Optional[Exception] = None
    for dataset_name, dataset_config, split, text_field in candidates:
        try:
            if dataset_config is None:
                dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
            else:
                dataset = load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir)
            return dataset, dataset_name, dataset_config, split, text_field
        except Exception as exc:
            last_err = exc
            continue

    if last_err is None:
        raise RuntimeError("No LAMBADA dataset candidates were provided.")
    raise last_err


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
def evaluate_arc_challenge(
    model,
    tokenizer,
    dataset_name: str,
    dataset_config: str,
    split: str,
    device: str,
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> float:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0

    for example in tqdm(dataset, desc=f"Evaluating {dataset_name}/{dataset_config}:{split}"):
        question = example["question"]
        if isinstance(question, dict):
            stem = question.get("stem", "")
            choices_struct = question.get("choices", {})
            choice_texts = list(choices_struct.get("text", []))
            choice_labels = list(choices_struct.get("label", []))
        else:
            stem = str(question)
            choices = example.get("choices", {})
            if isinstance(choices, dict):
                choice_texts = list(choices.get("text", []))
                choice_labels = list(choices.get("label", []))
            else:
                choice_texts = [str(c.get("text", "")) for c in choices]
                choice_labels = [str(c.get("label", "")) for c in choices]

        if not choice_texts or not choice_labels or len(choice_texts) != len(choice_labels):
            continue

        prompt = normalize_whitespace(f"Question: {stem}\nAnswer:")
        scores: List[float] = []
        for choice in choice_texts:
            total_logprob, token_count = conditional_logprob(model, tokenizer, prompt, normalize_whitespace(choice), device)
            scores.append(total_logprob / max(token_count, 1))

        pred_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        pred_label = str(choice_labels[pred_idx]).strip()
        gold_label = str(example.get("answerKey", "")).strip()

        correct += int(pred_label == gold_label)
        total += 1

    return correct / max(total, 1)


@torch.no_grad()
def evaluate_lambada(
    model,
    tokenizer,
    dataset_name: Optional[str],
    dataset_config: Optional[str],
    split: str,
    text_field: str,
    device: str,
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Tuple[float, str]:
    dataset, resolved_name, resolved_config, resolved_split, resolved_field = load_dataset_with_fallback(
        dataset_name, dataset_config, split, text_field, cache_dir
    )

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0

    for example in tqdm(dataset, desc=f"Evaluating {resolved_name}:{resolved_split}"):
        raw_text = example.get(resolved_field, "")
        if not isinstance(raw_text, str):
            continue
        text = raw_text.rstrip()
        if not text:
            continue

        full_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        if full_ids.shape[1] < 2:
            continue

        context_ids = full_ids[:, :-1]
        target_id = int(full_ids[0, -1].item())

        outputs = model(context_ids, use_cache=False)
        pred_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())

        correct += int(pred_id == target_id)
        total += 1

    resolved_display = resolved_name if resolved_config is None else f"{resolved_name}/{resolved_config}"
    return correct / max(total, 1), resolved_display


def run_mode(mode: str, args: argparse.Namespace) -> BenchmarkResult:
    dtype = resolve_dtype(args.dtype)
    print(f"\n=== Loading {args.model_name} in {mode.upper()} mode on {args.device} ({args.dtype}) ===")
    model, tokenizer = load_for_mode(mode, args.model_name, args.device, dtype)

    lambada_acc = None
    arc_acc = None

    if not args.skip_lambada:
        lambada_acc, resolved_lambada = evaluate_lambada(
            model=model,
            tokenizer=tokenizer,
            dataset_name=args.lambada_dataset,
            dataset_config=args.lambada_config,
            split=args.lambada_split,
            text_field=args.lambada_text_field,
            device=args.device,
            cache_dir=args.cache_dir,
            max_samples=args.max_samples,
        )
        print(f"{mode.upper()} LAMBADA acc ({resolved_lambada}): {lambada_acc:.4%}")

    if not args.skip_arc:
        arc_acc = evaluate_arc_challenge(
            model=model,
            tokenizer=tokenizer,
            dataset_name=args.arc_dataset,
            dataset_config=args.arc_config,
            split=args.arc_split,
            device=args.device,
            cache_dir=args.cache_dir,
            max_samples=args.max_samples,
        )
        print(f"{mode.upper()} ARC-Challenge acc: {arc_acc:.4%}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return BenchmarkResult(
        mode=mode,
        lambada_acc=lambada_acc,
        arc_challenge_acc=arc_acc,
    )


def print_summary(results: Sequence[BenchmarkResult]) -> None:
    print("\n=== SUMMARY ===")
    header = f"{'Mode':<8} {'LAMBADA':>12} {'ARC-Chal':>12}"
    print(header)
    print("-" * len(header))
    for res in results:
        lb = f"{100 * res.lambada_acc:.2f}%" if res.lambada_acc is not None else "-"
        arc = f"{100 * res.arc_challenge_acc:.2f}%" if res.arc_challenge_acc is not None else "-"
        print(f"{res.mode:<8} {lb:>12} {arc:>12}")

    if len(results) == 2:
        a, b = results
        print("\n=== DELTA (second - first) ===")
        if a.lambada_acc is not None and b.lambada_acc is not None:
            print(f"LAMBADA delta: {(100.0 * (b.lambada_acc - a.lambada_acc)):+.2f} pts")
        if a.arc_challenge_acc is not None and b.arc_challenge_acc is not None:
            print(f"ARC-Challenge delta: {(100.0 * (b.arc_challenge_acc - a.arc_challenge_acc)):+.2f} pts")


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
