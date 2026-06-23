#!/usr/bin/env python3
"""
###############################################################################
# Module:        Attention.py
# Description:   Multi-head attention layer implementation for the Small Language Model (SLM) architecture.
#
# Synopsis:      This module implements the attention layer's matrix multiplication,
#                acting as a functional switch based on the global execution mode.
#                It selectively executes either the MESO IMC-based dot-product
#                kernel (when mode = 1) or the standard CMOS GPU-based dot-product
#                implementation (when mode = 0) for comparative analysis.
#
#                This patched version also records MESO-vs-CMOS numerical error at
#                the three existing "#Error Calc" points:
#                1. scaled QK^T scores before masking/softmax
#                2. attention probabilities after masking/softmax
#                3. attention output after probabilities @ V
#
###############################################################################
"""

import csv
import math
import os
import sys
from collections import OrderedDict

import torch
from torch import nn

sys.path.append('../../../../')

from DotProductMESO.TRDP_MESO import FPMatMulMESO4D
from DotProductMESO.TRDP_MESO import FPMatMulMESO4D2

from DotProductMESO.AttMESOTimePlex import FPMatMulMESO4DTimePlex
from DotProductMESO.AttMESOTimePlex import FPMatMulMESO4D2TimePlex

###python GPT2XL_Benchmarks.py --mode both

ERROR_STAGES = ("qk_scores", "softmax_probs", "attention_output")
DEFAULT_ERROR_PHASES = ("train", "valid")


def _empty_error_stats():
    return {
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "sum_ref_abs": 0.0,
        "sum_ref_sq": 0.0,
        "max_abs": 0.0,
        "numel": 0,
        "samples": 0,
        "batches": 0,
    }


def _clone_empty_error_stats():
    return {stage: _empty_error_stats() for stage in ERROR_STAGES}


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, mode, bit_width, cycle_res, var):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        ###
        self.attention = ScaleDotProductAttention(mode, bit_width, cycle_res, var)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out = self.attention(q, k, v, mask=mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        # we are doing this transposing because then we can use in-built MMM operations. The MMM is performed between the last two dimensions. view basically changes the dimensions into what we want.
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        # for transmission of data, we retranspose the data into the initial form.
        # contiguous is basically a memory format. For ease, we use the same format for all the transmitted input data.
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class ScaleDotProductAttention(nn.Module):

    def __init__(self, mode, bit_width, cycle_res, var):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.mode = mode
        self.qktdp = FPMatMulMESO4DTimePlex(bit_width=bit_width[0].item(), cycle_res=cycle_res[0].item(), var=var)
        self.smvadp = FPMatMulMESO4D2TimePlex(bit_width=bit_width[1].item(), cycle_res=cycle_res[1].item(), var=var)

        # Error tracking is intentionally kept out of state_dict/checkpoints.
        self.collect_error_stats = True
        self.error_phase = None
        self._error_stats = OrderedDict((phase, _clone_empty_error_stats()) for phase in DEFAULT_ERROR_PHASES)

    def _active_error_phase(self):
        if self.error_phase is not None:
            return self.error_phase
        return "train" if self.training else "valid"

    def reset_error_stats(self, phase=None):
        """
        Reset accumulated error statistics.

        Args:
            phase: None resets all phases; otherwise reset only that phase, e.g. "train" or "valid".
        """
        if phase is None:
            self._error_stats = OrderedDict((p, _clone_empty_error_stats()) for p in DEFAULT_ERROR_PHASES)
            return

        if phase not in self._error_stats:
            self._error_stats[phase] = _clone_empty_error_stats()
        else:
            self._error_stats[phase] = _clone_empty_error_stats()

    def set_error_phase(self, phase=None):
        """
        Override the phase used for error accumulation.

        Passing None returns to automatic behavior: model.train() => "train", model.eval() => "valid".
        """
        self.error_phase = phase

    def _stage_stats(self, stage):
        phase = self._active_error_phase()
        if phase not in self._error_stats:
            self._error_stats[phase] = _clone_empty_error_stats()
        if stage not in self._error_stats[phase]:
            self._error_stats[phase][stage] = _empty_error_stats()
        return self._error_stats[phase][stage]

    def _record_error(self, stage, meso_tensor, cmos_tensor):
        """
        Accumulate MESO-vs-CMOS error for one tensor pair.

        The CMOS tensor is treated as the reference for relative error denominators.
        Recorded metrics can be read with get_error_stats().
        """
        if not self.collect_error_stats:
            return

        with torch.no_grad():
            if meso_tensor.shape != cmos_tensor.shape:
                raise RuntimeError(
                    f"Cannot record attention error for stage '{stage}': "
                    f"MESO shape {tuple(meso_tensor.shape)} != CMOS shape {tuple(cmos_tensor.shape)}"
                )

            meso = meso_tensor.detach().float()
            cmos = cmos_tensor.detach().float()
            diff = meso - cmos

            finite = torch.isfinite(diff) & torch.isfinite(cmos)
            count = int(finite.sum().item())
            if count == 0:
                return

            zero = torch.zeros((), dtype=diff.dtype, device=diff.device)
            abs_diff = torch.where(finite, diff.abs(), zero)
            sq_diff = torch.where(finite, diff * diff, zero)
            abs_ref = torch.where(finite, cmos.abs(), zero)
            sq_ref = torch.where(finite, cmos * cmos, zero)

            stats = self._stage_stats(stage)
            stats["sum_abs"] += float(abs_diff.sum().item())
            stats["sum_sq"] += float(sq_diff.sum().item())
            stats["sum_ref_abs"] += float(abs_ref.sum().item())
            stats["sum_ref_sq"] += float(sq_ref.sum().item())
            stats["max_abs"] = max(stats["max_abs"], float(abs_diff.max().item()))
            stats["numel"] += count
            stats["samples"] += int(meso_tensor.shape[0]) if meso_tensor.dim() > 0 else 1
            stats["batches"] += 1

    def get_error_stats(self, phase=None, eps=1e-12):
        """
        Return accumulated error statistics.

        The returned dict is nested as phase -> stage -> metrics. Metrics include:
        mae, rmse, relative_l1, relative_l2, max_abs, samples, batches, numel.
        """
        phases = [phase] if phase is not None else list(self._error_stats.keys())
        out = OrderedDict()

        for phase_name in phases:
            if phase_name not in self._error_stats:
                continue
            phase_out = OrderedDict()
            for stage, raw in self._error_stats[phase_name].items():
                numel = raw["numel"]
                if numel == 0:
                    continue

                sum_sq = raw["sum_sq"]
                sum_ref_sq = raw["sum_ref_sq"]
                phase_out[stage] = OrderedDict([
                    ("mae", raw["sum_abs"] / numel),
                    ("rmse", math.sqrt(sum_sq / numel)),
                    ("relative_l1", raw["sum_abs"] / (raw["sum_ref_abs"] + eps)),
                    ("relative_l2", math.sqrt(sum_sq) / (math.sqrt(sum_ref_sq) + eps)),
                    ("max_abs", raw["max_abs"]),
                    ("samples", raw["samples"]),
                    ("batches", raw["batches"]),
                    ("numel", numel),
                ])

            if phase_out:
                out[phase_name] = phase_out

        return out

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = q.size()
        k_t = k.transpose(2, 3)
        scale = math.sqrt(d_tensor)

        # Keep the active execution path differentiable. Compute the inactive
        # comparison path without autograd so error tracking does not inflate the
        # training graph.
        if self.mode == 1:
            score_meso = self.qktdp(q, k) / scale
            with torch.no_grad():
                score_cmos = (q @ k_t) / scale
            score = score_meso
        else:
            score_cmos = (q @ k_t) / scale
            with torch.no_grad():
                score_meso = self.qktdp(q, k) / scale
            score = score_cmos

        # Error Calc 1: scaled QK^T scores before masking/softmax.
        self._record_error("qk_scores", score_meso, score_cmos)

        if mask is not None:
            masked_positions = mask == 0
            score = score.masked_fill(masked_positions, -1e9)
            score_meso = score_meso.masked_fill(masked_positions, -1e9)
            score_cmos = score_cmos.masked_fill(masked_positions, -1e9)

        if self.mode == 1:
            prob_meso = self.softmax(score)
            with torch.no_grad():
                prob_cmos = self.softmax(score_cmos)
            prob = prob_meso
        else:
            prob_cmos = self.softmax(score)
            with torch.no_grad():
                prob_meso = self.softmax(score_meso)
            prob = prob_cmos

        # Error Calc 2: attention probabilities after masking/softmax.
        self._record_error("softmax_probs", prob_meso, prob_cmos)

        if self.mode == 1:
            out_meso = self.smvadp(prob, v)
            with torch.no_grad():
                out_cmos = prob_cmos @ v
            out = out_meso
        else:
            out_cmos = prob @ v
            with torch.no_grad():
                out_meso = self.smvadp(prob_meso, v)
            out = out_cmos

        # Error Calc 3: attention output after probabilities @ V.
        self._record_error("attention_output", out_meso, out_cmos)
        return out


class FFN(nn.Module):
    def __init__(self, d_model, hidden, output, drop_prob=0.1):
        super(FFN, self).__init__()
        self.f1 = nn.Linear(d_model, hidden)
        self.f2 = nn.Linear(hidden, output)
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.f1(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.f2(x)
        return x


def iter_attention_error_modules(model):
    """Yield (module_name, ScaleDotProductAttention) for every attention block in a model."""
    for name, module in model.named_modules():
        if isinstance(module, ScaleDotProductAttention):
            yield name, module


def reset_attention_error_stats(model, phase=None):
    """Reset error stats for all attention blocks in the model."""
    for _, module in iter_attention_error_modules(model):
        module.reset_error_stats(phase=phase)


def set_attention_error_phase(model, phase=None):
    """
    Set error accumulation phase for all attention blocks.

    Use phase="train" before the training iterator and phase="valid" before the
    validation iterator. Use phase=None to return to automatic train/eval behavior.
    """
    for _, module in iter_attention_error_modules(model):
        module.set_error_phase(phase=phase)


def set_attention_error_tracking(model, enabled=True):
    """Enable or disable attention error collection for all attention blocks."""
    for _, module in iter_attention_error_modules(model):
        module.collect_error_stats = bool(enabled)


def get_attention_error_summary(model, epoch=None, phase=None):
    """
    Return a flat list of per-layer attention error rows.

    Each row contains epoch, phase, layer, stage, mae, rmse, relative_l1,
    relative_l2, max_abs, samples, batches, and numel.
    """
    rows = []
    for name, module in iter_attention_error_modules(model):
        module_stats = module.get_error_stats(phase=phase)
        for phase_name, phase_stats in module_stats.items():
            for stage, metrics in phase_stats.items():
                row = OrderedDict()
                row["epoch"] = epoch
                row["phase"] = phase_name
                row["layer"] = name
                row["stage"] = stage
                row.update(metrics)
                rows.append(row)
    return rows


def format_attention_error_summary(model, epoch=None, phase=None, digits=6):
    """Format the current attention error summary as a printable table."""
    rows = get_attention_error_summary(model, epoch=epoch, phase=phase)
    if not rows:
        label = f" for epoch {epoch}" if epoch is not None else ""
        return f"No attention error statistics recorded{label}."

    header = (
        "epoch phase layer stage samples batches numel "
        "mae rmse relative_l1 relative_l2 max_abs"
    )
    lines = [header]
    for row in rows:
        epoch_value = "" if row["epoch"] is None else str(row["epoch"])
        lines.append(
            f"{epoch_value} "
            f"{row['phase']} "
            f"{row['layer']} "
            f"{row['stage']} "
            f"{row['samples']} "
            f"{row['batches']} "
            f"{row['numel']} "
            f"{row['mae']:.{digits}e} "
            f"{row['rmse']:.{digits}e} "
            f"{row['relative_l1']:.{digits}e} "
            f"{row['relative_l2']:.{digits}e} "
            f"{row['max_abs']:.{digits}e}"
        )
    return "\n".join(lines)


def print_attention_error_summary(model, epoch=None, phase=None, digits=6):
    """Print the current attention error summary."""
    print(format_attention_error_summary(model, epoch=epoch, phase=phase, digits=digits))


def append_attention_error_summary_csv(model, csv_path, epoch=None, phase=None):
    """Append the current attention error summary to a CSV file."""
    rows = get_attention_error_summary(model, epoch=epoch, phase=phase)
    if not rows:
        return

    fieldnames = [
        "epoch", "phase", "layer", "stage", "mae", "rmse", "relative_l1",
        "relative_l2", "max_abs", "samples", "batches", "numel",
    ]
    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
