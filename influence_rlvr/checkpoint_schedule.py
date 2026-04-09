from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

_MOD_NAME = "influence_rlvr.modes"
if _MOD_NAME in sys.modules:
    _MOD = sys.modules[_MOD_NAME]
else:
    _MOD_PATH = Path(__file__).resolve().parent / "modes.py"
    _SPEC = importlib.util.spec_from_file_location(_MOD_NAME, _MOD_PATH)
    _MOD = importlib.util.module_from_spec(_SPEC)
    assert _SPEC.loader
    sys.modules[_MOD_NAME] = _MOD
    _SPEC.loader.exec_module(_MOD)
CheckpointThinningConfig = _MOD.CheckpointThinningConfig
CheckpointThinningMode = _MOD.CheckpointThinningMode

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def checkpoint_step(checkpoint_dir):
    name = os.path.basename(os.path.normpath(checkpoint_dir))
    match = _CHECKPOINT_RE.match(name)
    if match is None:
        raise ValueError(f"Invalid checkpoint directory: {checkpoint_dir}")
    return int(match.group(1))


def list_checkpoint_dirs(output_dir):
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Checkpoint output directory not found: {output_dir}")

    checkpoint_dirs = []
    for entry in os.listdir(output_dir):
        path = os.path.join(output_dir, entry)
        if os.path.isdir(path) and _CHECKPOINT_RE.match(entry):
            checkpoint_dirs.append(path)

    checkpoint_dirs.sort(key=checkpoint_step)
    return checkpoint_dirs


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _lookup_learning_rate(trainer_state, step, default_learning_rate):
    last_seen = None
    for item in trainer_state.get("log_history", []):
        item_step = item.get("step")
        if item_step is None or "learning_rate" not in item:
            continue
        if item_step == step:
            return float(item["learning_rate"])
        if item_step <= step and (last_seen is None or item_step > last_seen[0]):
            last_seen = (item_step, float(item["learning_rate"]))

    if last_seen is not None:
        return last_seen[1]
    return float(default_learning_rate)


def build_checkpoint_schedule(output_dir, default_learning_rate):
    schedule = []
    for path in list_checkpoint_dirs(output_dir):
        trainer_state_path = os.path.join(path, "trainer_state.json")
        trainer_state = _load_json(trainer_state_path) if os.path.exists(trainer_state_path) else {}
        step = int(trainer_state.get("global_step", checkpoint_step(path)))
        learning_rate = _lookup_learning_rate(trainer_state, step, default_learning_rate)
        schedule.append({
            "step": step,
            "path": path,
            "learning_rate": learning_rate,
        })

    schedule.sort(key=lambda item: item["step"])
    return schedule


def _thin_checkpoint_schedule_polynomial(schedule, target_count, power):
    n = len(schedule)
    if n <= target_count:
        return list(schedule)
    max_idx = n - 1
    oversamples = (target_count * 2, target_count * 4, max(n, target_count * 8))
    indices = np.array([], dtype=np.int64)
    for size in oversamples:
        x = np.linspace(0.0, 1.0, int(size))
        curve = np.power(x, float(power))
        indices = np.unique(np.round(curve * max_idx).astype(np.int64))
        if len(indices) >= target_count:
            break
    if len(indices) > target_count:
        pick = np.linspace(0, len(indices) - 1, target_count)
        indices = np.sort(np.unique(indices[pick.round().astype(np.int64)]))
    while len(indices) < target_count:
        extra = np.linspace(0, max_idx, target_count * 2)
        for e in extra:
            gi = int(round(float(e)))
            gi = max(0, min(max_idx, gi))
            if gi not in set(indices.tolist()):
                indices = np.sort(np.unique(np.append(indices, gi)))
            if len(indices) >= target_count:
                break
        if len(indices) < target_count:
            for gi in range(n):
                if gi not in set(indices.tolist()):
                    indices = np.sort(np.unique(np.append(indices, gi)))
                if len(indices) >= target_count:
                    break
    if len(indices) > target_count:
        pick = np.linspace(0, len(indices) - 1, target_count)
        indices = indices[pick.round().astype(np.int64)]
        indices = np.sort(np.unique(indices))
    return [schedule[int(i)] for i in indices]


def _thin_checkpoint_schedule_piecewise(
    schedule,
    early_last_index,
    mid_last_index,
    mid_stride,
    late_stride,
):
    thinned = []
    for i, cp in enumerate(schedule):
        if i <= early_last_index:
            thinned.append(cp)
        elif i <= mid_last_index:
            if mid_stride <= 0 or i % mid_stride == 0:
                thinned.append(cp)
        else:
            if late_stride <= 0 or i % late_stride == 0:
                thinned.append(cp)
    return thinned


def _thin_checkpoint_schedule_learning_rate(schedule, target_count):
    n = len(schedule)
    if n <= target_count:
        return list(schedule)
    cumulative_lrs = []
    current_sum = 0.0
    for cp in schedule:
        current_sum += float(cp["learning_rate"])
        cumulative_lrs.append(current_sum)
    total = cumulative_lrs[-1]
    if total <= 0.0:
        return _thin_checkpoint_schedule_polynomial(schedule, target_count, power=1.0)
    cumulative_array = np.array(cumulative_lrs, dtype=np.float64)
    target_milestones = np.linspace(0.0, total, int(target_count))
    selected_indices = []
    for target in target_milestones:
        idx = int(np.argmin(np.abs(cumulative_array - target)))
        if not selected_indices or idx != selected_indices[-1]:
            selected_indices.append(idx)
    seen = set()
    ordered = []
    for idx in selected_indices:
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    selected_indices = ordered
    for idx in range(n):
        if len(selected_indices) >= target_count:
            break
        if idx not in seen:
            seen.add(idx)
            selected_indices.append(idx)
    return [schedule[i] for i in selected_indices[:target_count]]


def thin_checkpoint_schedule(
    schedule,
    config: CheckpointThinningConfig | None = None,
    *,
    log: bool = True,
):
    if not schedule:
        return []
    cfg = config or CheckpointThinningConfig()
    mode = CheckpointThinningMode.parse(cfg.mode)
    n = len(schedule)
    steps_before = [cp["step"] for cp in schedule]

    if mode == CheckpointThinningMode.NONE:
        if log:
            print(
                f"Checkpoint thinning: mode=none — using all {n} checkpoints",
                flush=True,
            )
            if n <= 48:
                print(f"  steps={steps_before}", flush=True)
            else:
                print(
                    f"  steps: first 8 {steps_before[:8]} … last 8 {steps_before[-8:]}",
                    flush=True,
                )
        return list(schedule)

    if mode == CheckpointThinningMode.PIECEWISE_BUCKET:
        thinned = _thin_checkpoint_schedule_piecewise(
            schedule,
            cfg.piecewise_early_last_index,
            cfg.piecewise_mid_last_index,
            cfg.piecewise_mid_stride,
            cfg.piecewise_late_stride,
        )
        if log:
            steps_after = [cp["step"] for cp in thinned]
            print(
                f"Checkpoint thinning: mode=piecewise_bucket "
                f"(early_last_index={cfg.piecewise_early_last_index}, "
                f"mid_last_index={cfg.piecewise_mid_last_index}, "
                f"mid_stride={cfg.piecewise_mid_stride}, late_stride={cfg.piecewise_late_stride})",
                flush=True,
            )
            print(f"  {n} -> {len(thinned)} checkpoints", flush=True)
            print(f"  selected steps: {steps_after}", flush=True)
            _dropped = sorted(set(steps_before) - set(steps_after))
            if len(_dropped) <= 48:
                print(f"  dropped steps: {_dropped}", flush=True)
            else:
                print(
                    f"  dropped {len(_dropped)} steps (showing first 24): {_dropped[:24]} …",
                    flush=True,
                )
        return thinned

    tc = cfg.target_count
    if tc is None or tc <= 0 or n <= tc:
        if log:
            print(
                f"Checkpoint thinning: mode={mode.value} skipped "
                f"(target_count={tc}, have={n})",
                flush=True,
            )
            if n <= 48:
                print(f"  steps={steps_before}", flush=True)
            else:
                print(
                    f"  steps: first 8 {steps_before[:8]} … last 8 {steps_before[-8:]}",
                    flush=True,
                )
        return list(schedule)

    if mode == CheckpointThinningMode.POLYNOMIAL:
        thinned = _thin_checkpoint_schedule_polynomial(schedule, tc, cfg.polynomial_power)
    elif mode == CheckpointThinningMode.LEARNING_RATE:
        thinned = _thin_checkpoint_schedule_learning_rate(schedule, tc)
    else:
        return list(schedule)

    if log:
        steps_after = [cp["step"] for cp in thinned]
        print(
            f"Checkpoint thinning: mode={mode.value}, target_count={tc}"
            + (
                f", polynomial_power={cfg.polynomial_power}"
                if mode == CheckpointThinningMode.POLYNOMIAL
                else ""
            ),
            flush=True,
        )
        print(f"  {n} -> {len(thinned)} checkpoints", flush=True)
        print(f"  selected steps: {steps_after}", flush=True)
        _dropped = sorted(set(steps_before) - set(steps_after))
        if len(_dropped) <= 48:
            print(f"  dropped steps: {_dropped}", flush=True)
        else:
            print(
                f"  dropped {len(_dropped)} steps (showing first 24): {_dropped[:24]} …",
                flush=True,
            )
    return thinned
