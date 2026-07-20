#!/usr/bin/env python3

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


TENSOR_RE = re.compile(
    r"^blk\.(?P<layer>[0-9]+)\."
    r"(?P<stage>attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$"
)


def as_int(row, key):
    value = row.get(key, "")
    return int(value or 0)


def fail(errors, message):
    errors.append(message)


def main():
    parser = argparse.ArgumentParser(description="Validate an AiF parallel GEMV CSV log")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--layers", type=int, default=0, help="expected layer count; 0 = infer")
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("AIF parallel validation: FAIL (CSV has no rows)")

    errors = []
    required = {
        "call_index", "phase", "decode_call_index", "success", "execution_mode",
        "tensor_name", "rows", "full_rows", "host_rows", "host_matrix_nbytes",
        "host_actual_ns", "parallel_elapsed_ns", "overlap_ns", "stats_valid",
        "output_mismatches", "layout_miss_count", "predicted_device_ns",
        "qkv_group_id", "qkv_schedule", "q_heads", "kv_heads", "qkv_group_summary",
        "qkv_group_modeled_ns", "qkv_first_head_ready_ns", "qkv_last_head_ready_ns",
        "host_attention_actual_ns", "host_idle_wait_ns", "aif_idle_wait_ns",
        "head_pipeline_finish_ns", "head_pipeline_tail_wait_ns",
    }
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise SystemExit("AIF parallel validation: FAIL (missing columns: " + ", ".join(missing) + ")")

    call_indices = [as_int(row, "call_index") for row in rows]
    if len(set(call_indices)) != len(call_indices):
        fail(errors, "call_index values are not unique")
    if sorted(call_indices) != list(range(min(call_indices), max(call_indices) + 1)):
        fail(errors, "call_index values are not contiguous")

    for row in rows:
        name = row["tensor_name"]
        if as_int(row, "success") != 1:
            fail(errors, f"{name}: success != 1")
        if as_int(row, "stats_valid") != 1:
            fail(errors, f"{name}: stats_valid != 1")
        if as_int(row, "output_mismatches") != 0:
            fail(errors, f"{name}: output_mismatches != 0")
        if as_int(row, "layout_miss_count") != 0:
            fail(errors, f"{name}: layout_miss_count != 0")

    decode_rows = [row for row in rows if row.get("phase") == "decode"]
    by_step = defaultdict(list)
    for row in decode_rows:
        by_step[as_int(row, "decode_call_index")].append(row)
    if not by_step:
        fail(errors, "no decode rows found")

    inferred_layers = set()
    for row in decode_rows:
        match = TENSOR_RE.match(row["tensor_name"])
        if match:
            inferred_layers.add(int(match.group("layer")))
    layer_count = args.layers or len(inferred_layers)
    if layer_count <= 0:
        fail(errors, "could not infer layer count")
    elif inferred_layers != set(range(layer_count)):
        fail(errors, f"layer set mismatch: got {sorted(inferred_layers)}, expected 0..{layer_count - 1}")

    expected_stage_counts = {
        "attn_q": layer_count,
        "attn_k": layer_count,
        "attn_v": layer_count,
        "attn_output": layer_count,
        "ffn_gate": layer_count,
        "ffn_up": layer_count,
        "ffn_down": layer_count,
        "output": 1,
    }
    for step, step_rows in sorted(by_step.items()):
        counts = Counter()
        for row in step_rows:
            if row["tensor_name"] == "output.weight":
                counts["output"] += 1
                continue
            match = TENSOR_RE.match(row["tensor_name"])
            if match:
                counts[match.group("stage")] += 1
        if counts != Counter(expected_stage_counts):
            fail(errors, f"decode step {step}: stage counts {dict(counts)} != {expected_stage_counts}")

    mode_counts = Counter(row["execution_mode"] for row in decode_rows)
    step_count = len(by_step)
    expected_modes = Counter({
        "head_async": 3 * layer_count * step_count,
        "tensor_parallel": 3 * layer_count * step_count,
        "parallel_sync": (layer_count + 1) * step_count,
    })
    if mode_counts != expected_modes:
        fail(errors, f"execution-mode counts {dict(mode_counts)} != {dict(expected_modes)}")

    qkv_rows = [row for row in decode_rows if row["execution_mode"] == "head_async"]
    qkv_groups = defaultdict(list)
    for row in qkv_rows:
        group_id = as_int(row, "qkv_group_id")
        if group_id == 0:
            fail(errors, f"{row['tensor_name']}: qkv_group_id == 0")
            continue
        qkv_groups[group_id].append(row)

    expected_group_count = layer_count * step_count
    if len(qkv_groups) != expected_group_count:
        fail(errors, f"QKV group count {len(qkv_groups)} != {expected_group_count}")

    for group_id, group_rows in sorted(qkv_groups.items()):
        if len(group_rows) != 3:
            fail(errors, f"QKV group {group_id}: row count {len(group_rows)} != 3")
            continue

        stages = set()
        for row in group_rows:
            match = TENSOR_RE.match(row["tensor_name"])
            if match:
                stages.add(match.group("stage"))
            if row["qkv_schedule"] != "head_major":
                fail(errors, f"QKV group {group_id}: schedule '{row['qkv_schedule']}' != head_major")
        if stages != {"attn_q", "attn_k", "attn_v"}:
            fail(errors, f"QKV group {group_id}: stages {sorted(stages)} are incomplete")

        decode_indices = {as_int(row, "decode_call_index") for row in group_rows}
        layers = set()
        for row in group_rows:
            match = TENSOR_RE.match(row["tensor_name"])
            if match:
                layers.add(int(match.group("layer")))
        if len(decode_indices) != 1 or len(layers) != 1:
            fail(errors, f"QKV group {group_id}: rows span multiple decode steps or layers")

        q_heads = {as_int(row, "q_heads") for row in group_rows}
        kv_heads = {as_int(row, "kv_heads") for row in group_rows}
        if len(q_heads) != 1 or min(q_heads) <= 0:
            fail(errors, f"QKV group {group_id}: invalid q_heads {sorted(q_heads)}")
        if len(kv_heads) != 1 or min(kv_heads) <= 0 or max(kv_heads) > max(q_heads):
            fail(errors, f"QKV group {group_id}: invalid kv_heads {sorted(kv_heads)}")

        summaries = [row for row in group_rows if as_int(row, "qkv_group_summary") == 1]
        if len(summaries) != 1:
            fail(errors, f"QKV group {group_id}: summary row count {len(summaries)} != 1")
            continue
        summary = summaries[0]
        if ".attn_v.weight" not in summary["tensor_name"]:
            fail(errors, f"QKV group {group_id}: summary row is not attn_v")

        modeled = as_int(summary, "qkv_group_modeled_ns")
        predicted = sum(as_int(row, "predicted_device_ns") for row in group_rows)
        first_ready = as_int(summary, "qkv_first_head_ready_ns")
        last_ready = as_int(summary, "qkv_last_head_ready_ns")
        pipeline_finish = as_int(summary, "head_pipeline_finish_ns")
        if modeled == 0 or modeled != predicted:
            fail(errors, f"QKV group {group_id}: modeled {modeled} != predicted sum {predicted}")
        if first_ready <= 0 or first_ready >= last_ready:
            fail(errors, f"QKV group {group_id}: invalid first/last readiness {first_ready}/{last_ready}")
        if last_ready != modeled:
            fail(errors, f"QKV group {group_id}: last readiness {last_ready} != modeled {modeled}")
        if as_int(summary, "host_attention_actual_ns") == 0:
            fail(errors, f"QKV group {group_id}: host_attention_actual_ns == 0")
        if pipeline_finish < last_ready:
            fail(errors, f"QKV group {group_id}: pipeline finishes before final QKV head")

    ffn_rows = [row for row in decode_rows if row["execution_mode"] == "tensor_parallel"]
    for row in ffn_rows:
        if as_int(row, "host_rows") + as_int(row, "rows") != as_int(row, "full_rows"):
            fail(errors, f"{row['tensor_name']}: host_rows + AIF rows != full_rows")
        if as_int(row, "host_matrix_nbytes") == 0:
            fail(errors, f"{row['tensor_name']}: host_matrix_nbytes == 0")
        if as_int(row, "host_actual_ns") == 0:
            fail(errors, f"{row['tensor_name']}: host_actual_ns == 0")

    first_ready = sum(as_int(row, "head_first_ready_wait_ns") for row in decode_rows)
    overlap = sum(as_int(row, "overlap_ns") for row in decode_rows)
    pipeline_tail_wait = sum(as_int(row, "head_pipeline_tail_wait_ns") for row in decode_rows)
    if first_ready == 0:
        fail(errors, "head-level first-ready wait was never recorded")
    if overlap == 0:
        fail(errors, "no host/AIF overlap was recorded")

    print(f"rows={len(rows)} decode_steps={step_count} layers={layer_count}")
    print("execution_modes=" + ",".join(f"{key}:{value}" for key, value in sorted(mode_counts.items())))
    print(f"qkv_schedule=head_major qkv_groups={len(qkv_groups)}")
    print(
        f"head_first_ready_wait_ns={first_ready} recorded_overlap_ns={overlap} "
        f"pipeline_tail_wait_ns={pipeline_tail_wait}"
    )
    if errors:
        for message in errors[:20]:
            print("FAIL", message, file=sys.stderr)
        if len(errors) > 20:
            print(f"FAIL ... and {len(errors) - 20} more", file=sys.stderr)
        raise SystemExit(f"AIF parallel validation: FAIL ({len(errors)} checks)")

    print("AIF parallel validation: PASS")


if __name__ == "__main__":
    main()
