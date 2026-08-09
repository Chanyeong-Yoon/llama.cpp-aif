#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


EVAL_RE = re.compile(
    r"eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*runs\s*"
    r"\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)"
)
PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens\s*"
    r"\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)"
)
TOTAL_RE = re.compile(r"total time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens")
AIF_MODELED_RE = re.compile(
    r"GEMV-only modeled decode latency:.*avg=([0-9.]+)\s*us/step.*estimated=([0-9.]+)\s*tok/s"
)
AIF_IOCTL_RE = re.compile(
    r"ioctl-observed decode path:.*avg=([0-9.]+)\s*us/step\s*estimated=([0-9.]+)"
)
AIF_E2E_RE = re.compile(
    r"AIF eval:.*avg=([0-9.]+)\s*us/token\s*tps=([0-9.]+)"
)
MEMORY_SSD_IO_RE = re.compile(
    r"Memory\+SSD matrix-read (?:reference|lower bound):.*avg=([0-9.]+)\s*us/step.*estimated=([0-9.]+)"
)
MEMORY_SSD_FULL_RE = re.compile(
    r"Memory\+SSD estimated full decode:.*avg=([0-9.]+)\s*us/token\s*tps=([0-9.]+)"
)
AIF_CALLS_RE = re.compile(r"calls:\s*([0-9]+)\s*success:\s*([0-9]+)\s*stats_valid:\s*([0-9]+)")
AIF_COUNTER_RE = re.compile(
    r"final device counters:\s*post=([0-9]+)\s*gemv=([0-9]+)\s*tensors=([0-9]+)\s*layout_hit=([0-9]+)\s*layout_miss=([0-9]+)"
)
PARALLEL_RE = re.compile(
    r"parallel execution:\s*rows=([0-9]+)\s*host_matrix=([0-9.]+)\s*MiB\s*"
    r"host_modeled=([0-9.]+)\s*us\s*host_actual=([0-9.]+)\s*us\s*"
    r"first_ready_wait=([0-9.]+)\s*us\s*dependency_wait=([0-9.]+)\s*us\s*"
    r"component_elapsed=([0-9.]+)\s*us\s*recorded_overlap=([0-9.]+)\s*us"
)
HEAD_PIPELINE_RE = re.compile(
    r"head pipeline:\s*schedule=head_major\s*groups=([0-9]+)\s*"
    r"qkv_modeled=([0-9.]+)\s*us\s*first_head_ready=([0-9.]+)\s*us\s*"
    r"last_head_ready=([0-9.]+)\s*us\s*host_attention=([0-9.]+)\s*us\s*"
    r"host_idle=([0-9.]+)\s*us\s*aif_idle=([0-9.]+)\s*us\s*"
    r"pipeline_finish=([0-9.]+)\s*us\s*tail_wait=([0-9.]+)\s*us"
)

PAPER_PCIE_MIB_S = 8_000_000_000 / (1024 * 1024)


def shell_join(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_cmd(cmd, log_path, env, dry_run=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_line = shell_join(cmd)

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command_line}\n")
        log.flush()

        if dry_run:
            return 0

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def drop_caches(log_path, dry_run=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_line = "sync && echo 3 > /proc/sys/vm/drop_caches"

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {command_line}\n")
        log.flush()

        if dry_run:
            return 0

        sync_rc = subprocess.run(["sync"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if sync_rc.returncode != 0:
            log.write(sync_rc.stdout)
            return sync_rc.returncode

        try:
            Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="utf-8")
        except OSError as exc:
            log.write(f"drop_caches failed: {exc}\n")
            return 1

    return 0


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_perf_log(path):
    text = read_text(path)
    out = {
        "prompt_eval_total_ms": "",
        "prompt_eval_tokens": "",
        "prompt_eval_ms_per_token": "",
        "prompt_eval_tps": "",
        "eval_total_ms": "",
        "eval_runs": "",
        "eval_ms_per_token": "",
        "eval_tps": "",
        "total_ms": "",
        "total_tokens": "",
    }

    prompt = PROMPT_EVAL_RE.search(text)
    if prompt:
        out["prompt_eval_total_ms"] = prompt.group(1)
        out["prompt_eval_tokens"] = prompt.group(2)
        out["prompt_eval_ms_per_token"] = prompt.group(3)
        out["prompt_eval_tps"] = prompt.group(4)

    # Use the last non-prompt eval line, matching llama.cpp's final perf block.
    eval_matches = EVAL_RE.findall(text)
    if eval_matches:
        total_ms, runs, ms_per_token, tps = eval_matches[-1]
        out["eval_total_ms"] = total_ms
        out["eval_runs"] = runs
        out["eval_ms_per_token"] = ms_per_token
        out["eval_tps"] = tps

    total = TOTAL_RE.search(text)
    if total:
        out["total_ms"] = total.group(1)
        out["total_tokens"] = total.group(2)

    return out


def parse_aif_report(path):
    text = read_text(path)
    out = {
        "aif_calls": "",
        "aif_success": "",
        "aif_stats_valid": "",
        "aif_post_count": "",
        "aif_gemv_count": "",
        "aif_tensor_count": "",
        "aif_layout_hit": "",
        "aif_layout_miss": "",
        "aif_modeled_us_per_step": "",
        "aif_modeled_tps": "",
        "aif_ioctl_us_per_step": "",
        "aif_ioctl_tps": "",
        "aif_e2e_us_per_token": "",
        "aif_e2e_tps": "",
        "memory_ssd_io_us_per_step": "",
        "memory_ssd_io_tps": "",
        "memory_ssd_full_us_per_token": "",
        "memory_ssd_full_tps": "",
        "aif_parallel_rows": "",
        "aif_host_matrix_mib": "",
        "aif_host_modeled_us": "",
        "aif_host_actual_us": "",
        "aif_head_first_ready_wait_us": "",
        "aif_dependency_wait_us": "",
        "aif_parallel_component_us": "",
        "aif_recorded_overlap_us": "",
        "aif_qkv_groups": "",
        "aif_qkv_modeled_us": "",
        "aif_qkv_first_head_ready_us": "",
        "aif_qkv_last_head_ready_us": "",
        "aif_host_attention_us": "",
        "aif_host_idle_us": "",
        "aif_device_idle_us": "",
        "aif_head_pipeline_finish_us": "",
        "aif_head_pipeline_tail_wait_us": "",
    }

    calls = AIF_CALLS_RE.search(text)
    if calls:
        out["aif_calls"] = calls.group(1)
        out["aif_success"] = calls.group(2)
        out["aif_stats_valid"] = calls.group(3)

    counters = AIF_COUNTER_RE.search(text)
    if counters:
        out["aif_post_count"] = counters.group(1)
        out["aif_gemv_count"] = counters.group(2)
        out["aif_tensor_count"] = counters.group(3)
        out["aif_layout_hit"] = counters.group(4)
        out["aif_layout_miss"] = counters.group(5)

    modeled = AIF_MODELED_RE.search(text)
    if modeled:
        out["aif_modeled_us_per_step"] = modeled.group(1)
        out["aif_modeled_tps"] = modeled.group(2)

    ioctl = AIF_IOCTL_RE.search(text)
    if ioctl:
        out["aif_ioctl_us_per_step"] = ioctl.group(1)
        out["aif_ioctl_tps"] = ioctl.group(2)

    aif_e2e = AIF_E2E_RE.search(text)
    if aif_e2e:
        out["aif_e2e_us_per_token"] = aif_e2e.group(1)
        out["aif_e2e_tps"] = aif_e2e.group(2)

    memory_ssd_io = MEMORY_SSD_IO_RE.search(text)
    if memory_ssd_io:
        out["memory_ssd_io_us_per_step"] = memory_ssd_io.group(1)
        out["memory_ssd_io_tps"] = memory_ssd_io.group(2)

    memory_ssd_full = MEMORY_SSD_FULL_RE.search(text)
    if memory_ssd_full:
        out["memory_ssd_full_us_per_token"] = memory_ssd_full.group(1)
        out["memory_ssd_full_tps"] = memory_ssd_full.group(2)

    parallel = PARALLEL_RE.search(text)
    if parallel:
        out["aif_parallel_rows"] = parallel.group(1)
        out["aif_host_matrix_mib"] = parallel.group(2)
        out["aif_host_modeled_us"] = parallel.group(3)
        out["aif_host_actual_us"] = parallel.group(4)
        out["aif_head_first_ready_wait_us"] = parallel.group(5)
        out["aif_dependency_wait_us"] = parallel.group(6)
        out["aif_parallel_component_us"] = parallel.group(7)
        out["aif_recorded_overlap_us"] = parallel.group(8)

    head_pipeline = HEAD_PIPELINE_RE.search(text)
    if head_pipeline:
        out["aif_qkv_groups"] = head_pipeline.group(1)
        out["aif_qkv_modeled_us"] = head_pipeline.group(2)
        out["aif_qkv_first_head_ready_us"] = head_pipeline.group(3)
        out["aif_qkv_last_head_ready_us"] = head_pipeline.group(4)
        out["aif_host_attention_us"] = head_pipeline.group(5)
        out["aif_host_idle_us"] = head_pipeline.group(6)
        out["aif_device_idle_us"] = head_pipeline.group(7)
        out["aif_head_pipeline_finish_us"] = head_pipeline.group(8)
        out["aif_head_pipeline_tail_wait_us"] = head_pipeline.group(9)

    return out


def write_summary(path, rows):
    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_model_args(args):
    if args.model:
        return ["-m", args.model]
    return ["-hf", args.hf]


def main():
    parser = argparse.ArgumentParser(
        description="Run reproducible baseline and AIF decode benchmark passes for llama.cpp + NVMeVirt."
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("AIF_DEVICE", ""),
        help="NVMe namespace used for AIF passthrough; defaults to AIF_DEVICE",
    )
    parser.add_argument("--hf", default="ggml-org/tiny-llamas:Q4_0", help="Hugging Face model spec for llama.cpp -hf")
    parser.add_argument("--model", default="", help="Local GGUF model path for llama.cpp -m; overrides --hf")
    parser.add_argument("--prompt", default="Once upon a time,", help="Prompt text")
    parser.add_argument("-n", "--n-predict", type=int, default=128, help="Generated tokens per run")
    parser.add_argument("--runs", type=int, default=3, help="Number of repeated baseline/AIF pairs")
    parser.add_argument("--out-dir", default="", help="Output directory; default under /tmp/aif-bench-*")
    parser.add_argument("--bin-dir", default="./build/bin", help="Directory containing llama binaries")
    parser.add_argument("--cache", default="", help="LLAMA_CACHE value")
    parser.add_argument("--post-max", type=int, default=0, help="AIF post max, 0 = all matching tensors")
    parser.add_argument("--tensor-filter", default="", help="Optional --aif-tensor-filter regex")
    parser.add_argument("--top", type=int, default=12, help="Top tensors shown in AIF report")
    parser.add_argument("--steps", type=int, default=12, help="Decode steps shown in AIF report")
    parser.add_argument(
        "--pcie-mib-s",
        type=float,
        default=PAPER_PCIE_MIB_S,
        help="PCIe bandwidth used by llama-aif-report; default matches the AiF 8.0 GB/s setting",
    )
    parser.add_argument("--skip-baseline", action="store_true", help="Do not run CPU baseline")
    parser.add_argument("--skip-aif", action="store_true", help="Do not run AIF replacement")
    parser.add_argument("--no-reset", action="store_true", help="Do not issue AIF reset before each AIF run")
    parser.add_argument("--drop-caches", action="store_true", help="Drop Linux page cache before each llama-completion run; requires root")
    parser.add_argument("--aif-all-phases", action="store_true", help="Pass --aif-all-phases to llama.cpp")
    parser.add_argument("--aif-log-gemv", action="store_true", help="Pass --aif-log-gemv to llama.cpp")
    parser.add_argument("--parallel", action="store_true", help="run the AIF head/tensor parallel replacement mode")
    parser.add_argument("--host-budget-mib", type=int, default=8192, help="logical host-memory budget for --parallel")
    parser.add_argument("--kv-cache-mib", type=int, default=1024, help="KV-cache reservation for --parallel")
    parser.add_argument("--host-bandwidth-gbps", type=float, default=86.5, help="modeled retained-submatrix host bandwidth")
    parser.add_argument("--allow-eos", action="store_true", help="Allow generation to stop early on EOS; default ignores EOS for fixed-length timing")
    parser.add_argument("--extra", action="append", default=[], help="Extra argument passed to both llama-completion runs")
    parser.add_argument("--dry-run", action="store_true", help="Write commands but do not execute them")
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be positive")
    if not args.skip_aif and not args.device:
        parser.error("--device is required for AIF runs (or set AIF_DEVICE)")

    root = Path(args.out_dir) if args.out_dir else Path("/tmp") / (
        "aif-bench-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.cache:
        env["LLAMA_CACHE"] = args.cache

    bin_dir = Path(args.bin_dir)
    llama_completion = bin_dir / "llama-completion"
    aif_probe = bin_dir / "llama-aif-probe"
    aif_report = bin_dir / "llama-aif-report"
    parallel_validator = Path(__file__).with_name("validate_parallel.py")

    common_args = [
        *make_model_args(args),
        "-p", args.prompt,
        "-n", str(args.n_predict),
        "--no-warmup",
        "-fit", "off",
        "-no-cnv",
        *(["--ignore-eos"] if not args.allow_eos else []),
        *args.extra,
    ]

    rows = []
    manifest = {
        "device": args.device,
        "hf": args.hf,
        "model": args.model,
        "prompt": args.prompt,
        "n_predict": args.n_predict,
        "runs": args.runs,
        "post_max": args.post_max,
        "tensor_filter": args.tensor_filter,
        "pcie_mib_s": args.pcie_mib_s,
        "parallel": args.parallel,
        "qkv_schedule": "head_major" if args.parallel else "none",
        "host_budget_mib": args.host_budget_mib,
        "kv_cache_mib": args.kv_cache_mib,
        "host_bandwidth_gbps": args.host_bandwidth_gbps,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "commands": [],
    }

    for run_idx in range(1, args.runs + 1):
        run_dir = root / f"run-{run_idx:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        baseline_log = run_dir / "baseline.log"
        if not args.skip_baseline:
            if args.drop_caches:
                rc = drop_caches(run_dir / "drop-caches-baseline.log", args.dry_run)
                if rc != 0:
                    raise SystemExit(f"drop-caches before baseline run {run_idx} failed with exit code {rc}")

            cmd = [str(llama_completion), *common_args]
            manifest["commands"].append({"run": run_idx, "name": "baseline", "cmd": cmd})
            rc = run_cmd(cmd, baseline_log, env, args.dry_run)
            if rc != 0:
                raise SystemExit(f"baseline run {run_idx} failed with exit code {rc}")

            row = {"run": run_idx, "mode": "baseline", **parse_perf_log(baseline_log)}
            rows.append(row)

        if not args.skip_aif:
            if not args.no_reset:
                reset_log = run_dir / "aif-reset.log"
                cmd = [str(aif_probe), args.device, "reset"]
                manifest["commands"].append({"run": run_idx, "name": "aif-reset", "cmd": cmd})
                rc = run_cmd(cmd, reset_log, env, args.dry_run)
                if rc != 0:
                    raise SystemExit(f"AIF reset for run {run_idx} failed with exit code {rc}")

            stats_before_log = run_dir / "aif-stats-before.log"
            cmd = [str(aif_probe), args.device, "stats"]
            manifest["commands"].append({"run": run_idx, "name": "aif-stats-before", "cmd": cmd})
            rc = run_cmd(cmd, stats_before_log, env, args.dry_run)
            if rc != 0:
                raise SystemExit(f"AIF stats-before for run {run_idx} failed with exit code {rc}")

            aif_csv = run_dir / "aif.csv"
            aif_log = run_dir / "aif.log"
            if args.drop_caches:
                rc = drop_caches(run_dir / "drop-caches-aif.log", args.dry_run)
                if rc != 0:
                    raise SystemExit(f"drop-caches before AIF run {run_idx} failed with exit code {rc}")

            cmd = [
                str(llama_completion),
                *common_args,
                "--aif-dev", args.device,
                "--aif-post-max", str(args.post_max),
                *( ["--aif-parallel",
                    "--aif-host-budget-mib", str(args.host_budget_mib),
                    "--aif-kv-cache-mib", str(args.kv_cache_mib),
                    "--aif-host-bandwidth-gbps", str(args.host_bandwidth_gbps)]
                   if args.parallel else ["--aif-replace-gemv"] ),
                "--aif-replace-gemv-max", "0",
                "--aif-shadow-log", str(aif_csv),
            ]
            if args.tensor_filter:
                cmd.extend(["--aif-tensor-filter", args.tensor_filter])
            if args.aif_all_phases:
                cmd.append("--aif-all-phases")
            if args.aif_log_gemv:
                cmd.append("--aif-log-gemv")

            manifest["commands"].append({"run": run_idx, "name": "aif", "cmd": cmd})
            rc = run_cmd(cmd, aif_log, env, args.dry_run)
            if rc != 0:
                raise SystemExit(f"AIF run {run_idx} failed with exit code {rc}")

            stats_after_log = run_dir / "aif-stats-after.log"
            cmd = [str(aif_probe), args.device, "stats"]
            manifest["commands"].append({"run": run_idx, "name": "aif-stats-after", "cmd": cmd})
            rc = run_cmd(cmd, stats_after_log, env, args.dry_run)
            if rc != 0:
                raise SystemExit(f"AIF stats-after for run {run_idx} failed with exit code {rc}")

            report_log = run_dir / "aif-report.txt"
            cmd = [
                str(aif_report),
                str(aif_csv),
                "--top", str(args.top),
                "--steps", str(args.steps),
                "--aif-log", str(aif_log),
                "--pcie-mib-s", str(args.pcie_mib_s),
            ]
            if baseline_log.exists():
                cmd.extend(["--baseline-log", str(baseline_log)])

            manifest["commands"].append({"run": run_idx, "name": "aif-report", "cmd": cmd})
            rc = run_cmd(cmd, report_log, env, args.dry_run)
            if rc != 0:
                raise SystemExit(f"AIF report for run {run_idx} failed with exit code {rc}")

            if args.parallel:
                validation_log = run_dir / "aif-parallel-validation.log"
                cmd = [sys.executable, str(parallel_validator), str(aif_csv)]
                manifest["commands"].append({"run": run_idx, "name": "aif-parallel-validation", "cmd": cmd})
                rc = run_cmd(cmd, validation_log, env, args.dry_run)
                if rc != 0:
                    raise SystemExit(f"AIF parallel validation for run {run_idx} failed with exit code {rc}")

            row = {
                "run": run_idx,
                "mode": "aif-parallel" if args.parallel else "aif-replace",
                **parse_perf_log(aif_log),
                **parse_aif_report(report_log),
            }
            rows.append(row)

    write_summary(root / "summary.csv", rows)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nAIF bench output: {root}")
    print(f"summary: {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
