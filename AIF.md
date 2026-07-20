# AiF llama.cpp Extension

This branch connects llama.cpp to the matching AiF NVMeVirt extension through
Linux NVMe passthrough ioctls. It supports tensor registration, graph-level GEMV
observation or replacement, CSV reporting, and the experimental head/tensor
parallel timing path.

This is a performance simulator. Replacement and parallel modes do not preserve
model output correctness because offloaded GEMV nodes receive dummy outputs.

## Prerequisites

1. Build and load the matching NVMeVirt branch described in its `AIF.md`.
2. Identify the emulated namespace by model name rather than by a fixed device
   number.
3. Ensure the current user can open the namespace read/write, or run only the
   device-accessing commands with sufficient privileges.

```bash
lsblk -d -o NAME,SIZE,MODEL
export AIF_DEVICE=/dev/nvmeXnY
```

Check that both repositories use an identical protocol header:

```bash
cd /path/to/llama.cpp
./tools/aif-bench/check_protocol.py \
  --nvmevirt-header /path/to/nvmevirt/aif.h
```

The final line must be `AIF protocol headers: MATCH`.

## Build

```bash
cd /path/to/llama.cpp
cmake -B build -DGGML_NATIVE=ON
cmake --build build -j"$(nproc)"
```

Verify device access and the profile:

```bash
./build/bin/llama-aif-probe "$AIF_DEVICE" stats
```

llama.cpp refuses to start the AIF path when the protocol version differs or
when the device does not report the expected 8-channel, 16-chip, 4-plane,
16 KiB-page profile.

## Execution Modes

`--aif-dev` enables model-load-time `AIF_OP_POST`. The default tensor filter
selects Q/K/V projections, attention output projection, the three FFN matrices,
and `output.weight` when present.

- Shadow mode sends matching GEMVs to NVMeVirt but retains normal backend
  computation. Use it to inspect command coverage without replacing nodes.
- Replacement mode sends matching decode GEMVs to NVMeVirt, waits for completion,
  and fills the GGML output tensor with zeros.
- Parallel mode additionally models head-major QKV/attention overlap and splits
  eligible FFN rows between the host and AiF according to bandwidth and the host
  memory budget.

Example replacement run:

```bash
./build/bin/llama-completion \
  -m /path/to/model.gguf \
  -p "Once upon a time," -n 32 \
  --no-warmup -fit off -no-cnv --ignore-eos \
  --aif-dev "$AIF_DEVICE" \
  --aif-post-max 0 \
  --aif-replace-gemv \
  --aif-replace-gemv-max 0 \
  --aif-shadow-log /tmp/aif.csv
```

The generated text is not meaningful in replacement or parallel mode. Use the
timing counters and logs, not model accuracy, as the output of the experiment.

## Reproducible Benchmark

The benchmark requires `--device` or the `AIF_DEVICE` environment variable:

```bash
cd /path/to/llama.cpp
python3 tools/aif-bench/aif_bench.py \
  --device "$AIF_DEVICE" \
  --model /path/to/model.gguf \
  --runs 3 -n 64 \
  --parallel \
  --host-budget-mib 8192 \
  --kv-cache-mib 1024 \
  --out-dir ./aif-results/model-n64-r3
```

Each run resets the AIF state unless `--no-reset` is passed. The output directory
contains `summary.csv`, `manifest.json`, per-run llama logs, `aif.csv`, before/after
device stats, reports, and parallel validation output.

Use `--drop-caches` only when intentionally measuring cold page-cache behavior;
it requires root and affects the whole machine.

## Main Code Locations

- `src/llama-aif-proto.h`: shared AIF wire protocol and ABI checks.
- `common/aif-client.*`: NVMe namespace open, NSID lookup, and ioctl wrapper.
- `common/common.cpp`: tensor posting, GEMV payloads, graph callbacks, async device
  worker, head-major QKV timing, FFN split, and CSV logging.
- `ggml/include/ggml-backend.h` and `ggml/src/ggml-backend.cpp`: scheduler node
  override callback.
- `tools/aif-probe`: C++ device probe.
- `tools/aif-report`: CSV aggregation and timing report.
- `tools/aif-bench`: repeated baseline/AIF runner and validators.

## Interpretation Limits

- NVMeVirt records a physical layout but does not store actual model weight bytes.
- AIF GEMV returns deterministic dummy data, and llama.cpp writes zero-valued node
  outputs in replacement/parallel mode.
- The host attention interval in parallel mode is measured from the existing GGML
  graph and partitioned into a head-level timing model; attention is not physically
  executed as one independent host kernel per head.
- QKV commands remain full-tensor ioctls. Head-major readiness is a modeled
  timeline derived from their device service times.
- Per-GEMV stats collection and CSV logging add control-path overhead. Modeled
  device time and measured end-to-end time must be reported separately.
- Baseline results still depend on CPU configuration, threading, mmap/page-cache
  state, quantization, prompt length, and model version. Record `manifest.json`
  and the machine configuration with every reported result.
