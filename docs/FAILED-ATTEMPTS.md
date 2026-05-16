# Failed attempts and rejected configurations

This document records every model, framework, and optimisation we tried that did not improve performance, in keeping with reproducibility norms. Each entry lists what we tried, the root cause of failure, and the time invested.

## Models

### Llama 3.3 70B Q40 -- DOES NOT FIT IN MEMORY

- 38 GB total weights, ~9.5 GB per node (4-way tensor parallel).
- 16 GB RAM per node leaves no margin for KV cache once the weights are loaded.
- Observed: continuous swap thrashing, 0.15 tok/s.
- Time invested: 45 minutes.
- Lesson: at Q40 quantisation, ~14 B parameters is the practical upper bound for 4 x 16 GB nodes.

### DeepSeek R1 Distill Llama 8B -- UPSTREAM BUG

- dllama-api crashes on the second inference request when this specific tokenizer is used.
- `dllama inference` (single-node mode) does work; the HTTP API does not.
- We were unable to identify the exact cause in the time allotted.
- Time invested: 20 minutes.
- Workaround: skip; Qwen3 works.

## Frameworks

### EXO (exo-explore/exo) -- REQUIRES APPLE SILICON

- EXO depends on Apple's MLX framework.
- On ARM Linux without MLX, EXO startup fails with `ModuleNotFoundError: No module named 'mlx'`.
- There is no fallback backend; EXO is effectively Apple-Silicon-only despite running on ARM64 generically.
- Time invested: 1 hour.
- Lesson: "ARM" is not a hardware category. Apple Silicon has Metal GPU, Apple Neural Engine and unified memory; Pi 5 has none of these.

### prima.cpp (Lizonghang/prima.cpp) -- HUNGS DURING DISCOVERY

- Same author as HALO. Pipelined-ring parallelism with disk prefetch.
- Cluster bootstrap hangs in the "profiling device" phase for 10+ minutes with all four workers idle (0% CPU).
- ZMQ socket discovery and topology negotiation never completes; the logs offer no failure mode.
- Single-node `prima.cpp` works but slower than dllama (2.14 tok/s).
- Time invested: 1 hour.

### llama.cpp + RPC -- COMMUNITY DOCUMENTED 25x REGRESSION

- Jeff Geerling measured 0.28 tok/s on a 4-Pi cluster vs 6.0 tok/s single-node, a 25x regression.
- The point-to-point RPC protocol is dominated by per-token network overhead.
- Not retested by us; the public evidence was conclusive.
- Reference: https://www.jeffgeerling.com/blog/2025/i-regret-building-3000-pi-ai-cluster/

### HALO (arXiv:2601.11676) -- INAPPLICABLE IN OUR REGIME

- Three combined techniques: semantic-aware predictor (SAP), overlap scheme, PLR-aware scheduler.
- 3.41x speedup -- only at 5% packet-loss rate. In clean LAN (0% PLR), reported speedup is 0.88x ratio TCP/UDP, i.e. ~1.12x.
- SAP requires offline training of a neural network on representative activations.
- Code is not public.
- Estimated re-implementation: 3-6 months.
- Conclusion: not worth pursuing for a clean-LAN cluster.

### Rust frameworks (mistral.rs, candle, cake) -- NO MULTI-NODE ARM CLUSTER

- mistral.rs: GPU/CUDA-focused; no ARM Linux cluster support.
- candle: ARM CPU works single-node but underperforms llama.cpp.
- cake (evilsocket): Rust+Candle. No Pi 5 cluster benchmarks; project less mature.

## Hardware accelerators

### Pi 5 V3D GPU (Vulkan) -- NO COMPUTE SHADERS FOR LLM

- V3D driver implements only render pipeline features.
- LLM workloads need FP16 subgroup operations and substantial shared memory; neither is available.
- Community reports Vulkan compute mode is slower than CPU on Pi 5.

### Hailo-8 NPU (M.2 HAT+) -- VISION-CLASS, NOT TRANSFORMER-CLASS

- Designed for convolutional inference (YOLO, ResNet).
- Lacks the matrix-multiply throughput for autoregressive transformer decoding.
- Approximate cost ~150 USD per node, no LLM benefit.

### CPU overclock 2.7 GHz -- EXCLUDED BY POLICY

- Community evidence suggests +12-15% with active cooling.
- Excluded by operational policy in this deployment.

## Code-level optimisations that regressed throughput

### MSG_ZEROCOPY in writeMany -- REGRESSION

- Added `MSG_ZEROCOPY` flag for `send()` calls >= 32 KB.
- Without a completion-queue handler reading the kernel's `MSG_ERRQUEUE` notifications, the kernel falls back to a copy AND adds a notification round-trip.
- Measured regression: -0.9% throughput.
- Reverted.

### TCP_QUICKACK persistent (re-arm on every recv) -- REGRESSION

- Re-armed `TCP_QUICKACK` before every `recv()` to prevent delayed-ACK between sync rounds.
- The added `setsockopt` syscall per recv (thousands per inference at 48 layers x N batches) exceeded the saving from avoided delayed-ACK.
- Measured regression: -0.9% throughput.
- Reverted.

### Profile-Guided Optimisation (PGO) -- INSTRUMENTATION BREAKS TIMING

- Built dllama with `-fprofile-generate` to collect runtime profile.
- On first request the all-reduce protocol failed with `NnTransferSocketException: Error writing to socket`.
- Hypothesis: instrumentation overhead desynchronises the per-layer barrier timing between workers; one node's collection writes shift it just enough that another node times out.
- Time invested: 45 minutes.
- Conclusion: PGO is incompatible with dllama's tight all-reduce timing in this build.

### `nthreads > 4` -- HARD LIMIT IN DLLAMA

- Setting `--nthreads 5` or higher produces `This configuration supports max 4 threads`.
- The maximum is determined by the model topology, not the SoC core count.

## Sysadmin tunings tested but inconclusive

### Disabling kernel security mitigations (Spectre/Meltdown)

- `mitigations=off` kernel boot parameter.
- Expected gain: 0.5-2.0 tok/s.
- Not applied due to security implications; documented for future consideration on dedicated hardware.

### isolcpus / nohz_full / rcu_nocbs kernel boot parameters

- Expected gain: 0.2-0.8 tok/s reduction in scheduler jitter.
- Not applied because they require reboot and the operator declined.

### PREEMPT_RT kernel

- Expected: lower p99 latency, marginal effect on mean throughput.
- Not pursued; effort/benefit not justified.

### Pi 5 NVMe PCIe Gen 3 force (`dtparam=pciex1_gen=3`)

- Pi 5 ships with NVMe at PCIe Gen 2 (700 MB/s); forcing Gen 3 doubles raw throughput.
- Only affects model load time, not inference throughput (mmap warms the page cache after first run).
- Not applied.
