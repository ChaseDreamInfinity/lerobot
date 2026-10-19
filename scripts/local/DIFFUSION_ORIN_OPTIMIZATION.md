# Diffusion Policy on the Jetson Orin: speed-up work

How to run the SO-101 Diffusion Policy on the AGX Orin with TensorRT, and what has been measured
so far. The latency log format and its fields are the same as for ACT; see
[ACT_ORIN_OPTIMIZATION.md](ACT_ORIN_OPTIMIZATION.md#reading-the-latency-log).

Checkpoint: `chasedreaminf/diffusion_so101-pick-block-in-cup-2cam-run1`.

## Quick start

```bash
bash scripts/local/orin_max_perf.sh                                    # after every reboot

bash scripts/local/eval_pick_block_diffusion_orin.sh --trt             # TensorRT fp16, DDIM/5, 15 Hz
bash scripts/local/eval_pick_block_diffusion_orin.sh                   # baseline: PyTorch, no TensorRT

NUM_INFERENCE_STEPS=10 bash scripts/local/eval_pick_block_diffusion_orin.sh --trt   # more denoising steps
DURATION=30 bash scripts/local/eval_pick_block_diffusion_orin.sh --trt --dry-run    # print the command only
```

- **First run:** `--trt` builds the engines the first time a checkpoint is used (about 3 min), in
  `outputs/trt_engines/<checkpoint>_orin_fp16/`, and reuses them afterwards.
- **Building separately:** `bash scripts/local/build_diffusion_trt_engine_orin.sh`
  (`POLICY_PATH=…`, `PRECISION=fp16|fp32`).
- **No rebuild for sampler changes:** the sampler (`NOISE_SCHEDULER_TYPE`) and the number of
  denoising steps (`NUM_INFERENCE_STEPS`) can be changed freely.

**Rebuild the engines** after any of these:
- a new checkpoint
- a JetPack or TensorRT upgrade
- a different camera resolution

Mismatched engines are refused when they load, before the robot connects.

## The model

One prediction encodes the last 2 observations, then denoises a 16-step action sequence starting
from random noise. The first 8 actions are executed:

```
2 frames × (wrist, top) ─ ResNet-18 (shared, GroupNorm) ─ spatial softmax ─ 4 × 64 features ─┐
2 frames × arm state (6) ────────────────────────────────────────────────────────────────────┤
                                                                       conditioning (268) ────┤
                                                                                              ▼
random noise (16×6) ─► [ 1D conv UNet, FiLM-conditioned on step + conditioning ] × K steps ─► 16×6 actions
```

- **Observation encoder:** runs once per prediction, over 4 images: 2 cameras × 2 frames. Each
  image becomes 32 keypoints, which is 64 numbers. Together with the two arm states, that gives a
  268-number conditioning vector.
- **UNet:** a 1D temporal conv UNet with channel widths 512 → 1024 → 2048. It predicts the noise to
  remove, and runs once per denoising step: K = `NUM_INFERENCE_STEPS`.
- **Sampler:** trained with DDPM over 100 steps. The eval script samples with DDIM in 5 steps by
  default, which is much faster but further from training.

**Size:**

| Part | Parameters |
|---|---|
| Observation encoder (ResNet-18 + spatial softmax) | 11.2M |
| UNet: down path | 108.0M |
| UNet: middle | 87.2M |
| UNet: up path | 58.8M |
| UNet: step embedding + final conv | 1.4M |
| **Total** | **266.6M**, all used at inference |

**On disk:**
- `model.safetensors` is 1.07 GB (fp32).
- The TensorRT fp16 engines are 23 MB for the encoder and 490 MB for the UNet.

This is about 8× ACT's inference size, and the UNet runs K times per prediction.

## Progress log

### 2026-10-18 — baseline measured

Setup: MAXN power mode with maximum clocks, 15 Hz, DDIM/5, 8 actions per prediction (a new
prediction every 0.53 s). Measured without moving the arm.

| Stage | Time |
|---|---|
| Observation encoder, 4 images | 26.6 ms |
| **One UNet denoising step** | **45.2 ms** |
| One prediction, DDIM/5 | 244 ms |
| One prediction, DDIM/10 / DDIM/20 / DDPM/100 | 462 ms / 898 ms / 4.4 s |
| Tick without a prediction | 4.2 ms |

**Findings:**
- **The arm freezes regularly.** A 244 ms prediction is nearly 4 ticks of the 66.7 ms budget, so
  every 8th tick the arm stops for about a quarter of a second, twice a second.
- **It misses its rate target.** Measured in the real rollout loop it reaches only **11.0 Hz**,
  not 15, with 13% of ticks over budget.
- **Mixed precision barely helps:** 207 ms instead of 244. Same CPU-bound pattern as ACT.
- **First inference is slow:** the warm-up takes 645 ms.

### 2026-10-18 — TensorRT

Two engines: the observation encoder, and one UNet step. The denoising loop and the scheduler
stay in PyTorch.

| Part | PyTorch | TensorRT fp32 | TensorRT fp16 |
|---|---|---|---|
| Observation encoder | 26.6 ms | 15.2 ms | **8.2 ms** |
| One UNet step | 45.2 ms | 6.6 ms | **3.6 ms** (12.7×) |

Full predictions, compared against PyTorch fp32 with the same starting noise:

| Sampler | PyTorch | TensorRT fp16 | Largest difference (normalized) |
|---|---|---|---|
| DDIM/5 | 244 ms | **30.2 ms** (8.1×) | 0.0009 |
| DDIM/10 | 462 ms | **51.5 ms** | 0.0012 |
| DDIM/20 | 898 ms | 94.2 ms | 0.0010 |
| DDPM/100 | 4.4 s | 436 ms | not comparable* |

The fp16 error does not grow with more denoising steps.

\* DDPM adds fresh random noise at every step, so two runs never match, even in fp32. That is why
the load-time self-check compares the encoder and a single UNet step directly instead of a full
prediction.

Real `lerobot-rollout` loop for 11 s, with the real cameras and a stand-in arm (no motors):

| Configuration | Actual rate | Prediction tick | Overruns |
|---|---|---|---|
| PyTorch DDIM/5 @ 15 Hz (baseline) | 11.0 Hz | 253 ms | 13.2% |
| **TensorRT DDIM/5 @ 15 Hz** | **15.0 Hz** | 37.8 ms | **0** |
| TensorRT DDIM/10 @ 15 Hz | 15.0 Hz | 58.2 ms | 0 (budget 66.7 ms) |
| TensorRT DDIM/5 @ 30 Hz | 29.6 Hz | 38.9 ms | 12.5%: about 4 ms over the 33 ms budget |

With TensorRT, warm-up takes 60 ms (was 645 ms). Loading the engines drops the PyTorch UNet,
freeing about 1 GB of GPU memory.

### Next: robot trials (task success not measured yet)

| Configuration | Command | Attempts | Successes | Notes |
|---|---|---|---|---|
| Baseline: PyTorch DDIM/5 @ 15 Hz | `eval_pick_block_diffusion_orin.sh` | | | stalls ~250 ms twice a second |
| TensorRT DDIM/5 @ 15 Hz | `… --trt` | | | same sampler, no stalls |
| TensorRT DDIM/10 @ 15 Hz | `NUM_INFERENCE_STEPS=10 … --trt` | | | closer to the trained sampler |
| TensorRT DDIM/20 @ 10 Hz | `FPS=10 NUM_INFERENCE_STEPS=20 … --trt` | | | 94 ms prediction fits the 100 ms budget |

Remaining speed-up leads:
- **30 Hz:** a prediction tick is about 38 ms, so DDIM/5 at 30 Hz is just over budget. Options:
  - batch the 5 UNet steps into a single engine, removing the per-step Python overhead
    (about 4 ms)
  - run inference asynchronously (RTC), though that needs the policy's observation queues
    reworked first (see the eval script header)
- **Encoder at 8.2 ms:** 4 images at 480×640 through a GroupNorm ResNet. Resizing the inputs would
  need retraining.

## How the speed-up works

- **Two engines in one directory.** `lerobot-rollout --tensorrt_engine=<dir>` loads
  `rgb_encoder.engine` and `unet.engine`:
  - The encoder engine replaces `DiffusionModel._prepare_global_conditioning`.
  - The UNet engine replaces `DiffusionModel.unet`, keeping the same call signature.
  - Observation and action queues, the noise scheduler and pre/post-processing stay in PyTorch.
  - The code is in `src/lerobot/policies/diffusion/tensorrt_diffusion.py`.
- **Shared helpers:** import, ONNX export, engine build and the engine runner are in
  `src/lerobot/utils/tensorrt.py` and shared with ACT.
- **Load-time check:** each engine is compared against the PyTorch model when it loads, as a
  relative error with a 5% limit. Measured fp16 error is 0.0001 for the encoder and about 0.003
  for the UNet; an engine from another checkpoint is far above the limit.
- **Restrictions:** sync inference only, which diffusion needs anyway.
- **Disk:** the ONNX files are about 1 GB for the UNet, so they are built in a temporary
  directory and not kept.

## Tests

`tests/policies/test_diffusion_tensorrt.py` uses a tiny Diffusion Policy and covers:
- the export wrappers match the policy
- build, then a full DDIM prediction through both engines matches PyTorch
- `select_action` runs through the engines
- engines from another checkpoint are refused

It skips without TensorRT or `onnx`. Run it with:

```bash
uv run --no-sync --with pytest --with onnx --with numpy==1.26.4 python -m pytest tests/policies/test_diffusion_tensorrt.py
```
