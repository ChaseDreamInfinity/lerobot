# Diffusion Policy on the Jetson Orin: speed-up work and findings

How to run the SO-101 Diffusion Policy on the AGX Orin with TensorRT, what has been measured, and
what has been learned about the checkpoint's behaviour on the robot. The latency log format and
its fields are the same as for ACT; see
[ACT_ORIN_OPTIMIZATION.md](ACT_ORIN_OPTIMIZATION.md#reading-the-latency-log).

Checkpoint: `chasedreaminf/diffusion_so101-pick-block-in-cup-2cam-run1`.

## Quick start

```bash
bash scripts/local/orin_max_perf.sh                                    # after every reboot

bash scripts/local/eval_pick_block_diffusion_orin.sh --trt             # TensorRT fp16, DDIM/5, 30 Hz
bash scripts/local/eval_pick_block_diffusion_orin.sh                   # PyTorch: cannot keep up at 30 Hz

DURATION=30 bash scripts/local/eval_pick_block_diffusion_orin.sh --trt --dry-run    # print the command only
```

- **Control rate:** the script runs at **30 Hz** by default, the rate the training data was
  recorded at. Running slower makes the policy hesitate; see the findings below. Only `--trt` keeps
  up at 30 Hz.
- **First run:** `--trt` builds the engines the first time a checkpoint is used (about 3 min), in
  `outputs/trt_engines/<checkpoint>_orin_fp16/`, and reuses them afterwards.
- **Building separately:** `bash scripts/local/build_diffusion_trt_engine_orin.sh`
  (`POLICY_PATH=…`, `PRECISION=fp16|fp32`).
- **No rebuild for sampler changes:** the sampler (`NOISE_SCHEDULER_TYPE`) and the number of
  denoising steps (`NUM_INFERENCE_STEPS`) can be changed freely. At 30 Hz, only DDIM/5 fits:
  DDIM/10 takes about 45 ms per prediction tick.

**Rebuild the engines** after any of these:
- a new checkpoint
- a JetPack or TensorRT upgrade
- a different camera resolution
- an engine directory from before 2026-10-18 22:00, which holds `rgb_encoder.engine` instead of
  `frame_encoder.engine`; `--trt` rebuilds it automatically

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

- **Observation encoder:** 4 images per prediction: 2 cameras × 2 frames. Each image becomes 32
  keypoints, which is 64 numbers. With the two arm states, that gives a 268-number conditioning
  vector, 134 per frame.
- **UNet:** a 1D temporal conv UNet with channel widths 512 → 1024 → 2048. It predicts the noise to
  remove, and runs once per denoising step: K = `NUM_INFERENCE_STEPS`.
- **Sampler:** trained with DDPM over 100 steps; the eval script samples with DDIM in 5 steps. On
  replayed training data, DDIM/5 is about as accurate as DDPM/100 (see the findings).
- **Timing:** each action is one 1/30 s step, the dataset's recording rate.

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
- The TensorRT fp16 engines are 24 MB for the frame encoder and 513 MB for the UNet.

This is about 8× ACT's inference size, and the UNet runs K times per prediction.

**Training:**
- 200k steps at batch 16, on `chasedreaminf/record-pick-block-in-cup-2cam-run1_20260505_191315`.
- No image augmentation: `crop_shape` is unset and dataset image transforms are disabled.
- See the dataset notes under the findings.

## Progress log

### 2026-10-18 — baseline measured

Setup: MAXN power mode with maximum clocks, 15 Hz (the script's default at the time), DDIM/5,
8 actions per prediction. Measured without moving the arm.

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
| Observation encoder, 4 images | 26.6 ms | 15.2 ms | **8.2 ms** |
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
| TensorRT DDIM/5 @ 15 Hz | 15.0 Hz | 37.8 ms | 0 |
| TensorRT DDIM/10 @ 15 Hz | 15.0 Hz | 58.2 ms | 0 (budget 66.7 ms) |
| TensorRT DDIM/5 @ 30 Hz | 29.6 Hz | 38.9 ms | 12.5%: about 4 ms over the 33 ms budget |

With TensorRT, warm-up takes 60 ms (was 645 ms). Loading the engines drops the PyTorch UNet,
freeing about 1 GB of GPU memory.

### 2026-10-18 — on the real arm: fast enough, but no attempt at the block

TensorRT DDIM/10 at 15 Hz on the real arm: steady 15 Hz, 0 overruns, slowest tick 60.7 ms of 66.7.
But **the arm did not try to grab the block.**

**What was ruled out**, all without moving the arm:

- **Cameras.** The USB ports match the scripts: wrist = Microdia `0c45:6366` on 4.4, top = ARC
  `05a3:9230` on 4.1. Live frames match training frames in role, framing, lighting and arm pose.
- **TensorRT and the sampler.** Replay test on training data: 48 points across 8 episodes,
  comparing the 8 predicted actions with what the demonstrator did. Mean error in joint units:

  | | Mean error | First action |
  |---|---|---|
  | Holding still (floor) | 2.89 | — |
  | PyTorch DDPM/100 (trained sampler) | 2.13 | 1.45 |
  | PyTorch DDIM/10 | 2.15 | 1.47 |
  | PyTorch DDIM/5 | 2.19 | 1.51 |
  | TensorRT DDIM/10 | 2.13 | 1.47 |
  | TensorRT DDIM/5 | 2.19 | 1.46 |

  TensorRT matches PyTorch, and DDIM/5 is about as good as DDPM/100. Neither is the problem.

**Cause: control rate.** The dataset was recorded at **30 Hz**, but the script ran at 15 Hz:
- **Slow motion:** each predicted action is one 1/30 s step, so at 15 Hz the plan plays back at
  half speed.
- **Unfamiliar history:** the 2-frame observation history spans twice the time the model saw in
  training.
- **Effect:** every prediction starts from a situation unlike any in training: the arm is slower
  and further behind than expected. The 15 Hz default dated from when eager inference could not
  keep up.

**At 30 Hz (TensorRT DDIM/5), the arm goes for the block but cannot locate it well, and does not
pick it.** Remaining suspects, most likely first:

1. **No image augmentation in training.** Diffusion Policy's standard recipe trains on random
   crops. Without them the model can match the training pixels closely yet aim poorly when the
   scene is slightly different. Consistent with this, on training data it is only about 26%
   better than holding still.
2. **The task is ambiguous to the policy.** The dataset has three tasks: brown (82 episodes), blue
   (62) and green (62). Later episodes have 2–3 blocks of different colours on the mat. Diffusion
   Policy does not read the task text, so it learned an average over different targets.
3. **Residual timing.** At that point the prediction tick was about 40 ms against 33, so the loop
   averaged about 27 Hz. Fixed in the next entry.

ACT was trained on the same dataset, also without augmentation or language. Running it on the
same scene separates checkpoint-specific problems from dataset or scene problems.

### 2026-10-18 — prediction tick fitted into 30 Hz

**Where the time went:** the prediction tick took 40.5 ms against a 33.3 ms budget:

| Stage | Time |
|---|---|
| Image conversion (`prepare_observation_for_inference`) | 2.6 ms |
| Pre-processing | 1.2 ms |
| Encoder engine, 4 images | 9.1 ms |
| UNet engine × 5 | 4.25 ms each (3.56 ms of GPU time + overhead) |
| Scheduler step × 5 | 0.71 ms each, Python |
| Post-processing | 0.6 ms |

**Three changes:**
1. **CUDA graph of the denoising loop.** The 5 UNet engine calls and the DDIM scheduler math are
   captured once and replayed as one launch. 21.3 → 18.0 ms for the loop, which is the GPU's own
   time. Still matches PyTorch fp32 to within 0.006.
2. **Each frame is encoded once.** The encoder engine now takes one frame, both cameras. Each
   frame's encoding is cached while the frame is in the history. The tick before a prediction
   encodes its newest frame, which becomes the "previous frame" of the prediction, so the
   prediction tick encodes only the current frame. Joining per-frame encodings matches encoding
   the full history to within 2.5e-5.
3. **Image conversion on the GPU.** `prepare_observation_for_inference` now copies the 8-bit image
   to the GPU and converts it there, instead of converting to float on the CPU and copying 4×
   the bytes. 3.1 → 1.1 ms per tick. This helps every policy, ACT included. The values differ by
   at most 6e-8 (half a float32 rounding step), because the GPU divides by 255 by multiplying by
   1/255.

**Results**, DDIM/5, per tick:

| Tick | Before | After |
|---|---|---|
| Prediction tick | 40.5 ms | **25.1 ms** |
| Tick before a prediction | ~4 ms | 7.2 ms (encodes a frame ahead) |
| Other ticks | 4.2 ms | **2.3 ms** |

Real `lerobot-rollout` loop at **30 Hz** for 16 s, with the real cameras and a stand-in arm:
**29.9 Hz, 0 overruns, slowest tick 27.3 ms.** The real arm adds about 2 ms for the motors, which
leaves about 4 ms of margin. DDIM/10 now takes about 45 ms per prediction tick: fine at 15 Hz,
but not at 30.

**Confirmed on the real arm** (TensorRT DDIM/5 at 30 Hz), two 5 s windows:

| Window | Rate | `obs` | `policy` mean/p99/max | `send` | `loop` mean/p99/max | Overruns |
|---|---|---|---|---|---|---|
| 1 | 30.1 Hz | 1.3/1.4/1.4 | 6.2/26.1/26.2 | 0.4/0.5/0.5 | 7.9/28.1/28.2 | 0 |
| 2 | 29.9 Hz | 1.3/1.5/1.5 | 6.2/25.7/26.0 | 0.4/0.5/0.5 | 7.9/27.6/28.0 | 0 |

The slowest tick is 28.2 ms, about 5 ms under the 33.3 ms budget.

**Behaviour: still no pick.** At a true 30 Hz with no overruns, the arm still reaches without
locating or picking the block. That rules out timing as the cause of the poor aim: the remaining
suspects are the checkpoint and dataset (no augmentation, mixed block colours; see the previous
entry).

### Next

**Robot trials at 30 Hz** (task success not measured yet):

| Configuration | Command | Attempts | Successes | Notes |
|---|---|---|---|---|
| TensorRT DDIM/5 @ 30 Hz | `eval_pick_block_diffusion_orin.sh --trt` | 2 | 0 | reaches but misses, before and after the timing fix |
| ACT on the same scene, as a control | `eval_pick_block_act_orin.sh --trt` | | | same dataset |

**Retraining**, which is the likely fix for poor localization:
- **Diffusion with random-crop augmentation,** e.g. `--policy.crop_ratio=0.9`, with
  `crop_is_random` left at its default. Train on the brown-block episodes only, or on a scene with
  one block.
- **Or a policy that reads the task text,** such as SmolVLA or pi0.5, both of which have eval
  scripts here.

## How the speed-up works

- **Two engines in one directory.** `lerobot-rollout --tensorrt_engine=<dir>` loads
  `frame_encoder.engine` and `unet.engine`:
  - The frame encoder replaces the image + state encoding, one frame at a time, with a cache.
  - The UNet engine replaces `DiffusionModel.unet`, keeping the same call signature.
  - `DiffusionModel.conditional_sample` is replaced by a CUDA-graph replay, which falls back to
    the eager loop if capture fails.
  - Observation and action queues, the noise scheduler and pre/post-processing stay in PyTorch.
  - The code is in `src/lerobot/policies/diffusion/tensorrt_diffusion.py`.
- **Graph is fixed at load.** The graph is captured for the sampler and step count configured
  when the engines load. Changing `NUM_INFERENCE_STEPS` needs a restart, not a rebuild.
- **Shared helpers:** import, ONNX export, engine build and the engine runner are in
  `src/lerobot/utils/tensorrt.py` and shared with ACT.
- **Load-time check:** each engine is compared against the PyTorch model when it loads, as a
  relative error with a 5% limit. Measured fp16 error is 0.0002 for the encoder and about 0.002
  for the UNet; an engine from another checkpoint is far above the limit.
- **Restrictions:** sync inference only, which diffusion needs anyway.
- **Disk:** the ONNX files are about 1 GB for the UNet, so they are built in a temporary
  directory and not kept.

## Tests

`tests/policies/test_diffusion_tensorrt.py` uses a tiny Diffusion Policy and covers:
- the export wrappers match the policy
- build, then a full DDIM prediction through both engines and the CUDA graph matches PyTorch
- the cached per-frame path matches PyTorch as the observation history moves on
- `select_action` runs through the engines
- engines from another checkpoint are refused

It skips without TensorRT or `onnx`. Tolerances allow for TensorRT using TF32 in "fp32" engines.
Run it with:

```bash
uv run --no-sync --with pytest --with onnx --with numpy==1.26.4 python -m pytest tests/policies/test_diffusion_tensorrt.py
```
