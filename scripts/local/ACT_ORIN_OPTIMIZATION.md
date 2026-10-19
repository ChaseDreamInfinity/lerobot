# ACT on the Jetson Orin: speed-up work and latency logs

How to run the SO-101 ACT policy on the AGX Orin with the TensorRT speed-up, how to read the
latency log, and what has been measured so far.

Checkpoint: `chasedreaminf/act_so101-pick-block-in-cup-2cam-run1`. The model is ResNet-18 + a
transformer, 100-action chunks, and it reads two 640×480 cameras plus the arm state.

## Quick start

```bash
bash scripts/local/orin_max_perf.sh                             # after every reboot

bash scripts/local/eval_pick_block_act_orin.sh --trt            # TensorRT fp16 (recommended)
bash scripts/local/eval_pick_block_act_orin.sh                  # baseline: PyTorch, no TensorRT

N_ACTION_STEPS=25 bash scripts/local/eval_pick_block_act_orin.sh --trt   # shorter chunks
DURATION=30 bash scripts/local/eval_pick_block_act_orin.sh --trt --dry-run   # print the command only
```

- **First run:** `--trt` builds the engine the first time a checkpoint is used (about 80 s), at
  `outputs/trt_engines/<checkpoint>_orin_fp16.engine`, and reuses it afterwards.
- **Which path is running:** the startup line `Forward pass: …` says whether TensorRT or plain
  PyTorch is in use.
- **Building separately:** you can also build an engine on its own with
  `bash scripts/local/build_act_trt_engine_orin.sh` (`POLICY_PATH=…`, `PRECISION=fp16|fp32`).

**Rebuild the engine** after any of these:
- a new checkpoint
- a JetPack or TensorRT upgrade
- a different camera resolution

A stale or mismatched engine is refused when it loads, before the robot connects.

## Reading the latency log

Every 5 s while the robot runs (`LATENCY_LOG_INTERVAL_S`, `0` turns it off):

```
Latency over 5s: 150 ticks, 29.9 Hz (target 30) | obs 0.1/0.1/0.1 | process 0.0/0.1/0.1 | policy 5.3/9.9/11.0 | send 0.1/0.2/0.2 | loop 5.6/10.2/11.3 ms (mean/p99/max) | overruns 0 (budget 33.3 ms)
```

At the end of the run, one summary line for the whole run (mean/max only):

```
Latency for the whole run (120s): 3598 ticks, 30.0 Hz (target 30) | … ms (mean/max) | overruns 0 (0.0%)
```

Each control tick does the stages below in order. Times are in milliseconds, written as
`mean/p99/max`: the average, the value 99% of ticks stay under, and the slowest tick.

| Field | What it times | Typical on this rig |
|---|---|---|
| `obs` | Reading the 6 motor positions over USB, plus taking the latest frame from each camera | ~1 ms |
| `process` | Cleaning up the observation before it reaches the policy | ~0 ms |
| `policy` | ACT itself: image prep, the network, turning its output into joint targets | ~5 ms on most ticks; on ticks that compute a new chunk, ~10 ms with TensorRT vs ~42 ms without |
| `send` | Writing the joint targets to the motors over USB | ~1 ms |
| `loop` | All of the above: the total work for one tick | ~7 ms |
| `Hz` | Ticks actually run per second | should match the target (30) |
| `overruns` | Ticks over the 33.3 ms budget at 30 Hz; the arm waits for its next command | should be 0 |

**How to read it:**
- **Healthy:** if `loop` max stays under 33 ms and `overruns` is 0, the arm gets a fresh command
  on every tick.
- **Finding the cause of overruns:** look at which stage has the large `max`. A large `policy`
  means the model is too slow; a large `obs` means a camera or the motor bus is lagging.

The motors and cameras are read in `obs`, so a camera dropout shows up there.

To keep a copy of the log: `bash scripts/local/eval_pick_block_act_orin.sh --trt 2>&1 | tee outputs/act_trt_orin.log`.

In plain `lerobot-rollout` the flag is `--latency_log_interval_s=5`. It is off by default, and only
the base strategy, which the eval script uses, reports it.

## The model

One prediction turns two camera images and the arm state into a chunk of 100 future joint
targets (100×6):

```
wrist image ─┐                                   1 latent token (zeros)
             ├─ ResNet-18 ─ 15×20 map ─ 300 tokens ─┐   1 arm-state token
top image  ──┘   (shared)   per camera   per camera  │        │
                                                     ▼        ▼
                         602 tokens ─ transformer encoder (4 layers, width 512)
                                                     │
               100 action queries ─ transformer decoder (1 layer) ─ linear head ─ 100×6 actions
```

- **Image embedding:** one ResNet-18 processes both cameras, one image at a time. Each 480×640
  image becomes a 15×20 feature map, which is projected to 512 dims: 300 tokens per camera.
- **Encoder:** 4 layers over 602 tokens: 1 latent + 1 arm state + 2 × 300 image tokens, plus
  position embeddings.
- **Decoder:** 1 layer. Its 100 query tokens, one per future action, attend to the encoder output.
- **Action head:** a linear layer from each decoder output to 6 joint targets.
- **VAE encoder, training only:** it encodes a demonstrated action sequence into a style latent.
  At inference there is no demonstration to encode, so it is skipped and the latent is zeros.

**Size**, from the checkpoint:

| Part | Parameters | Used at inference |
|---|---|---|
| ResNet-18 backbone | 11.2M | yes |
| Transformer encoder (4 layers) | 17.3M | yes |
| Transformer decoder (1 layer) | 5.4M | yes |
| Projections, embeddings, action head | 0.4M | yes |
| VAE encoder | 17.4M | no |
| **Total** | **51.6M** | **~34.2M** |

**On disk:**
- `model.safetensors` is 207 MB: all 51.6M parameters in fp32.
- The TensorRT fp16 engine is 70 MB: only the inference parts, mostly in fp16.

That is about 1/100 the size of a VLA such as pi0.5 (around 3B parameters).

**Where the time goes:** in plain PyTorch, about 17 ms of the 36.5 ms forward pass is the two
ResNet passes and about 20 ms is the transformer. The transformer is small, so most of its time
on the Orin is the CPU launching many small GPU operations rather than computing. TensorRT fuses
the whole model, including the ResNet, encoder, decoder and head, into one engine, which brings
it to 4.8 ms.

## Progress log

### 2026-10-18 — baseline measured

Setup: MAXN power mode with maximum clocks. The checkpoint as trained runs 100 actions per
prediction with no temporal ensembling, at 30 Hz.

**Where the time goes** (measured without moving the arm):

| Stage | Time |
|---|---|
| Motor position read | 1.0 ms (max 1.6) |
| Camera frame read | ~0 ms; frames are ~17 ms old on average |
| Policy, tick that only replays a stored action | 4.5 ms |
| Policy, tick that computes a new chunk (1 in 100) | **42 ms**, over the 33 ms budget |
| ↳ of which the network itself | 36.5 ms: ResNet backbones 17 ms, transformer ~20 ms |

**Findings:**
- **The network is CPU-bound, not GPU-bound.** About 34 ms of CPU time per run, so the GPU mostly
  waits. Mixed precision (AMP) changed nothing (37.1 ms), and neither did `cudnn.benchmark`.
- **One overrun every 3.3 s:** the tick that computes a new chunk takes about 42 ms.
- **The arm runs blind for 3.3 s at a time.** It replays all 100 actions without looking at the
  scene again. This is probably the main limit on task success, more than speed.
- **First inference is slow without TensorRT:** about 368 ms while CUDA warms up, so the arm
  stalls at the start of every run (seen in the latency log). Fixed by the warm-up below.

### 2026-10-18 — speed-ups tried

Network time only, same inputs (two 480×640 images + state → 100×6 action chunk):

| Option | Time | Speed-up | Accuracy vs. baseline | Result |
|---|---|---|---|---|
| PyTorch fp32 (baseline) | 36.5 ms | 1× | — | |
| + AMP fp16 | 37.1 ms | none | | Rejected: CPU-bound |
| + token-concat cleanup in `modeling_act.py` | 34.3 ms | 1.06× | bit-exact | Kept as cleanup |
| `torch.compile` | — | — | — | Not possible: no Triton on Jetson |
| Manual CUDA graph, fp32 | 25.7 ms | 1.4× | exact | |
| Manual CUDA graph + fp16 | 16.1 ms | 2.3× | max 0.001 | |
| Manual CUDA graph + bf16 | 17.6 ms | 2.1× | max 0.007 | |
| TensorRT fp32 | 10.7 ms | 3.4× | max 0.003 | |
| **TensorRT fp16** | **4.8 ms** | **7.6×** | max 0.009, mean 0.0008 | **Adopted** |

Accuracy is measured in normalized action units. After converting to joint units (range ±100),
the TensorRT fp16 worst case over a whole run is **0.08**, about one encoder step on these
servos. At most 0.36 in the per-joint worst case on random inputs.

### 2026-10-18 — TensorRT in the rollout

Per-tick cost through the real pre- and post-processing:

| Schedule | PyTorch | TensorRT fp16 |
|---|---|---|
| 100-action chunks (checkpoint default), tick that computes a new chunk | 42 ms, over budget | **9.3 ms** |
| Temporal ensembling, new prediction every tick | 40 ms every tick, cannot reach 30 Hz | **10 ms** (p99 12.6) |

Real `lerobot-rollout` loop for 12 s, with the real cameras and a stand-in arm (no motors):

| | Rate | `policy` mean / max | Overruns |
|---|---|---|---|
| PyTorch | 29.0 Hz | 7.0 / 368 ms | 4 (1.1%) |
| TensorRT fp16 | 29.9 Hz | 5.5 / 25 ms | 0 |

### 2026-10-18 — first run on the real arm (TensorRT fp16, 100-action chunks)

Two consecutive 5 s windows from the live rollout:

| Window | Rate | `obs` | `policy` mean/p99/max | `send` | `loop` mean/p99/max | Overruns |
|---|---|---|---|---|---|---|
| 1 | 30.1 Hz | 1.3/1.5/1.5 | 5.1/9.9/24.6 | 0.5/0.6/0.6 | 7.0/11.8/26.6 | 0 |
| 2 | 29.9 Hz | 1.3/1.5/1.5 | 5.2/9.8/11.0 | 0.5/0.7/0.7 | 7.1/11.7/12.9 | 0 |

These match the offline measurements:
- **On target:** steady 30 Hz with no overruns. The worst tick uses 39% of the 33 ms budget in
  window 2.
- **Real hardware adds little:** reading the motors and cameras costs 1.3 ms and sending the action
  0.5 ms.
- **First-window spike:** the 24.6 ms `policy` max in window 1 was the first tick paying one-time
  costs. Fixed in the next entry.

### 2026-10-18 — warm-up before the control loop

**Cause of the spike:** the first tick (22 ms vs ~5–7 ms afterwards) was not the engine, which the
load-time check has already run. The cost came from first calls through pre-processing (6.5 ms),
the policy call (6.2 ms) and post-processing (5.6 ms).

**Fix:** `SyncInferenceEngine.start()` now sends one all-zero observation through the full
pipeline before the loop starts, then resets the policy so nothing from it is executed. It logs
`Inference warm-up done in … ms`. This applies to every policy run with sync inference, not just
TensorRT.

Real `lerobot-rollout` loop for 11 s, with the real cameras and a stand-in arm:

| | Warm-up (at startup) | First window `policy` max | Overruns over the whole run |
|---|---|---|---|
| TensorRT fp16 | 25 ms | **11.6 ms** (was 24.6) | 0 |
| PyTorch | 366 ms | 42.7 ms (was 368) | 4 (1.2%): the usual tick that computes a new chunk |

### Next: robot trials (task success not measured yet)

The speed-up makes these schedules possible at 30 Hz. Success rate still has to be measured with
supervised runs, resetting the block and cup between attempts:

| Schedule | Command | Attempts | Successes | Notes |
|---|---|---|---|---|
| Baseline: PyTorch, 100-action chunks | `eval_pick_block_act_orin.sh` | | | |
| TensorRT, 100-action chunks | `… --trt` | | | should match the baseline |
| TensorRT, 50-action chunks | `N_ACTION_STEPS=50 … --trt` | | | |
| TensorRT, 25-action chunks | `N_ACTION_STEPS=25 … --trt` | | | |
| TensorRT, temporal ensembling (0.01) | needs a script setting (not added yet) | | | the original ACT recipe |

Remaining speed-up lead:
- **Image prep:** ~2.5 ms of the ~5 ms `policy` time on ordinary ticks is image preparation that
  runs even when the result is not used. Running it only on ticks that compute a new chunk would
  bring ordinary ticks to ~2 ms.

## How the speed-up works

- **Engine replaces the network only.** `lerobot-rollout --tensorrt_engine=<path>` swaps out the
  network call (`ACTPolicy.predict_action_chunk`). Pre- and post-processing, the action queue and
  temporal ensembling stay in PyTorch, so behaviour differs only by fp16 rounding. The code is in
  `src/lerobot/policies/act/tensorrt_act.py`.
- **Load-time check:** when the engine loads, it is compared against the PyTorch model on random
  inputs. If they differ by more than 0.05 (normalized), it is refused: that means the wrong
  checkpoint or a broken build.
- **Restrictions:** ACT only, CUDA only, sync inference only. It refuses `--use_torch_compile`.
- **Input size check:** an engine has fixed input sizes, so a camera resolution that does not
  match the checkpoint gives a clear error.
- **No extra installs:** TensorRT 10.3 comes with JetPack, under
  `/usr/lib/python3.10/dist-packages`. The code imports only `tensorrt` from there, so the
  project's own packages are not affected.

## Environment notes (Jetson)

- **Never run `uv sync` on this box.** It replaces the CUDA build of torch with a CPU-only one
  (see `_cuda_guard.sh`).
- **Tools that are not in `.venv`:** run them with `uv run --no-sync --with …`. Always pin numpy,
  because an unpinned overlay pulls numpy 2.x, which breaks the Jetson torch build:

  ```bash
  uv run --no-sync --with pytest --with numpy==1.26.4 python -m pytest tests/policies/test_act_tensorrt.py tests/test_rollout.py
  ```

  The engine build script already does this for `onnx`.
- **Tests:**
  - `tests/policies/test_act_tensorrt.py`: export, build, load, and refusing a mismatched engine.
    It skips when TensorRT or `onnx` is missing.
  - `tests/test_rollout.py`: covers the latency monitor.
  - The ACT backward-compatibility test only runs on x86_64. The `modeling_act.py` change was
    checked bit-exact against the original here, but should be confirmed there once.
