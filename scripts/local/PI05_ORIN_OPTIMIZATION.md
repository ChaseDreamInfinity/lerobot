# Pi-0.5 on the Jetson Orin: speed-up work and findings

The pi-0.5 (`pi05`) pick-block policy on the SO-101, run from a Jetson AGX Orin 64 GB (JetPack
6.2, MAXN). Companion to `SMOLVLA_ORIN_OPTIMIZATION.md`: the same approach (CUDA graphs around the
KV fill and denoising step, then TensorRT engines), applied to a model 5× larger.

## Quick start

```bash
bash scripts/local/orin_max_perf.sh                       # once per boot
bash scripts/local/eval_pick_block_pi05_orin.sh --trt     # builds the engines on first use (~4 min)
bash scripts/local/eval_pick_block_pi05_orin.sh --help    # every override
```

| Mode | Startup | Prediction (real loop) | RTC delay at 30 Hz | Actions vs fp32 |
|---|---|---|---|---|
| Before (eager, 10 Hz script) | 213 s | ~1.2 s | (13 ticks at 10 Hz) | bf16 |
| Eager | 7 s | ~1.2 s | ~35 ticks: the queue runs dry at 30 Hz | bf16 |
| `--cuda-graph` | ~35 s | ~0.52 s | 16 ticks, 0 overruns | bf16 (TF32 images) |
| **`--trt`** | ~60 s | **~0.26 s** | **8 ticks**, 0 overruns | **~2× closer than bf16** |

## The model

- **PaliGemma** (SigLIP so400m vision tower, Gemma 2B, 18 layers) plus a **Gemma 300M action
  expert** (18 layers). 9.35 GB checkpoint, 9.4 GB in memory: bf16, with the vision tower,
  projector and norms kept in fp32.
- **Inputs:** three 224×224 image slots, 256 tokens each. Two hold cameras (top → `base_0_rgb`,
  wrist → `left_wrist_0_rgb`); the third (`right_wrist_0_rgb`) is padded with `-1` and masked
  out. Plus 200 language tokens, so the prefix is 968 tokens. The state is tokenized into the
  prompt.
- **Output:** a chunk of 50 actions from 10 flow-matching steps.
- **Training:** 60k steps at batch 4, `train_expert_only=true`, image augmentation on.
  Fine-tuned from `lerobot/pi05_base` on the same 3-color dataset as SmolVLA.

## Progress log

### 2026-10-19 — the "crash" was not memory

The memory: the eval script "crashes whenever run". On a Jetson, CPU and GPU share 61 GB, and an
out-of-memory spiral freezes the board, so every run here went through a watchdog that kills the
process if available memory drops below 8 GB.

- **Offline load and predictions:** no crash. The lowest point was 33 GB available, 9.5 GB on
  the GPU.
- **The script's exact configuration** through the real `lerobot-rollout` loop, with the real
  cameras and a stand-in arm (no motor bus): 90 s at 10 Hz, 0 overruns, clean shutdown.
- **No trace in the kernel log** (no OOM kills). The journal only keeps the current boot.

Likely causes, neither reproducible now:
- **Camera dropouts:** the script's own header said "cameras drop within ~15s". That predates
  the dropout fix (`ceadc2d1`).
- **The 213 s load,** which looks like a hang.

### 2026-10-19 — loading: 213 s → 7 s, bit-identical

**207 s of the 213 s load** was spent randomly initializing ~3.6B fp32 parameters on the Orin's
CPU, all of which the checkpoint then overwrites. Reading the weights (mmap) takes 0.1 s and
`load_state_dict` 1.7 s.

`PI05Policy.from_pretrained` now builds the model under transformers' `no_init_weights()`, but
only once the weights file has resolved. If the checkpoint leaves any parameter out (or has
extra keys), it rebuilds the model with initialization and loads the original way, so
uninitialized memory is never returned. Verified against the original load on the real
checkpoint: all 819 parameters and buffers bit-identical, and the same sampled actions.

### 2026-10-19 — baseline: where the 1.15 s goes

| Stage | Eager | GPU-bound? |
|---|---|---|
| Image encoding (3 slots, SigLIP in fp32) | 241 ms | yes: ~80 ms per image |
| KV fill (Gemma 2B over 968 tokens, ~3.8 TFLOP) | 296 ms | yes: ~13 TFLOPS |
| 10 denoising steps | 10 × 60 ms | **no**: launch-bound, ~5 ms of weight reads per step |

The profiler can't read GPU counters here (CUPTI needs elevated privileges), so "GPU-bound" is
inferred from FLOP and byte counts.

### 2026-10-19 — CUDA graphs: 1.15 s → 0.59 s (bit-identical), 0.50 s with TF32 images

`lerobot.policies.pi05.cuda_graph_pi05` is SmolVLA's graphed sampler adapted to pi-0.5 (shared
helpers in `lerobot.utils.cuda_graph`):
- **Two graphs, captured at startup:** the prefix KV fill, and one denoising step reading that
  cache. The denoising loop stays in Python, so RTC guidance works unchanged.
- **Empty camera slot:** its embedding (the constant `-1` image) is computed once and cached.
- **Host sync removed:** `embed_suffix` built its attention mask from a Python list, a
  host-to-device copy that blocks capture. It's now built on the device.
  (`att_masks[0] = 1` copies too: a CPU scalar; the mask now comes from `arange(n) == 0`.)
- **Capture under `no_grad`:** `torch.inference_mode(False)` also *re-enables autograd*, so
  graphs were being captured recording autograd ops. On the real checkpoint this only worked
  because the language model's weights are frozen. SmolVLA's sampler had the same latent bug;
  both are fixed.

| | Prediction | Images | KV fill | Step |
|---|---|---|---|---|
| Eager | 1148 ms | 241 ms (3 slots) | 296 ms | 60 ms |
| CUDA graphs | 588 ms, **bit-identical** | 161 ms (2 slots) | 290 ms | 14.1 ms |
| + TF32 images (default) | ~497 ms | ~70 ms | 290 ms | 14.1 ms |

**TF32 image encoding:** SigLIP runs in fp32, and TF32 tensor cores encode an image 2.3× faster
(80.5 → 34.5 ms) with a relative error of 4e-4, about 10× below the bf16 rounding of the
language model it feeds. It is scoped to image encoding: the global TF32 flags are restored
after each call. `enable_cuda_graphs(..., tf32_images=False)` keeps sampling bit-identical to
eager.

**Real loop** (`lerobot-rollout`, real cameras, stand-in arm, new script defaults, 60 s): 29.9
Hz, 0 overruns, RTC inference 516 ms (max 543), delay 16 ticks, queue 35 after merge.

### 2026-10-19 — eval script: 30 Hz, RTC replace mode

The script ran at **10 Hz**, a third of the dataset's 30 Hz, so each plan played back in slow
motion: the same bug the SmolVLA and Diffusion scripts had. It also ran RTC in **append mode**
(`RTC_ENABLED=false`), with a queue threshold of 30. In append mode a new chunk is requested
when the queue falls to the threshold, then appended *behind* the queued actions. Every
executed chunk was therefore predicted from an observation at least 30 ticks old: **3 s stale at
10 Hz**.

The header justified append mode by "jump back" corrections in replace mode. Two things explain
those instead:
- **the RTC delay bug** fixed earlier today (`77143bf8`: skip the actions executed during
  inference, not an estimate);
- **an execution horizon (10) shorter than the inference delay** (14 ticks at 1.4 s and 10 Hz).
  The leftover plan is cut to the horizon, so the actions executed during inference were never
  guided.

New defaults: FPS 30, 50 actions per chunk, RTC replace mode, horizon 25, threshold 25. With
`--cuda-graph`, the delay is 16 ticks, so after each merge 34–35 actions remain and the next
inference starts with 25 left. The old behaviour is one line away:
`FPS=10 NUM_ACTION_STEPS=20 RTC_ENABLED=false RTC_EXECUTION_HORIZON=10 RTC_QUEUE_THRESHOLD=30`.
The script warns when running eager above 10 Hz, since eager inference outlasts the queue.

### 2026-10-19 — TensorRT: 0.5 s → 0.25 s, more accurate than bf16

`lerobot.policies.pi05.tensorrt_pi05` builds three engines, staged and measured one at a time:

| Engine | PyTorch (graphed) | TensorRT | Error vs fp32: PyTorch bf16 → TensorRT |
|---|---|---|---|
| Vision, 2 images (fp16) | 161 ms fp32, ~69 ms TF32 | 20.5 ms | (fp32 → 0.0043) |
| KV fill, Gemma 2B (strongly typed fp16) | 290 ms | 152 ms | 0.025 → 0.0078 |
| One denoising step (strongly typed fp16) | 14.1 ms | 7.4 ms | 0.0063 → 0.0013 |

**fp16, not bf16.** TensorRT 10.3 builds bf16 on the Orin, and a strongly typed bf16 step
(7.97 ms) matches PyTorch's arithmetic. fp16 has 3 more mantissa bits and doesn't overflow
where it matters, so it is both faster and about 5× more accurate:
- **Expert:** finite throughout.
- **Gemma 2B:** hidden states peak at ~22k, inside fp16's 65,504. Only the last layer's output
  reaches 161,792 and overflows, and the KV cache doesn't use it: the last layer's K/V come from
  its input.

**Strongly typed or wrong.** As with SmolVLA, a plain `--fp16` build of the fp16 export is
silently wrong (step relative error 1.03, all finite). Building strongly typed keeps the model's
explicit fp32 upcasts in RMSNorm and attention.

**Precomputed AdaRMS modulations.** The expert's adaptive norms project the time embedding
through 37 fp32 3072×1024 matrices (475 MB). They depend only on the timestep, and the loop only
visits 10 fixed timesteps. So the step engine takes the modulations as an input instead, from a
`[10, 37, 3072]` table computed once at attach time with the model's own ops and looked up on
the device inside the CUDA graph. This removed 475 MB of weight reads from every step and
shrank the step engine to 625 MB. It also kept the model's float64 time embedding out of
TensorRT, which has no float64: computing it in float32 shifted `v_t` by up to 0.026, because
pi0.5's 0.004 minimum period makes the sine arguments large.

**Smaller fixes along the way:**
- `PiGemmaModel` casts its input to bf16 only when its weights are bf16, so the fp16 export
  copies cast the embeddings to the weight dtype themselves (a no-op for bf16).
- `TensorRTEngine` maps bf16 tensors.

**Tests.** pi0.5 initializes the expert's AdaRMS projections to zero, which zeroes every expert
layer's gated residual in a fresh model. The miniature test policies now get random projection
weights, so the graph and TensorRT tests actually exercise the expert.

**Results:**

| | Prediction (offline) | With RTC | Actions vs fp32, max (mean) | With RTC |
|---|---|---|---|---|
| bf16 eager = CUDA graphs | 588 ms | | 0.037 (0.0053) | 0.014 (0.0021) |
| **TensorRT** | **246 ms** | **249 ms** | **0.019 (0.0032)** | **0.0066 (0.0008)** |

**Real loop** (`--trt`, real cameras, stand-in arm, 60 s): 29.9 Hz, 0 overruns, RTC inference
258–263 ms (max 277), delay 8 ticks, queue 42–43 after merge.
- **Startup:** about a minute (loading 5.2 GB of engines, self-check, capture).
- **Engine build:** 200 s, with at least 21 GB of memory still available throughout.

### Next

**Robot trials** at 30 Hz:

| Configuration | Command | Attempts | Successes | Notes |
|---|---|---|---|---|
| TensorRT, blue block | `eval_pick_block_pi05_orin.sh --trt` | | | new defaults |
| TensorRT, brown block | `TASK="pick up the brown block …" … --trt` | | | compare with SmolVLA |

**Remaining time:** KV fill 152 ms, 10 steps 74 ms, vision 20 ms. The KV fill is compute-bound
(3.8 TFLOP at ~25 TFLOPS), and the step is near its weight-bandwidth floor (~5 ms for 625 MB).

## How the speed-up works

- **Loading:** `PI05Policy.from_pretrained` skips random initialization when the checkpoint
  covers every parameter.
- **TensorRT:** `lerobot-rollout --tensorrt_engine=<dir>` attaches the three engines
  (`attach_tensorrt_engines`), checks each one against PyTorch, and replays them through the same
  graphed sampler as `--cuda_graph`. Build them with `scripts/local/build_pi05_trt_engine_orin.sh`,
  or let `eval_pick_block_pi05_orin.sh --trt` do it.
- **CUDA graphs:** `lerobot-rollout --cuda_graph=true` (SmolVLA or PI0.5) calls
  `enable_cuda_graphs(policy, present_image_keys)`. That replaces `model.sample_actions` with
  `GraphedSampler` and captures the graphs from a dummy prediction. Batch sizes other than 1, or
  a non-default step count, fall back to eager.

## Tests

- `tests/policies/pi0_pi05/tiny_pi05.py`: a miniature PI0.5 fixture (tiny Gemma widths, a
  one-layer SigLIP, 28×28 images) that builds in seconds without the Hub. It needed one source
  change: the vision projection width now comes from the configured Gemma width (2048 for the
  real model, unchanged) instead of a hard-coded 2048.
- `test_pi05_loading.py`: `from_pretrained` matches the saved policy bit for bit, and a
  checkpoint missing a parameter gets the fallback (an initialized model, not skipped-init
  memory).
- `test_pi05_tensorrt.py`: engines built for the miniature policy; sampling matches eager with
  and without RTC, and engines from another checkpoint are rejected.
- `test_pi05_cuda_graph.py`:
  - graphed sampling matches eager with and without RTC;
  - TF32 images stay close to fp32, and the global TF32 flags are restored;
  - other batch sizes run eagerly;
  - other policy types are rejected.
