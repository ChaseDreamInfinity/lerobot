# SmolVLA on the Jetson Orin: speed-up work and findings

How to run the SO-101 SmolVLA policy on the AGX Orin with CUDA graphs, what has been measured,
and two configuration bugs found in the eval script. The latency log format is the same as for
ACT; see [ACT_ORIN_OPTIMIZATION.md](ACT_ORIN_OPTIMIZATION.md#reading-the-latency-log).

Checkpoint: `chasedreaminf/smolvla_so101-pick-color-block-in-cup-2cam-run2-v3`.

## Quick start

```bash
bash scripts/local/orin_max_perf.sh                                      # after every reboot

bash scripts/local/eval_pick_block_smolvla_orin.sh --trt                 # fastest: TensorRT, ~70 ms per chunk
bash scripts/local/eval_pick_block_smolvla_orin.sh --cuda-graph          # no engines needed: ~127 ms
bash scripts/local/eval_pick_block_smolvla_orin.sh                       # baseline: PyTorch eager, ~890 ms

TASK="pick up the green block and place it into the cup" \
  bash scripts/local/eval_pick_block_smolvla_orin.sh --cuda-graph        # another block colour
DURATION=30 bash scripts/local/eval_pick_block_smolvla_orin.sh --cuda-graph --dry-run
```

- **Asynchronous inference:** SmolVLA runs with Real-Time Chunking (RTC). Inference runs in a
  background thread while the control loop executes queued actions, so the control loop's latency
  log shows only cheap ticks.
- **Inference latency:** logged by the RTC engine every 10 predictions:

  ```
  RTC inference: latency last 127 ms, max of last 20 143 ms | delay 4 ticks | queue 46
  ```

  - **delay:** how many ticks passed while the chunk was computed. Those actions are already in
    the past, so RTC drops them.
  - **queue:** actions waiting after the new chunk is merged in.
- **Startup:** both options capture their CUDA graphs at startup, which takes about 2 s before the
  robot starts.
- **Engines for `--trt`:** built automatically on first use (about 3 min), in
  `outputs/trt_engines/<checkpoint>_orin_fp16/`, or explicitly with
  `bash scripts/local/build_smolvla_trt_engine_orin.sh`.
- **When to rebuild:** after a new checkpoint, a JetPack or TensorRT upgrade, or a change in the
  camera count. Mismatched engines are refused when they load. `--cuda-graph` needs nothing built.

## The model

SmolVLA is a vision-language-action model. It reads the task text, so unlike ACT and Diffusion
Policy it can tell the brown block from the blue and green ones. One prediction is a chunk of 50
actions:

```
camera1 (wrist), camera2 (top), empty slot ─ SigLIP vision (512×512) ─ 3 × 64 image tokens ─┐
task text (48 tokens, padded) ──────────────────────────── text embeddings ───────────────────┤
arm state ─────────────────────────────────────────────────── 1 state token ──────────────────┤
                                                                      241-token prefix ───────┤
                                                       SmolVLM2 text layers (16) ─ KV cache ──┤
                                                                                              ▼
noise (50×32) ─► [ action expert, cross-attends to the KV cache ] × 10 flow steps ─► 50×6 actions
```

**Size:**

| Part | Parameters |
|---|---|
| SigLIP vision encoder | 86M |
| SmolVLM2 text layers (16) | 205M |
| Action expert | 98M |
| Embeddings, projections | 61M |
| **Total** | **450M** in bf16; `model.safetensors` is 907 MB |

**Training:**
- 60k steps at batch 4, fine-tuned from `lerobot/smolvla_base`.
- Same dataset as ACT and Diffusion: `chasedreaminf/record-pick-block-in-cup-2cam-run1_20260505_191315`,
  recorded at 30 Hz, with brown, blue and green block tasks.
- Image augmentation enabled.
- Cameras renamed **wrist → camera1, top → camera2**.

## Progress log

### 2026-10-19 — two configuration bugs in the eval script

1. **The cameras were swapped.** The script passed
   `--rename_map='{top: camera1, wrist: camera2}'`, the opposite of the map the checkpoint was
   trained with:
   - `train_config.json` and the saved `policy_preprocessor.json` both say `wrist → camera1`,
     `top → camera2`.
   - `--rename_map` overrides the saved map, so the policy saw the overhead view in its
     gripper-camera slot and vice versa.
   - Fixed: the script now repeats the training map.
   - *Lesson:* an earlier "fix" took the mapping from a comment in the script header, which was
     written for an older checkpoint. The checkpoint's own `train_config.json` is the authority.
2. **The rate was wrong: 20 Hz instead of the dataset's 30 Hz.** Each predicted action is one
   1/30 s step, so at 20 Hz the plan played back at two-thirds speed. The same mistake at 15 Hz
   stopped Diffusion Policy from going for the block (see
   [DIFFUSION_ORIN_OPTIMIZATION.md](DIFFUSION_ORIN_OPTIMIZATION.md)). The script now defaults to
   30 Hz, which needs the speed-up below.

Both bugs are fixed in the script, but have not been tested on the robot yet.

### 2026-10-19 — baseline measured

Setup: MAXN power mode, batch size 1, 10 flow steps. Measured without moving the arm:

| Stage | Time |
|---|---|
| Image + text embedding (`embed_prefix`) | 50 ms |
| KV-cache fill (16-layer VLM pass over 241 tokens) | ~62 ms |
| **One denoising step** (action expert) | **56 ms** × 10 |
| **One prediction, no RTC** | **680 ms** |
| **One prediction, RTC guidance on** (the script's setting) | **882 ms** (72 ms per step) |

**Findings:**
- **The bottleneck is CPU launch overhead, as for ACT and Diffusion.** The weights are already
  bf16. One denoising step runs the 98M-parameter expert over only 50 tokens, so the GPU mostly
  waits for the CPU to issue hundreds of small kernels.
- **What RTC guidance really costs.** Its correction does *not* differentiate through the network.
  `v_t` is computed before `x_t` requires grad, so the "gradient" is exactly the error term. The
  extra 16 ms per step only comes from running the network with autograd recording.
- **Effect on the robot.** In the real rollout loop at 30 Hz, a prediction took 890 ms (up to
  1.2 s), so RTC's delay was **27 ticks**: 27 of each 50-action chunk were already in the past
  when it arrived, and the remaining 23 were planned from a 0.9 s old observation.

### 2026-10-19 — CUDA graphs

**Two graphs**, captured when the policy loads:
1. The prefix KV-cache fill.
2. One denoising step, reading the KV cache in place.

What stays eager:
- **The 10-step loop,** so the step count and RTC guidance are unchanged. RTC calls the graphed
  step as its denoiser.
- **Image and text embedding (~50 ms),** because the SigLIP embeddings in `transformers` use
  boolean-mask indexing, which syncs with the host.

**Model changes needed for capture.** Three per-call CPU-to-GPU copies in
`modeling_smolvla.py` became GPU-side equivalents:
- the suffix attention mask (`torch.ones`)
- the prefix attention mask (cached per shape pattern)
- the image-embedding scale (`torch.full`)

Outputs are bit-identical to the original code, for inference and for the training loss.

| | Eager | CUDA graphs |
|---|---|---|
| One denoising step | 55.6 ms | **6.4 ms** (8.7×) |
| One prediction, no RTC | 713 ms | **128 ms** |
| One prediction, RTC | 854 ms | **131 ms** (6.5×) |
| Difference from eager | — | **0** (bit-identical, with and without RTC) |

Real `lerobot-rollout` loop at **30 Hz** for 25 s, with RTC, the real cameras and a stand-in arm:

| | RTC inference latency | RTC delay | Queue after merge | Control loop |
|---|---|---|---|---|
| Eager | 890 ms (max 1223) | 27 ticks | 23 | 30 Hz, 0 overruns |
| **CUDA graphs** | **142 ms** (max 165) | **5 ticks** | **45** | 30 Hz, 0 overruns |

The control loop is fine either way, because inference runs in the background. What changes is
how fresh the actions are: with graphs the policy reacts to what the cameras saw about 0.14 s
earlier instead of 0.9 s, and keeps 45 of each 50 actions instead of 23.

### 2026-10-19 — how far can it go? Upper-bound study

Where the 128 ms went, against a published TensorRT implementation and a hardware floor:

| Part | Ours (CUDA graphs) | EdgeVLA-TRT, fp16 | Hardware floor, fp16 |
|---|---|---|---|
| Image encoding | 48 ms: 3 × 16 ms, eager, including the empty slot | 33.5 ms (2 images) | ~12 ms (2 × 6.1) |
| Prefix / KV-cache fill | 13.9 ms | 9.6 ms | ~2.4 ms |
| 10 denoising steps | 65 ms (6.5 per step) | 40.4 ms (4.0 per step) | ~12 ms (1.17 per step) |
| **Whole prediction** | **128 ms** | **83 ms** | **~27 ms** |

- **How the floor is worked out:** it assumes perfect kernels. Weights are read at the measured
  168 GB/s memory bandwidth (expert 196 MB, VLM 409 MB), and the vision encoder needs ~177 GFLOP
  per image at the measured 29 TFLOPS bf16. INT8 weights would roughly halve the memory-bound parts.
- **What EdgeVLA-TRT is:** a community fork of NVIDIA TensorRT Edge-LLM that runs on JetPack 6.2.
  Its 83 ms comes from its design notes: a C++ runtime, an SO-101 SmolVLA checkpoint with the same
  architecture, 2 cameras, 10 steps. Its LIBERO table reports 122 ms through a Python server.
- **What EdgeVLA-TRT needs:**
  - custom TensorRT plugins
  - CuTe-DSL attention kernels
  - a CUDA 12.9 runtime library swapped in
  - a separate CPU-only torch for export
- **Official TensorRT Edge-LLM:** it supports Orin only on JetPack 7.2, and lists LLMs and VLMs,
  not SmolVLA.

**Two changes tried:**
1. **Cache the empty camera slot: kept.** `prepare_images` pads the missing third camera with a
   constant image (all −1), and SigLIP was encoding it on every prediction. Its embedding is now
   computed once. Each image is compared with the constant on the GPU (0.36 ms for 3 images), so
   only an identical input uses the cache. Saves one 16 ms image encode per prediction.
2. **CUDA-graph the vision encoder: dropped.** It worked and was exact, but saved only ~1.6 ms per
   image: 14.5 ms graphed against 16.1 ms eager, because the ~1,000-token attention at 512×512 is
   real GPU work. For comparison, EdgeVLA-TRT's vision engine takes ~16.7 ms per image. Capture
   also needed a copy of `transformers`' SigLIP embedding code with `torch.where` in place of
   boolean-mask indexing, which could silently diverge on a `transformers` upgrade. Not worth
   ~3 ms.

**Two lessons from the vision-graph attempt:**
- **Keep every graph input referenced.** The patch mask was held only by the lambda passed to the
  capture. Once the lambda was garbage-collected, the mask's memory was freed while the graph kept
  reading it. Results were exact until something else, here loading a second model, reused that
  memory. Then they were wrong by tens of units.
- **Watch for graph-only mask paths.** `transformers` skips an all-true attention mask on the host
  in eager mode, but materializes it under capture. That runs a different SDPA kernel, which is
  equal in exact arithmetic but not bitwise in bf16. Calling the encoder with
  `attention_mask=None` keeps the eager kernel.

**Result:** outputs are bit-identical to eager PyTorch, with and without RTC.

| | Before | After |
|---|---|---|
| One prediction, RTC (offline) | 131 ms | **115 ms** |
| Real loop at 30 Hz: RTC inference | 142 ms | **127 ms** (max 143) |
| RTC delay | 5 ticks | **4 ticks** |

**What is left:**
- **Denoising steps: 65 ms against TensorRT's 40.** The remaining gap is kernel fusion, and
  closing it would need a TensorRT export of the expert with a 16-layer KV cache.
- **Image encoding: 32 ms.** Already about as fast as TensorRT.
- **Effect on the robot:** the denoising gain would take the delay from 4 ticks to 3, so a
  TensorRT export is not worth its cost on this board for now.

### 2026-10-19 — TensorRT study (stages 1–3)

The aim: find out, for study purposes, how much a plain TensorRT export can gain without the
EdgeVLA fork's custom plugins. To reproduce, run `scripts/local/smolvla_trt_study.py`
(`export` / `build` / `eval`). Engines go to `outputs/trt_engines/smolvla_study_orin/`. It is not
used by rollout.

**Stage 1: one denoising step.**
- **Export:** the action expert's step, with all 16 layers' cached K/V as 32 explicit inputs,
  exported to ONNX with the standard exporter in 8 s (403 MB). Two changes were needed:
  - **Masks.** TensorRT cannot sum or multiply bool tensors, which the step does on the padding
    masks for position ids and 2D masks. The wrapper does that arithmetic in int32, and is
    checked to match the model's own `denoise_step` exactly.
  - **Time embedding.** TensorRT has no float64, so the sinusoidal time embedding is computed in
    float32 for export.
- **Speed:**

  | One denoising step | GPU time |
  |---|---|
  | Our CUDA graph, bf16 | 6.5 ms |
  | TensorRT fp32 (TF32) | 4.8 ms |
  | TensorRT bf16 | 3.5 ms |
  | **TensorRT fp16** | **2.9 ms** |
  | EdgeVLA-TRT fork | 4.0 ms |
  | Memory floor | 1.17 ms |

**Stage 2: accuracy.** Full 10-step sampling with the same noise and K/V from PyTorch, compared
with fp32 PyTorch (actions have std 0.82):
- **bf16 PyTorch (today):** max 0.047, mean 0.0024.
- **TensorRT fp16 step:** max 0.054, mean 0.0022, the same accuracy as today.
- **TensorRT bf16 step:** least accurate, with a mean of 0.0042.

**Stage 3: KV-cache fill.**
- **Export:** the 16-layer prefix pass, with the same int32 mask rewrite, matching the model
  exactly (612 MB ONNX).
- **Results:**

  | KV fill | GPU time | K/V relative error vs fp32 |
  |---|---|---|
  | Our CUDA graph, bf16 | 13.9 ms | 0.058 (bf16 PyTorch) |
  | TensorRT fp32 | 12.7 ms | **0.002** |
  | TensorRT bf16 | 8.1 ms | 0.19, which made actions 4× worse than today |
  | TensorRT fp16 | 7.9 ms | **1.29: unusable** |
  | TensorRT fp16 with RMSNorm ops in fp32 | 7.9 ms | 1.24: still unusable |

- **fp16 overflows.** Prefix activations reach ~7,400, and the bf16-trained VLM has values fp16
  cannot hold (max 65,504). The engine produces no infinities, just silently wrong K/V. Keeping
  RMSNorm's square, mean and rsqrt in fp32 did not help, so the overflow is elsewhere. The likely
  culprit is the attention scores: PyTorch upcasts them to fp32 explicitly, but that cast
  disappears in an fp32 export. Pinning the overflowing layers needs a layer-by-layer comparison,
  e.g. with Polygraphy.

**End to end:** image encoding stays eager, with the empty-slot cache. The fp32 KV engine and the
fp16 step engine × 10 are captured in **one CUDA graph**, since TensorRT launches can be captured.
Without the graph, each engine call from Python costs ~1.6 ms extra for argument binding and
stream sync.

| Path | Prediction | Actions vs fp32 eager (max / mean) |
|---|---|---|
| PyTorch eager bf16 | 723 ms | 0.113 / 0.0027 |
| CUDA graphs (`--cuda_graph`, committed) | 113 ms | 0.113 / 0.0027 (identical to eager) |
| EdgeVLA-TRT fork (published, other checkpoint) | 83 ms | — |
| **TensorRT fp32 KV + fp16 step, in one CUDA graph** | **75.8 ms** | **0.007 / 0.0007** |

**Findings:**
- **Faster than the fork, with no custom plugins.** 75.8 ms against 83 ms, using only JetPack's
  TensorRT 10.3 and the standard ONNX exporter.
- **More accurate than today.** Only the action expert runs in fp16, and it does not overflow,
  while the KV fill keeps fp32. bf16 PyTorch is further from fp32 than this path.
- **Where the time goes now:** ~33 ms of eager image encoding (GPU-bound), ~13 ms of KV fill,
  ~29 ms of steps (10 × 2.9).
- **Next levers:**
  - **An fp16-safe KV fill** (find the overflowing layers): −5 ms.
  - **INT8 expert weights:** the step is memory-bound, so up to ~−15 ms.
  - Either way, about 75 → 55–70 ms.
- **What it would mean on the robot:** at 30 Hz, 75 ms is a delay of about 3 ticks against today's
  4. Integrating it would mean engines per checkpoint and a rewrite of the RTC loop around a
  graphed TensorRT sampler, which is not justified by one tick. It stays a study.

### 2026-10-19 — full TensorRT path, integrated into rollout

The aim: squeeze out the remaining latency before working on accuracy. Three engines, each
measured first:

1. **Image encoder, fp16, both cameras in one batch.** 18.0 ms for the two images, against 32 ms
   for two eager encodes. The SigLIP embeddings' boolean-mask indexing is replaced by an exact
   `torch.where` equivalent, on an export copy only.
2. **KV fill, now in fp16.** The overflow was not where assumed. Hooks showed **no layer's
   output** exceeds fp16's range (largest 2,717), and **plain PyTorch fp16 is accurate**: K/V
   relative error 0.0067, against 0.058 for bf16. PyTorch avoids the overflow because the model
   upcasts to fp32 inside RMSNorm and attention, and those casts are no-ops, so they vanish, in an
   fp32 ONNX export. The fix:
   - export from an **fp16 copy** of the model (bf16 weights converted to fp16, fp32 modules left
     as they are), so the upcasts appear as Cast nodes;
   - build **strongly typed**, so TensorRT keeps them.

   The result is 8.3 ms with relative error 0.024, against 12.7 ms for the fp32 engine.
3. **Step engine, fp16, with fp16 K/V inputs.** The step is exported from the same fp16 copy, so
   it reads the KV engine's output directly. Before this, the 32 K/V tensors were converted on
   every step. A strongly typed step was slower (3.4 ms against 2.9) for no accuracy gain, so the
   step is built with plain `--fp16`.

**Runtime.** The CUDA-graph sampler (`cuda_graph_smolvla.GraphedSampler`) now takes pluggable
backends:
- the KV fill and denoise functions;
- an optional batched image encoder.

The PyTorch path (`--cuda_graph`) and the TensorRT path share the same graph capture, the
empty-slot cache and the Python RTC loop. `--tensorrt_engine=<dir>` attaches the engines through
`lerobot.policies.smolvla.tensorrt_smolvla`, and checks each one against PyTorch when it loads.

**Results**, offline (sampling only, actions vs fp32 eager):

| | No RTC | RTC | Error vs fp32 (max / mean) |
|---|---|---|---|
| CUDA graphs | 113.8 ms | 116.3 ms | 0.041 / 0.0024 |
| **TensorRT** | **58.7 ms** | **61.1 ms** | **0.019 / 0.0012** |

Real `lerobot-rollout` loop at 30 Hz with RTC, the real cameras and a stand-in arm:

| | RTC inference | RTC delay | Queue after merge |
|---|---|---|---|
| Eager PyTorch | 890 ms | 27 ticks | 23 |
| CUDA graphs | 127 ms | 4 ticks | 47 |
| **TensorRT** | **69–71 ms** (max 81) | **3 ticks** | **48** |

**Where the ~60 ms goes now:**

| Part | Time |
|---|---|
| Vision engine (2 images) | 18 ms |
| Prefix assembly and checks | ~1 ms |
| KV engine | 8.3 ms |
| 10 steps | 10 × 2.86 ms |

**Remaining levers:**
- **INT8 weights for the step engine:** it runs at 2.4× its fp16 memory floor.
- **INT8 or FP8-style vision:** compute-bound, with accuracy risk.
- **Fewer flow steps:** this changes the policy, so it is an accuracy decision, not an
  optimization.

### Next

**Robot trials** with the fixed camera mapping, at 30 Hz:

| Configuration | Command | Attempts | Successes | Notes |
|---|---|---|---|---|
| CUDA graphs, brown block | `eval_pick_block_smolvla_orin.sh --cuda-graph` | | | first run with correct cameras |
| CUDA graphs, green block | `TASK="pick up the green block …" … --cuda-graph` | | | checks language conditioning |

Remaining speed-up lead: see the upper-bound study above.

## How the speed-up works

- **TensorRT:** `lerobot-rollout --tensorrt_engine=<dir>` uses the three engines, replayed through
  the same graphed sampler. See the progress log entry above.
- **Opt-in flag:** `lerobot-rollout --cuda_graph=true` (SmolVLA only) replaces
  `VLAFlowMatching.sample_actions` with a graphed version, and wraps
  `SmolVLMWithExpertModel.embed_image` with the empty-slot cache. The code is in
  `src/lerobot/policies/smolvla/cuda_graph_smolvla.py`.
- **Captured at startup:** the graphs are captured before the robot moves, from a dummy prediction
  with the same number of camera images. This matters for RTC, whose delay estimate uses the
  slowest of the last 100 inferences, so a capture during the first real prediction would distort
  it.
- **Fallbacks:** only batch size 1 is graphed; anything else runs the original eager code. A new
  prefix shape is captured on first use.
- **No rebuild ever needed:** no engines and no files. The graphs live in memory, captured fresh
  on every start.

## Tests

`tests/policies/smolvla/test_smolvla_tensorrt.py` builds engines for a small random SmolVLA, and
checks two things:
- TensorRT sampling matches eager, with and without RTC;
- engines from another checkpoint are refused.

It skips without TensorRT or `onnx`. Run it with `--with onnx`.

`tests/policies/smolvla/test_smolvla_cuda_graph.py` uses a small randomly initialized SmolVLA:
- graphed sampling matches eager, with and without RTC, across fresh inputs
- the graphs are captured once, and the empty camera slot is encoded once
- other batch sizes run eagerly
- non-SmolVLA policies are rejected

```bash
uv run --no-sync --with pytest --with numpy==1.26.4 python -m pytest tests/policies/smolvla
```
