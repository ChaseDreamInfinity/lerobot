# SmolVLA on the Jetson Orin: speed-up work and findings

How to run the SO-101 SmolVLA policy on the AGX Orin with CUDA graphs, what has been measured,
and two configuration bugs found in the eval script. The latency log format is the same as for
ACT; see [ACT_ORIN_OPTIMIZATION.md](ACT_ORIN_OPTIMIZATION.md#reading-the-latency-log).

Checkpoint: `chasedreaminf/smolvla_so101-pick-color-block-in-cup-2cam-run2-v3`.

## Quick start

```bash
bash scripts/local/orin_max_perf.sh                                      # after every reboot

bash scripts/local/eval_pick_block_smolvla_orin.sh --cuda-graph          # recommended
bash scripts/local/eval_pick_block_smolvla_orin.sh                       # baseline: PyTorch eager

TASK="pick up the green block and place it into the cup" \
  bash scripts/local/eval_pick_block_smolvla_orin.sh --cuda-graph        # another block colour
DURATION=30 bash scripts/local/eval_pick_block_smolvla_orin.sh --cuda-graph --dry-run
```

- **Asynchronous inference:** SmolVLA runs with Real-Time Chunking (RTC). Inference runs in a
  background thread while the control loop executes queued actions, so the control loop's latency
  log shows only cheap ticks.
- **Inference latency:** logged by the RTC engine every 10 predictions:

  ```
  RTC inference: latency last 142 ms, max of last 20 165 ms | delay 5 ticks | queue 45
  ```

  - **delay:** how many ticks passed while the chunk was computed. Those actions are already in
    the past, so RTC drops them.
  - **queue:** actions waiting after the new chunk is merged in.
- **Startup:** `--cuda-graph` captures its graphs at startup, which takes about 2 s before the
  robot starts. There is nothing to build or rebuild.

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

### Next

**Robot trials** with the fixed camera mapping, at 30 Hz:

| Configuration | Command | Attempts | Successes | Notes |
|---|---|---|---|---|
| CUDA graphs, brown block | `eval_pick_block_smolvla_orin.sh --cuda-graph` | | | first run with correct cameras |
| CUDA graphs, green block | `TASK="pick up the green block …" … --cuda-graph` | | | checks language conditioning |

Remaining speed-up lead:
- **Image + text embedding, about 50 ms of the 131.** The SigLIP vision tower runs 3 times, once
  per image slot, including the empty slot, whose output is constant. Capturing it needs either
  replacing the boolean-mask indexing in the `transformers` vision embeddings with a
  `torch.where` equivalent, or caching the empty slot's embedding.

## How the speed-up works

- **Opt-in flag:** `lerobot-rollout --cuda_graph=true` (SmolVLA only) replaces
  `VLAFlowMatching.sample_actions` with a graphed version. The code is in
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

`tests/policies/smolvla/test_smolvla_cuda_graph.py` uses a small randomly initialized SmolVLA:
- graphed sampling matches eager, with and without RTC, across fresh inputs
- the graphs are captured once
- other batch sizes run eagerly
- non-SmolVLA policies are rejected

```bash
uv run --no-sync --with pytest --with numpy==1.26.4 python -m pytest tests/policies/smolvla
```
