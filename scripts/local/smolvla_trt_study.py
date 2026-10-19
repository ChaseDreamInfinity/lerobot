#!/usr/bin/env python
"""SmolVLA TensorRT study (Jetson Orin): export, build, and compare against eager and CUDA graphs.

The first TensorRT study (fp32 KV fill, fp16 step). Superseded by the rollout integration in
lerobot.policies.smolvla.tensorrt_smolvla (adds a vision engine and a strongly typed fp16 KV fill);
kept to reproduce the study's numbers. Results are in scripts/local/SMOLVLA_ORIN_OPTIMIZATION.md.

Two engines replace the two parts of sampling that run per prediction after image encoding:
- KV fill: the 16-layer VLM pass over the 241-token prefix, emitting per-layer K/V (fp32: in fp16
  it overflows, since prefix activations reach ~7e3);
- step: one denoising step of the action expert, with all 32 K/V tensors as inputs (fp16).
Mask arithmetic is rewritten in int32 (TensorRT cannot sum or multiply bool tensors); the wrappers
are checked to match the model exactly before export. The KV engine and the 10 step calls are
then captured in one CUDA graph.

    uv run --no-sync --with onnx --with numpy==1.26.4 python scripts/local/smolvla_trt_study.py export
    uv run --no-sync python scripts/local/smolvla_trt_study.py build
    uv run --no-sync python scripts/local/smolvla_trt_study.py eval
"""

import argparse
import copy
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.smolvla import modeling_smolvla
from lerobot.policies.smolvla.cuda_graph_smolvla import _capture, enable_cuda_graphs
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.policies.smolvla.tensorrt_smolvla import KVFillExport, StepExport
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.tensorrt import TensorRTEngine, build_engine

POLICY = "chasedreaminf/smolvla_so101-pick-color-block-in-cup-2cam-run2-v3"
RENAME = {
    "observation.images.wrist": "observation.images.camera1",
    "observation.images.top": "observation.images.camera2",
}
TASK = "pick up the brown block and place it into the cup"
OUT = Path("outputs/trt_engines/smolvla_study_orin")


def load():
    cfg = PreTrainedConfig.from_pretrained(POLICY)
    cfg.pretrained_path, cfg.device = POLICY, "cuda"
    policy = get_policy_class("smolvla").from_pretrained(POLICY, config=cfg).to("cuda").eval()
    pre, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=POLICY,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": RENAME},
        },
    )
    return cfg, policy, pre


def sample_inputs(policy, pre, rng):
    obs = {
        "observation.state": rng.uniform(-50, 50, 6).astype(np.float32),
        "observation.images.wrist": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        "observation.images.top": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
    }
    with torch.inference_mode():
        b = pre(prepare_observation_for_inference(obs, torch.device("cuda"), TASK, "so101_follower"))
    images, img_masks = policy.prepare_images(b)
    lang = (b["observation.language.tokens"], b["observation.language.attention_mask"])
    return images, img_masks, *lang, policy.prepare_state(b)


# The export wrappers now live in the policy package (int32 mask arithmetic, flat K/V tuples).
KVFill, Step = KVFillExport, StepExport


KV_NAMES = [f"{k}_{i}" for i in range(16) for k in ("k", "v")]
KV_INPUTS = ["prefix_embs", "prefix_pad_masks", "prefix_att_masks"]
STEP_INPUTS = ["x_t", "timestep", "prefix_pad_masks", *KV_NAMES]


@torch.no_grad()
def export():
    _, policy, pre = load()
    m = policy.model
    images, img_masks, lt, lm, state = sample_inputs(policy, pre, np.random.default_rng(0))
    pe, ppm, pam = m.embed_prefix(images, img_masks, lt, lm, state=state)
    cpu = copy.deepcopy(m).float().cpu().eval()
    # TensorRT has no float64; the sinusoidal time embedding is computed in float32 instead.
    modeling_smolvla.get_safe_dtype = lambda dtype, device: torch.float32 if dtype == torch.float64 else dtype
    kv_args = (pe.float().cpu(), ppm.cpu(), pam.cpu())
    _, ref_kv = cpu.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(ppm.cpu(), pam.cpu()),
        position_ids=torch.cumsum(ppm.cpu(), 1) - 1,
        past_key_values=None,
        inputs_embeds=[kv_args[0], None],
        use_cache=True,
        fill_kv_cache=True,
    )
    kv = KVFill(cpu)(*kv_args)
    ref_flat = [t for i in range(16) for t in (ref_kv[i]["key_states"], ref_kv[i]["value_states"])]
    assert all(torch.equal(a, b) for a, b in zip(kv, ref_flat, strict=True)), (
        "KV wrapper differs from the model"
    )
    x, t = torch.randn(1, 50, 32), torch.tensor([0.7])
    expected = cpu.denoise_step(prefix_pad_masks=ppm.cpu(), past_key_values=ref_kv, x_t=x, timestep=t)
    assert torch.equal(Step(cpu)(x, t, ppm.cpu(), *kv), expected), "step wrapper differs from the model"

    OUT.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        KVFill(cpu),
        kv_args,
        str(OUT / "kv.onnx"),
        input_names=KV_INPUTS,
        output_names=KV_NAMES,
        opset_version=17,
        dynamo=False,
    )
    torch.onnx.export(
        Step(cpu),
        (x, t, ppm.cpu(), *kv),
        str(OUT / "step.onnx"),
        input_names=STEP_INPUTS,
        output_names=["v_t"],
        opset_version=17,
        dynamo=False,
    )
    print(f"exported {OUT}/kv.onnx and {OUT}/step.onnx (wrappers match the model exactly)")


def build():
    build_engine(OUT / "kv.onnx", OUT / "kv_fp32.engine", fp16=False)
    build_engine(OUT / "step.onnx", OUT / "step_fp16.engine", fp16=True)
    print(f"built {OUT}/kv_fp32.engine and {OUT}/step_fp16.engine")


@torch.no_grad()
def evaluate(trials: int):
    _, policy, pre = load()
    m32 = copy.deepcopy(policy.model).float().eval()  # reference
    eager = get_policy_class("smolvla").from_pretrained(POLICY, config=policy.config).to("cuda").eval()
    sampler = enable_cuda_graphs(policy, present_image_keys=list(RENAME.values()))
    m = policy.model
    kv_engine = TensorRTEngine(OUT / "kv_fp32.engine", KV_INPUTS, KV_NAMES)
    step_engine = TensorRTEngine(OUT / "step_fp16.engine", STEP_INPUTS, ["v_t"])

    rng = np.random.default_rng(1)
    images, img_masks, lt, lm, state = sample_inputs(policy, pre, rng)
    pe, ppm, pam = m.embed_prefix(sampler._with_cached_constants(images), img_masks, lt, lm, state=state)
    static = {
        "pe": pe.clone(),
        "ppm": ppm.clone(),
        "pam": pam.clone(),
        "noise": torch.randn(1, 50, 32, device="cuda"),
    }
    n_steps = policy.config.num_steps
    static["times"] = [torch.full((1,), 1.0 - s / n_steps, device="cuda") for s in range(n_steps)]

    def trt_loop():
        kv = kv_engine(static["pe"], static["ppm"], static["pam"])
        x = static["noise"]
        for s in range(n_steps):
            x = x + (-1.0 / n_steps) * step_engine(x, static["times"][s], static["ppm"], *kv)
        return x

    graph, out = _capture(trt_loop)

    def trt_predict(images, img_masks, lt, lm, state, noise):
        pe, ppm, pam = m.embed_prefix(sampler._with_cached_constants(images), img_masks, lt, lm, state=state)
        for key, value in (("pe", pe), ("ppm", ppm), ("pam", pam), ("noise", noise)):
            static[key].copy_(value)
        graph.replay()
        return out.clone()

    paths = {
        "PyTorch eager bf16": lambda *a, noise: eager.model.sample_actions(*a, noise=noise),
        "CUDA graphs (rollout --cuda_graph)": lambda *a, noise: m.sample_actions(*a, noise=noise),
        "TensorRT fp32 KV + fp16 step, graphed": lambda *a, noise: trt_predict(*a, noise),
    }
    errors = {name: [] for name in paths}
    for _ in range(trials):
        inputs = sample_inputs(policy, pre, rng)
        noise = torch.randn(1, 50, 32, device="cuda")
        reference = m32.sample_actions(*inputs, noise=noise)
        for name, fn in paths.items():
            diff = (fn(*inputs, noise=noise) - reference).abs()
            errors[name].append((diff.max().item(), diff.mean().item()))

    def latency(fn, n=15):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(times)

    inputs = sample_inputs(policy, pre, rng)
    noise = torch.randn(1, 50, 32, device="cuda")
    print(f"{'path':40s} {'latency':>9s}   vs fp32 eager: max / mean |diff| ({trials} inputs)")
    for name, fn in paths.items():
        ms = latency(lambda fn=fn: fn(*inputs, noise=noise), n=5 if "eager" in name else 15)
        worst = max(a for a, _ in errors[name])
        mean = statistics.mean(b for _, b in errors[name])
        print(f"{name:40s} {ms:7.1f} ms   {worst:.4f} / {mean:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=["export", "build", "eval"])
    parser.add_argument("--trials", type=int, default=4)
    args = parser.parse_args()
    {"export": export, "build": build, "eval": lambda: evaluate(args.trials)}[args.stage]()
