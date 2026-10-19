# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CUDA graphs for PI0.5 action sampling.

Eager PI0.5 sampling is mostly launch-bound on small GPUs such as Jetson: each denoising step runs
the 300M action expert over only ``chunk_size`` tokens, so the GPU waits for the CPU to issue
hundreds of small kernels (~60 ms per step eager on an AGX Orin).

Two graphs are captured per prefix shape, on first use:

- the prefix KV-cache fill (the PaliGemma pass over image and language tokens);
- one denoising step of the action expert, reading that KV cache.

Image encoding stays eager, by default in TF32 (``tf32_images``): SigLIP runs in fp32, and TF32
tensor cores encode an image ~2.3x faster (80 -> 35 ms at 224x224 on an AGX Orin) with a relative
error of ~4e-4, well below the bf16 rounding of the language model it feeds. Empty camera slots, the constant ``-1`` image ``_preprocess_images``
pads missing cameras with, are encoded once and cached (saving one image encode per prediction).
Each image is compared with the constant on the device, so the cache is only used for an input
identical to the one it was computed from.

The denoising loop stays in Python, so Real-Time Chunking guidance works unchanged: it calls the
graphed step as its denoiser. (Its correction does not differentiate through the network, since
``v_t`` is computed before ``x_t`` requires grad, so a replayed ``v_t`` gives the same result.)

Only batch size 1 with the configured number of steps is graphed; anything else runs the original
eager code.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator

import torch
from torch import Tensor

from lerobot.utils.cuda_graph import ConstantImageCache, EmptyImageSlots, capture

from .modeling_pi05 import PI05Policy, PI05Pytorch, make_att_2d_masks

logger = logging.getLogger(__name__)

# Backends for the two graphed stages. ``FillKV(embs, pad_masks, att_masks) -> kv`` builds the
# prefix KV cache in whatever form ``Denoise(x_t, timestep, pad_masks, kv) -> v_t`` reads.
FillKV = Callable[[Tensor, Tensor, Tensor], object]
Denoise = Callable[[Tensor, Tensor, Tensor, object], Tensor]


def pytorch_fill_kv(model: PI05Pytorch) -> FillKV:
    """The KV fill of ``PI05Pytorch.sample_actions``."""

    def fill_kv(embs: Tensor, pad_masks: Tensor, att_masks: Tensor):
        att_2d_masks_4d = model._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=torch.cumsum(pad_masks, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=True,
        )
        return past_key_values

    return fill_kv


def pytorch_denoise(model: PI05Pytorch) -> Denoise:
    def denoise(x_t: Tensor, timestep: Tensor, pad_masks: Tensor, past_key_values) -> Tensor:
        return model.denoise_step(
            prefix_pad_masks=pad_masks, past_key_values=past_key_values, x_t=x_t, timestep=timestep
        )

    return denoise


@contextlib.contextmanager
def _tf32() -> Iterator[None]:
    """Allow TF32 for fp32 matmuls and convolutions, restoring the previous settings after."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


def _in_tf32(embed_image: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    def embed(image: Tensor) -> Tensor:
        with _tf32():
            return embed_image(image)

    return embed


class _PrefixGraphs:
    """Static buffers and the two graphs (KV fill, one denoising step) for one prefix shape."""

    def __init__(
        self,
        model: PI05Pytorch,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        fill_kv: FillKV,
        denoise: Denoise,
    ):
        config = model.config
        device = prefix_embs.device
        self.embs = prefix_embs.clone()
        self.pad_masks = prefix_pad_masks.clone()
        self.att_masks = prefix_att_masks.clone()
        self.x_t = torch.zeros(1, config.chunk_size, config.max_action_dim, device=device)
        self.timestep = torch.ones(1, device=device)

        self.kv_graph, self.past_key_values = capture(
            lambda: fill_kv(self.embs, self.pad_masks, self.att_masks)
        )
        self.kv_graph.replay()  # real cache contents for the denoiser warm-up
        self.step_graph, self.v_t = capture(
            lambda: denoise(self.x_t, self.timestep, self.pad_masks, self.past_key_values),
            pool=self.kv_graph.pool(),
        )

    def load_prefix(self, prefix_embs: Tensor, prefix_pad_masks: Tensor, prefix_att_masks: Tensor) -> None:
        self.embs.copy_(prefix_embs)
        self.pad_masks.copy_(prefix_pad_masks)
        self.att_masks.copy_(prefix_att_masks)
        self.kv_graph.replay()

    def denoise_step(self, x_t: Tensor, timestep: Tensor) -> Tensor:
        with torch.no_grad():
            self.x_t.copy_(x_t)
            self.timestep.copy_(timestep)
            self.step_graph.replay()
            return self.v_t.clone()


class GraphedSampler:
    """Drop-in replacement for ``PI05Pytorch.sample_actions``.

    The KV fill and denoising step run from CUDA graphs over ``fill_kv`` and ``denoise`` (the
    PyTorch model by default). ``tf32_images`` encodes images with TF32 matmuls (see the module
    docstring); without it, sampling is bit-identical to eager.
    """

    def __init__(
        self,
        model: PI05Pytorch,
        fill_kv: FillKV | None = None,
        denoise: Denoise | None = None,
        tf32_images: bool = True,
    ):
        self.model = model
        self.eager = model.sample_actions
        self.fill_kv = fill_kv or pytorch_fill_kv(model)
        self.denoise = denoise or pytorch_denoise(model)
        self._graphs: dict[tuple, _PrefixGraphs] = {}

        paligemma_with_expert = model.paligemma_with_expert
        embed_image = paligemma_with_expert.embed_image
        self.image_cache = ConstantImageCache(_in_tf32(embed_image) if tf32_images else embed_image)
        paligemma_with_expert.embed_image = self.image_cache
        self._with_cached_constants = EmptyImageSlots(self.image_cache)

    @torch.no_grad()
    def __call__(self, images, img_masks, tokens, masks, noise=None, num_steps=None, **kwargs) -> Tensor:
        model = self.model
        config = model.config
        bsize = tokens.shape[0]
        if bsize != 1 or num_steps not in (None, config.num_inference_steps):
            return self.eager(images, img_masks, tokens, masks, noise=noise, num_steps=num_steps, **kwargs)
        device = tokens.device

        if noise is None:
            noise = model.sample_noise((bsize, config.chunk_size, config.max_action_dim), device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            self._with_cached_constants(images), img_masks, tokens, masks
        )
        key = (tuple(prefix_embs.shape), prefix_embs.dtype)
        if key not in self._graphs:
            logger.info("Capturing PI0.5 CUDA graphs for prefix shape %s", tuple(prefix_embs.shape))
            # Static buffers must be normal tensors, not inference tensors: they are written in
            # place on every call, whatever grad mode the caller (e.g. RTC guidance) is in.
            # inference_mode(False) also re-enables autograd, so turn it off again for capture.
            with torch.inference_mode(False), torch.no_grad():
                self._graphs[key] = _PrefixGraphs(
                    model, prefix_embs, prefix_pad_masks, prefix_att_masks, self.fill_kv, self.denoise
                )
        graphs = self._graphs[key]
        graphs.load_prefix(prefix_embs, prefix_pad_masks, prefix_att_masks)

        # Same loop as PI05Pytorch.sample_actions, with the graphed denoising step.
        num_steps = config.num_inference_steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return graphs.denoise_step(input_x_t, current_timestep)

            if model._rtc_enabled():
                v_t = model.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                    inference_delay=kwargs.get("inference_delay"),
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=kwargs.get("execution_horizon"),
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if model.rtc_processor is not None and model.rtc_processor.is_debug_enabled():
                model.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t


def enable_cuda_graphs(
    policy: PI05Policy, present_image_keys: list[str] | None = None, tf32_images: bool = True
) -> GraphedSampler:
    """Replay PI0.5's KV-cache fill and denoising steps from CUDA graphs, and encode images in TF32
    (unless ``tf32_images`` is False, which keeps sampling bit-identical to eager).

    Pass ``present_image_keys`` (the image keys the policy will see, after any rename) to capture
    now, from a dummy prediction; otherwise the first prediction captures them, taking a few
    seconds. Capturing up front matters under Real-Time Chunking, whose delay estimate uses the
    slowest recent inference.
    """
    if policy.name != "pi05":
        raise ValueError(f"Expected a PI0.5 policy, got policy type '{policy.name}'")
    device = next(policy.parameters()).device
    if device.type != "cuda":
        raise ValueError(f"CUDA graphs need a CUDA device, got {device}")
    sampler = GraphedSampler(policy.model, tf32_images=tf32_images)
    policy.model.sample_actions = sampler
    if present_image_keys is not None:
        _capture_with_dummy_prediction(policy, sampler, present_image_keys)
    return sampler


def _capture_with_dummy_prediction(
    policy: PI05Policy, sampler: GraphedSampler, present_image_keys: list[str]
) -> None:
    """Run one prediction on placeholder inputs shaped like ``_preprocess_images`` output.

    PI0.5 always fills every image slot (missing cameras with the ``-1`` constant, masked out), so
    the prefix shape does not depend on which cameras are present; their masks do.
    """
    config = policy.config
    device = next(policy.parameters()).device
    present = [key in present_image_keys for key in config.image_features]
    if not any(present):
        raise ValueError(f"None of {present_image_keys} are image features of this policy")
    height, width = config.image_resolution
    images = [
        torch.full((1, 3, height, width), 0.0 if is_present else -1.0, device=device)
        for is_present in present
    ]
    img_masks = [torch.tensor([is_present], device=device) for is_present in present]
    tokens = torch.zeros(1, config.tokenizer_max_length, dtype=torch.long, device=device)
    masks = torch.ones(1, config.tokenizer_max_length, dtype=torch.bool, device=device)
    sampler(images, img_masks, tokens, masks)
    logger.info("PI0.5 CUDA graphs captured (%d of %d camera slots present)", sum(present), len(present))
