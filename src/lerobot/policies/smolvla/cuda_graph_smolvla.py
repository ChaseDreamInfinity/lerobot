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
"""CUDA graphs for SmolVLA action sampling.

SmolVLA's sampling is launch-bound on small GPUs such as Jetson: each denoising step runs the
action expert over only ``chunk_size`` tokens, so the GPU mostly waits for the CPU to issue
hundreds of small kernels (~56 ms per step eager vs ~6 ms replayed on an AGX Orin).

Two graphs are captured per prefix shape, on first use:

- the prefix KV-cache fill (the VLM pass over image, language and state tokens);
- one denoising step of the action expert, reading that KV cache in place.

Image encoding stays eager: it is GPU-bound (~14.5 ms per 512x512 image graphed vs ~16 ms eager),
and ``transformers``' SigLIP embeddings would need patching to be capturable. Empty camera slots,
the constant image ``prepare_images`` pads missing cameras with, are encoded once and cached
(saving one image encode per prediction). Each image is compared with the constant on the device,
so the cache is only used for an input identical to the one it was computed from.

The denoising loop stays in Python, so Real-Time Chunking guidance works unchanged: it calls the
graphed step as its denoiser. (Its correction does not differentiate through the network, since
``v_t`` is computed before ``x_t`` requires grad, so a replayed ``v_t`` gives the same result.)

Only batch size 1 is graphed; anything else runs the original eager code.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from torch import Tensor

from .modeling_smolvla import SmolVLAPolicy, VLAFlowMatching, make_att_2d_masks

logger = logging.getLogger(__name__)


def _capture(fn: Callable[[], object], pool=None) -> tuple[torch.cuda.CUDAGraph, object]:
    """Warm ``fn`` up on a side stream, then capture it. Returns the graph and ``fn``'s output."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        output = fn()
    return graph, output


class _ConstantImageCache:
    """Stands in for ``SmolVLMWithExpertModel.embed_image``, caching embeddings of constant inputs.

    Constants are matched by tensor identity: callers swap an input for the registered tensor only
    after checking the values are equal.
    """

    def __init__(self, embed_image: Callable[[Tensor], Tensor]):
        self.embed_image = embed_image
        self.constants: dict[int, tuple[Tensor, Tensor]] = {}
        # Embeddings computed ahead of ``embed_prefix`` for one prediction (batched encoders).
        self.precomputed: dict[int, tuple[Tensor, Tensor]] = {}

    def __call__(self, image: Tensor) -> Tensor:
        for table in (self.constants, self.precomputed):
            hit = table.get(id(image))
            if hit is not None and hit[0] is image:
                return hit[1]
        return self.embed_image(image)

    def register(self, image: Tensor) -> None:
        if id(image) not in self.constants:
            self.constants[id(image)] = (image, self.embed_image(image))


# Backends for the two graphed stages. ``FillKV(embs, pad_masks, att_masks) -> kv`` builds the
# prefix KV cache in whatever form ``Denoise(x_t, timestep, pad_masks, kv) -> v_t`` reads.
FillKV = Callable[[Tensor, Tensor, Tensor], object]
Denoise = Callable[[Tensor, Tensor, Tensor, object], Tensor]


def pytorch_fill_kv(model: VLAFlowMatching) -> FillKV:
    def fill_kv(embs: Tensor, pad_masks: Tensor, att_masks: Tensor):
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(pad_masks, att_masks),
            position_ids=torch.cumsum(pad_masks, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        return past_key_values

    return fill_kv


def pytorch_denoise(model: VLAFlowMatching) -> Denoise:
    def denoise(x_t: Tensor, timestep: Tensor, pad_masks: Tensor, past_key_values) -> Tensor:
        return model.denoise_step(
            x_t=x_t, prefix_pad_masks=pad_masks, past_key_values=past_key_values, timestep=timestep
        )

    return denoise


class _PrefixGraphs:
    """Static buffers and the two graphs (KV fill, one denoising step) for one prefix shape."""

    def __init__(
        self,
        model: VLAFlowMatching,
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

        self.kv_graph, self.past_key_values = _capture(
            lambda: fill_kv(self.embs, self.pad_masks, self.att_masks)
        )
        self.kv_graph.replay()  # real cache contents for the denoiser warm-up
        self.step_graph, self.v_t = _capture(
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
    """Drop-in replacement for ``VLAFlowMatching.sample_actions``.

    The KV fill and denoising step run from CUDA graphs over ``fill_kv`` and ``denoise`` (the
    PyTorch model by default; TensorRT engines in ``tensorrt_smolvla``). ``encode_images``, if
    given, encodes all non-constant camera images of a prediction in one batched call.
    """

    def __init__(
        self,
        model: VLAFlowMatching,
        fill_kv: FillKV | None = None,
        denoise: Denoise | None = None,
        encode_images: Callable[[Tensor], Tensor] | None = None,
    ):
        self.model = model
        self.eager = model.sample_actions
        self.fill_kv = fill_kv or pytorch_fill_kv(model)
        self.denoise = denoise or pytorch_denoise(model)
        self.encode_images = encode_images
        self._graphs: dict[tuple, _PrefixGraphs] = {}
        self._empty_images: dict[tuple, Tensor] = {}

        self.image_cache = _ConstantImageCache(model.vlm_with_expert.embed_image)
        model.vlm_with_expert.embed_image = self.image_cache

    def _with_cached_constants(self, images: list[Tensor]) -> list[Tensor]:
        """Swap images equal to the empty-slot constant for one registered object, so their
        embedding comes from the cache. The comparison is exact, so only identical inputs hit it."""
        out = []
        for image in images:
            key = (tuple(image.shape), image.dtype, image.device)
            if key not in self._empty_images:
                with torch.inference_mode(False):
                    empty = torch.full(image.shape, -1.0, dtype=image.dtype, device=image.device)
                self.image_cache.register(empty)
                self._empty_images[key] = empty
            empty = self._empty_images[key]
            out.append(empty if torch.equal(image, empty) else image)
        return out

    @torch.no_grad()
    def __call__(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs) -> Tensor:
        model = self.model
        config = model.config
        bsize = state.shape[0]
        if bsize != 1 or not config.use_cache:
            return self.eager(images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs)
        device = state.device

        if noise is None:
            noise = model.sample_noise((bsize, config.chunk_size, config.max_action_dim), device)

        images = self._with_cached_constants(images)
        if self.encode_images is not None:
            real = [image for image in images if id(image) not in self.image_cache.constants]
            if real:
                embeddings = self.encode_images(torch.cat(real))
                self.image_cache.precomputed = {
                    id(image): (image, embeddings[i : i + 1]) for i, image in enumerate(real)
                }
        try:
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
        finally:
            self.image_cache.precomputed = {}
        key = (tuple(prefix_embs.shape), prefix_embs.dtype)
        if key not in self._graphs:
            logger.info("Capturing SmolVLA CUDA graphs for prefix shape %s", tuple(prefix_embs.shape))
            # Static buffers must be normal tensors, not inference tensors: they are written in
            # place on every call, whatever grad mode the caller (e.g. RTC guidance) is in.
            with torch.inference_mode(False):
                self._graphs[key] = _PrefixGraphs(
                    model, prefix_embs, prefix_pad_masks, prefix_att_masks, self.fill_kv, self.denoise
                )
        graphs = self._graphs[key]
        graphs.load_prefix(prefix_embs, prefix_pad_masks, prefix_att_masks)

        # Same loop as VLAFlowMatching.sample_actions, with the graphed denoising step.
        num_steps = config.num_steps
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


def enable_cuda_graphs(policy: SmolVLAPolicy, present_image_keys: list[str] | None = None) -> GraphedSampler:
    """Replay SmolVLA's KV-cache fill and denoising steps from CUDA graphs.

    Graphs are captured per prefix shape, which depends on how many camera images the policy
    receives. Pass ``present_image_keys`` (the image keys the policy will see, after any rename) to
    capture now, from a dummy prediction; otherwise the first prediction captures them, taking a
    few seconds. Capturing up front matters under Real-Time Chunking, whose delay estimate uses the
    slowest recent inference.
    """
    if policy.name != "smolvla":
        raise ValueError(f"Expected a SmolVLA policy, got policy type '{policy.name}'")
    device = next(policy.parameters()).device
    if device.type != "cuda":
        raise ValueError(f"CUDA graphs need a CUDA device, got {device}")
    sampler = GraphedSampler(policy.model)
    policy.model.sample_actions = sampler
    if present_image_keys is not None:
        _capture_with_dummy_prediction(policy, sampler, present_image_keys)
    return sampler


def _capture_with_dummy_prediction(
    policy: SmolVLAPolicy, sampler: GraphedSampler, present_image_keys: list[str]
) -> None:
    """Run one prediction on placeholder inputs shaped like ``prepare_images`` output."""
    config = policy.config
    device = next(policy.parameters()).device
    n_present = sum(key in present_image_keys for key in config.image_features)
    n_missing = len(config.image_features) - n_present
    n_images = n_present + min(n_missing, config.empty_cameras)  # as in SmolVLAPolicy.prepare_images
    if n_images == 0:
        raise ValueError(f"None of {present_image_keys} are image features of this policy")
    if config.resize_imgs_with_padding is not None:
        height, width = config.resize_imgs_with_padding
    else:
        height, width = next(iter(config.image_features.values())).shape[1:]
    images = [torch.zeros(1, 3, height, width, device=device) for _ in range(n_images)]
    img_masks = [torch.ones(1, dtype=torch.bool, device=device) for _ in range(n_images)]
    lang_tokens = torch.zeros(1, config.tokenizer_max_length, dtype=torch.long, device=device)
    lang_masks = torch.ones(1, config.tokenizer_max_length, dtype=torch.bool, device=device)
    state = torch.zeros(1, config.max_state_dim, device=device)
    sampler(images, img_masks, lang_tokens, lang_masks, state)
    logger.info("SmolVLA CUDA graphs captured (%d camera images per prediction)", n_images)
