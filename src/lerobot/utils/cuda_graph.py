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
"""Shared CUDA-graph helpers for the graphed samplers of flow-matching VLAs (SmolVLA, PI0.5)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def capture(fn: Callable[[], object], pool=None) -> tuple[torch.cuda.CUDAGraph, object]:
    """Warm ``fn`` up on a side stream, then capture it. Returns the graph and ``fn``'s output.

    Every tensor the graph reads must stay referenced for as long as the graph is replayed (keep
    the inputs next to the graph): a buffer held only by ``fn``'s closure can be freed and its
    memory reused, and the graph then silently reads whatever is there.
    """
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


class ConstantImageCache:
    """Stands in for a model's ``embed_image``, caching embeddings of constant inputs.

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


class EmptyImageSlots:
    """Swaps images equal to the empty-camera constant (all ``-1``) for one registered tensor, so
    their embedding comes from a ``ConstantImageCache``. The comparison is exact, so only inputs
    identical to the constant hit the cache."""

    def __init__(self, cache: ConstantImageCache):
        self.cache = cache
        self._empty_images: dict[tuple, Tensor] = {}

    def __call__(self, images: list[Tensor]) -> list[Tensor]:
        out = []
        for image in images:
            key = (tuple(image.shape), image.dtype, image.device)
            if key not in self._empty_images:
                with torch.inference_mode(False):
                    empty = torch.full(image.shape, -1.0, dtype=image.dtype, device=image.device)
                self.cache.register(empty)
                self._empty_images[key] = empty
            empty = self._empty_images[key]
            out.append(empty if torch.equal(image, empty) else image)
        return out
