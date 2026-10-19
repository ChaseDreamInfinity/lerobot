#!/usr/bin/env python

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

import pytest
import torch

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.random_utils import set_seed
from tests.policies.pi0_pi05.tiny_pi05 import CAMERAS, sample_inputs, tiny_pi05  # noqa: F401
from tests.utils import require_cuda


def make_policy(make_config, rtc: bool):
    from lerobot.policies.pi05 import PI05Policy

    set_seed(0)
    config = make_config()
    if rtc:
        config.rtc_config = RTCConfig(enabled=True, execution_horizon=5)
    policy = PI05Policy(config).to("cuda").eval()
    if rtc:
        policy.init_rtc_processor()
    return policy


@require_cuda
@pytest.mark.parametrize("rtc", [False, True])
@torch.no_grad()
def test_graphed_sampling_matches_eager(tiny_pi05, rtc):  # noqa: F811
    from lerobot.policies.pi05.cuda_graph_pi05 import GraphedSampler, enable_cuda_graphs

    reference = make_policy(tiny_pi05, rtc)  # same seed: same weights, stays fully eager
    policy = make_policy(tiny_pi05, rtc)
    sampler = enable_cuda_graphs(policy, present_image_keys=CAMERAS, tf32_images=False)
    assert isinstance(policy.model.sample_actions, GraphedSampler)
    assert len(sampler._graphs) == 1  # captured up front, from the dummy prediction

    kwargs = {"inference_delay": 2, "execution_horizon": 5} if rtc else {}
    for seed in range(3):  # fresh observations and noise through the same static buffers
        inputs, noise = sample_inputs(policy, seed)
        inputs[0][2] = torch.full_like(inputs[0][2], -1.0)  # the empty slot, as _preprocess_images pads it
        if rtc:
            kwargs["prev_chunk_left_over"] = torch.randn(1, 10, 6, device="cuda")
        expected = reference.model.sample_actions(*inputs, noise=noise, **kwargs)
        actual = policy.model.sample_actions(*inputs, noise=noise, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
    assert len(sampler._graphs) == 1  # no recapture for the same prefix shape
    assert len(sampler.image_cache.constants) == 1  # the padded empty-camera slot is encoded once


@require_cuda
@torch.no_grad()
def test_tf32_images_stay_close_to_fp32(tiny_pi05):  # noqa: F811
    from lerobot.policies.pi05.cuda_graph_pi05 import enable_cuda_graphs

    reference = make_policy(tiny_pi05, rtc=False)
    policy = make_policy(tiny_pi05, rtc=False)
    enable_cuda_graphs(policy, present_image_keys=CAMERAS)  # TF32 image encoding by default
    tf32_before = torch.backends.cuda.matmul.allow_tf32
    inputs, noise = sample_inputs(policy, 0)
    expected = reference.model.sample_actions(*inputs, noise=noise)
    actual = policy.model.sample_actions(*inputs, noise=noise)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.02 * expected.abs().max().item())
    assert torch.backends.cuda.matmul.allow_tf32 == tf32_before  # scoped to image encoding


@require_cuda
@torch.no_grad()
def test_other_batch_sizes_run_eagerly(tiny_pi05):  # noqa: F811
    from lerobot.policies.pi05.cuda_graph_pi05 import enable_cuda_graphs

    reference = make_policy(tiny_pi05, rtc=False)
    policy = make_policy(tiny_pi05, rtc=False)
    sampler = enable_cuda_graphs(policy, tf32_images=False)  # lazy: nothing captured yet
    (images, img_masks, tokens, masks), noise = sample_inputs(policy, 0)
    batched = (
        [img.repeat(2, 1, 1, 1) for img in images],
        [mask.repeat(2) for mask in img_masks],
        tokens.repeat(2, 1),
        masks.repeat(2, 1),
    )
    expected = reference.model.sample_actions(*batched, noise=noise.repeat(2, 1, 1))
    torch.testing.assert_close(policy.model.sample_actions(*batched, noise=noise.repeat(2, 1, 1)), expected)
    assert len(sampler._graphs) == 0


def test_enable_rejects_other_policies():
    from lerobot.policies.pi05.cuda_graph_pi05 import enable_cuda_graphs

    class NotPI05:
        name = "smolvla"

    with pytest.raises(ValueError, match="Expected a PI0.5 policy"):
        enable_cuda_graphs(NotPI05())
