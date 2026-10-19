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

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.random_utils import set_seed
from tests.utils import require_cuda, skip_if_package_missing

CAMERAS = ["observation.images.camera1", "observation.images.camera2"]


def make_policy(rtc: bool):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    set_seed(0)
    config = SmolVLAConfig(
        max_action_dim=6, chunk_size=20, n_action_steps=20, num_steps=4, num_vlm_layers=2, empty_cameras=1
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)) for key in CAMERAS},
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    config.__post_init__()  # adds the empty camera slot to the image features
    if rtc:
        config.rtc_config = RTCConfig(enabled=True, execution_horizon=5)
    policy = SmolVLAPolicy(config).to("cuda").eval()
    if rtc:
        policy.init_rtc_processor()
    return policy


def sample_inputs(policy, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    config = policy.config
    batch = {key: torch.rand(1, 3, 64, 64, device="cuda", generator=generator) for key in CAMERAS}
    batch["observation.state"] = torch.randn(1, 6, device="cuda", generator=generator)
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = torch.randint(0, 1000, (1, config.tokenizer_max_length), device="cuda", generator=generator)
    lang_masks = torch.ones(1, config.tokenizer_max_length, dtype=torch.bool, device="cuda")
    lang_masks[:, 10:] = False
    noise = torch.randn(1, config.chunk_size, config.max_action_dim, device="cuda", generator=generator)
    return (images, img_masks, lang_tokens, lang_masks, state), noise


@skip_if_package_missing("transformers")
@require_cuda
@pytest.mark.parametrize("rtc", [False, True])
@torch.no_grad()
def test_graphed_sampling_matches_eager(rtc):
    from lerobot.policies.smolvla.cuda_graph_smolvla import GraphedSampler, enable_cuda_graphs

    policy = make_policy(rtc)
    eager = policy.model.sample_actions
    sampler = enable_cuda_graphs(policy, present_image_keys=CAMERAS)
    assert isinstance(policy.model.sample_actions, GraphedSampler)
    assert len(sampler._graphs) == 1  # captured up front, from the dummy prediction

    kwargs = {}
    if rtc:
        kwargs = {"inference_delay": 2, "execution_horizon": 5}
    for seed in range(3):  # fresh observations and noise through the same static buffers
        inputs, noise = sample_inputs(policy, seed)
        if rtc:
            kwargs["prev_chunk_left_over"] = torch.randn(1, 20, 6, device="cuda")
        expected = eager(*inputs, noise=noise, **kwargs)
        actual = policy.model.sample_actions(*inputs, noise=noise, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
    assert len(sampler._graphs) == 1  # no recapture for the same prefix shape


@skip_if_package_missing("transformers")
@require_cuda
@torch.no_grad()
def test_other_batch_sizes_run_eagerly():
    from lerobot.policies.smolvla.cuda_graph_smolvla import enable_cuda_graphs

    policy = make_policy(rtc=False)
    eager = policy.model.sample_actions
    sampler = enable_cuda_graphs(policy)  # lazy: nothing captured yet
    (images, img_masks, lang_tokens, lang_masks, state), noise = sample_inputs(policy, 0)
    batched = (
        [img.repeat(2, 1, 1, 1) for img in images],
        [mask.repeat(2) for mask in img_masks],
        lang_tokens.repeat(2, 1),
        lang_masks.repeat(2, 1),
        state.repeat(2, 1),
    )
    expected = eager(*batched, noise=noise.repeat(2, 1, 1))
    torch.testing.assert_close(policy.model.sample_actions(*batched, noise=noise.repeat(2, 1, 1)), expected)
    assert len(sampler._graphs) == 0


def test_enable_rejects_other_policies():
    from lerobot.policies.smolvla.cuda_graph_smolvla import enable_cuda_graphs

    class NotSmolVLA:
        name = "act"

    with pytest.raises(ValueError, match="Expected a SmolVLA policy"):
        enable_cuda_graphs(NotSmolVLA())
