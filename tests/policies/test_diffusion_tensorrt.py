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
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.tensorrt_diffusion import (
    _EncoderModule,
    _example_batch,
    _TensorRTUNet,
    _UNetModule,
    attach_tensorrt_engines,
    build_engines,
    encoder_input_names,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.tensorrt import import_tensorrt
from tests.utils import require_cuda


def make_policy(seed: int = 0, scheduler: str = "DDIM") -> DiffusionPolicy:
    torch.manual_seed(seed)
    config = DiffusionConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 96)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 96)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        crop_shape=None,
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        spatial_softmax_num_keypoints=8,
        noise_scheduler_type=scheduler,
        num_inference_steps=4,
        pretrained_backbone_weights=None,
    )
    return DiffusionPolicy(config).eval()


def test_encoder_input_names():
    assert encoder_input_names(make_policy().config) == ["state", "images"]


@torch.no_grad()
def test_export_modules_match_policy():
    policy = make_policy()
    model = policy.diffusion
    batch = _example_batch(policy.config, "cpu")
    expected = model._prepare_global_conditioning(batch)
    actual = _EncoderModule(model, ["state", "images"])(batch[OBS_STATE], batch[OBS_IMAGES])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    x, t = torch.randn(1, policy.config.horizon, 6), torch.tensor([3])
    torch.testing.assert_close(_UNetModule(model)(x, t, expected), model.unet(x, t, global_cond=expected))


@pytest.fixture
def tensorrt():
    pytest.importorskip("onnx")
    try:
        return import_tensorrt()
    except ImportError:
        pytest.skip("requires TensorRT")


@require_cuda
@torch.no_grad()
def test_tensorrt_engines_roundtrip(tensorrt, tmp_path):
    policy = make_policy().to("cuda")
    engine_dir = build_engines(policy, tmp_path / "engines", fp16=False)
    assert sorted(p.name for p in engine_dir.iterdir()) == ["rgb_encoder.engine", "unet.engine"]

    batch = _example_batch(policy.config, "cuda")
    noise = torch.randn(1, policy.config.horizon, 6, device="cuda")
    expected = policy.diffusion.generate_actions(batch, noise=noise)  # DDIM: deterministic given noise

    attach_tensorrt_engines(policy, engine_dir)
    assert isinstance(policy.diffusion.unet, _TensorRTUNet)
    torch.testing.assert_close(
        policy.diffusion.generate_actions(batch, noise=noise), expected, rtol=0, atol=1e-3
    )

    # select_action (observation queues + action queue) goes through the engines.
    policy.reset()
    obs = {
        OBS_STATE: torch.randn(1, 6, device="cuda"),
        "observation.images.wrist": torch.rand(1, 3, 64, 96, device="cuda"),
        "observation.images.top": torch.rand(1, 3, 64, 96, device="cuda"),
    }
    assert policy.select_action(obs).shape == (1, 6)


@require_cuda
def test_attach_rejects_engines_from_other_checkpoint(tensorrt, tmp_path):
    engine_dir = build_engines(make_policy(seed=0).to("cuda"), tmp_path / "engines", fp16=False)
    with pytest.raises(RuntimeError, match="disagrees with the PyTorch policy"):
        attach_tensorrt_engines(make_policy(seed=1).to("cuda"), engine_dir)
