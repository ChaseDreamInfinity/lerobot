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
"""A miniature PI0.5 for fast tests: same architecture, tiny widths, no Hub download.

``tiny_pi05`` is a fixture that patches the Gemma variants and the SigLIP vision config while a
test builds policies, so PI05Policy(config) and PI05Policy.from_pretrained(...) both produce the
miniature model.
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature

CAMERAS = ["observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb"]
IMAGE_SIZE = 28  # two 14-pixel patches per side


@pytest.fixture
def tiny_pi05(monkeypatch):
    pytest.importorskip("transformers")
    from lerobot.policies.pi05 import modeling_pi05

    variants = {
        "gemma_2b": modeling_pi05.GemmaConfig(
            width=64, depth=2, mlp_dim=128, num_heads=2, num_kv_heads=1, head_dim=32
        ),
        "gemma_300m": modeling_pi05.GemmaConfig(
            width=32, depth=2, mlp_dim=64, num_heads=2, num_kv_heads=1, head_dim=32
        ),
    }
    monkeypatch.setattr(modeling_pi05, "get_gemma_config", variants.__getitem__)

    config_mapping = modeling_pi05.CONFIG_MAPPING

    def paligemma_config():
        config = config_mapping["paligemma"]()
        vision = config.vision_config
        vision.hidden_size, vision.num_hidden_layers, vision.num_attention_heads = 32, 1, 2
        return config

    monkeypatch.setattr(
        modeling_pi05,
        "CONFIG_MAPPING",
        {"paligemma": paligemma_config, "gemma": config_mapping["gemma"]},
    )
    return make_config


def make_config(**overrides):
    from lerobot.policies.pi05 import PI05Config

    config = PI05Config(
        max_action_dim=8,
        max_state_dim=8,
        chunk_size=10,
        n_action_steps=10,
        num_inference_steps=3,
        tokenizer_max_length=16,
        image_resolution=(IMAGE_SIZE, IMAGE_SIZE),
        **overrides,
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)) for key in CAMERAS},
        "observation.images.right_wrist_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)
        ),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    return config


def sample_inputs(policy, seed: int):
    """``sample_actions`` inputs: two camera images, an empty third slot, padded tokens, noise."""
    device = next(policy.parameters()).device
    config = policy.config
    generator = torch.Generator(device=device).manual_seed(seed)
    images = [
        torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device, generator=generator) * 2 - 1 for _ in range(3)
    ]
    img_masks = [torch.tensor([present], device=device) for present in (True, True, False)]
    tokens = torch.randint(0, 1000, (1, config.tokenizer_max_length), device=device, generator=generator)
    masks = torch.zeros(1, config.tokenizer_max_length, dtype=torch.bool, device=device)
    masks[:, :10] = True
    noise = torch.randn(1, config.chunk_size, config.max_action_dim, device=device, generator=generator)
    return (images, img_masks, tokens, masks), noise
