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
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.tensorrt_act import (
    _ACTChunkModule,
    _example_inputs,
    attach_tensorrt_engine,
    build_engine,
    engine_input_keys,
    export_onnx,
    import_tensorrt,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.utils import require_cuda

WRIST = "observation.images.wrist"
TOP = "observation.images.top"


def make_policy(seed: int = 0) -> ACTPolicy:
    torch.manual_seed(seed)
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            WRIST: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 96)),
            TOP: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 96)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        chunk_size=10,
        n_action_steps=10,
        dim_model=64,
        n_heads=2,
        dim_feedforward=128,
        n_encoder_layers=1,
        pretrained_backbone_weights=None,
    )
    return ACTPolicy(config).eval()


def test_engine_input_keys_order():
    assert engine_input_keys(make_policy().config) == [OBS_STATE, WRIST, TOP]


@torch.no_grad()
def test_export_module_matches_policy():
    policy = make_policy()
    inputs = _example_inputs(policy.config, "cpu")
    batch = dict(zip(engine_input_keys(policy.config), inputs, strict=True))
    expected = policy.predict_action_chunk(batch)
    torch.testing.assert_close(_ACTChunkModule(policy)(*inputs), expected, rtol=0, atol=0)


@pytest.fixture
def tensorrt():
    pytest.importorskip("onnx")
    try:
        return import_tensorrt()
    except ImportError:
        pytest.skip("requires TensorRT")


@require_cuda
@torch.no_grad()
def test_tensorrt_engine_roundtrip(tensorrt, tmp_path):
    policy = make_policy().to("cuda")
    engine = build_engine(export_onnx(policy, tmp_path / "act.onnx"), tmp_path / "act.engine", fp16=False)

    batch = dict(zip(engine_input_keys(policy.config), _example_inputs(policy.config, "cuda"), strict=True))
    expected = policy.predict_action_chunk(batch)
    chunker = attach_tensorrt_engine(policy, engine)
    assert policy.predict_action_chunk is chunker
    actual = policy.predict_action_chunk(batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-3)

    # The output buffer is reused across calls; returned chunks must not alias it.
    policy.predict_action_chunk(
        dict(zip(chunker.input_keys, _example_inputs(policy.config, "cuda"), strict=True))
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-3)

    # select_action goes through the engine.
    policy.reset()
    assert policy.select_action(batch).shape == (1, 6)

    with pytest.raises(ValueError, match="engine was built for"):
        policy.predict_action_chunk({**batch, WRIST: torch.zeros(1, 3, 32, 32, device="cuda")})


@require_cuda
def test_attach_rejects_engine_from_other_checkpoint(tensorrt, tmp_path):
    engine = build_engine(
        export_onnx(make_policy(seed=0).to("cuda"), tmp_path / "act.onnx"),
        tmp_path / "act.engine",
        fp16=False,
    )
    with pytest.raises(RuntimeError, match="disagrees with the PyTorch policy"):
        attach_tensorrt_engine(make_policy(seed=1).to("cuda"), engine)
