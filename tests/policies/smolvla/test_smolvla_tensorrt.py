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

from tests.policies.smolvla.test_smolvla_cuda_graph import CAMERAS, make_policy, sample_inputs
from tests.utils import require_cuda, skip_if_package_missing


@pytest.fixture(scope="module")
def engine_dir(tmp_path_factory):
    pytest.importorskip("onnx")
    if not torch.cuda.is_available():
        pytest.skip("requires cuda")
    from lerobot.utils.tensorrt import import_tensorrt

    try:
        import_tensorrt()
    except ImportError:
        pytest.skip("requires TensorRT")
    from lerobot.policies.smolvla.tensorrt_smolvla import build_engines

    return build_engines(
        make_policy(rtc=False), tmp_path_factory.mktemp("smolvla_trt"), n_cameras=len(CAMERAS)
    )


@skip_if_package_missing("transformers")
@require_cuda
@pytest.mark.parametrize("rtc", [False, True])
@torch.no_grad()
def test_tensorrt_sampling_matches_eager(engine_dir, rtc):
    from lerobot.policies.smolvla.cuda_graph_smolvla import GraphedSampler
    from lerobot.policies.smolvla.tensorrt_smolvla import attach_tensorrt_engines

    reference = make_policy(rtc)  # same seed: same weights, stays fully eager
    policy = make_policy(rtc)
    sampler = attach_tensorrt_engines(policy, engine_dir, present_image_keys=CAMERAS)
    assert isinstance(policy.model.sample_actions, GraphedSampler)
    assert len(sampler._graphs) == 1  # graphs captured up front

    kwargs = {"inference_delay": 2, "execution_horizon": 5} if rtc else {}
    for seed in range(3):
        inputs, noise = sample_inputs(policy, seed)
        if rtc:
            kwargs["prev_chunk_left_over"] = torch.randn(1, 20, 6, device="cuda")
        expected = reference.model.sample_actions(*inputs, noise=noise, **kwargs)
        actual = policy.model.sample_actions(*inputs, noise=noise, **kwargs)
        # fp16 engines vs a bf16 model: compare relative to the actions' scale.
        assert ((actual - expected).abs().max() / expected.abs().max()).item() < 0.05


@skip_if_package_missing("transformers")
@require_cuda
def test_attach_rejects_engines_from_other_checkpoint(engine_dir):
    from lerobot.policies.smolvla.tensorrt_smolvla import attach_tensorrt_engines

    other = make_policy(rtc=False)
    torch.manual_seed(1)
    for parameter in other.model.vlm_with_expert.lm_expert.parameters():
        parameter.data.normal_(0, 0.05)  # a different action expert
    with pytest.raises(RuntimeError, match="disagrees with the PyTorch policy"):
        attach_tensorrt_engines(other, engine_dir)
