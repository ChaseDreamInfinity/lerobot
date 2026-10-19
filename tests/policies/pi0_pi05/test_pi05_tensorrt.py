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

from tests.policies.pi0_pi05.test_pi05_cuda_graph import make_policy
from tests.policies.pi0_pi05.tiny_pi05 import CAMERAS, sample_inputs, tiny_pi05  # noqa: F401
from tests.utils import require_cuda


@pytest.fixture
def engine_dir(tiny_pi05, tmp_path):  # noqa: F811
    pytest.importorskip("onnx")
    if not torch.cuda.is_available():
        pytest.skip("requires cuda")
    from lerobot.utils.tensorrt import import_tensorrt

    try:
        import_tensorrt()
    except ImportError:
        pytest.skip("requires TensorRT")
    from lerobot.policies.pi05.tensorrt_pi05 import build_engines

    return build_engines(make_policy(tiny_pi05, rtc=False), tmp_path / "engines", n_cameras=len(CAMERAS))


@require_cuda
@torch.no_grad()
def test_tensorrt_sampling_matches_eager(tiny_pi05, engine_dir):  # noqa: F811
    from lerobot.policies.pi05.cuda_graph_pi05 import GraphedSampler
    from lerobot.policies.pi05.tensorrt_pi05 import attach_tensorrt_engines

    for rtc in (False, True):  # one engine build for both
        reference = make_policy(tiny_pi05, rtc)  # same seed: same weights, stays fully eager
        policy = make_policy(tiny_pi05, rtc)
        sampler = attach_tensorrt_engines(policy, engine_dir, present_image_keys=CAMERAS)
        assert isinstance(policy.model.sample_actions, GraphedSampler)
        assert len(sampler._graphs) == 1  # graphs captured up front

        kwargs = {"inference_delay": 2, "execution_horizon": 5} if rtc else {}
        for seed in range(3):
            inputs, noise = sample_inputs(policy, seed)
            inputs[0][2] = torch.full_like(inputs[0][2], -1.0)  # the empty slot
            if rtc:
                kwargs["prev_chunk_left_over"] = torch.randn(1, 10, 6, device="cuda")
            expected = reference.model.sample_actions(*inputs, noise=noise, **kwargs)
            actual = policy.model.sample_actions(*inputs, noise=noise, **kwargs)
            # fp16 engines vs a bf16 model: compare relative to the actions' scale.
            assert ((actual - expected).abs().max() / expected.abs().max()).item() < 0.05, (rtc, seed)


@require_cuda
@torch.no_grad()
def test_attach_rejects_engines_from_other_checkpoint(tiny_pi05, engine_dir):  # noqa: F811
    from lerobot.policies.pi05.tensorrt_pi05 import attach_tensorrt_engines

    other = make_policy(tiny_pi05, rtc=False)
    torch.manual_seed(1)
    for parameter in other.model.paligemma_with_expert.parameters():
        parameter.data.normal_(0, 0.05)  # a different language model and action expert
    with pytest.raises(RuntimeError, match="disagrees with the PyTorch policy"):
        attach_tensorrt_engines(other, engine_dir)
