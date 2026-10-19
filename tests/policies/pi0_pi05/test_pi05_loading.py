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

import torch
from safetensors.torch import load_file, save_file

from lerobot.utils.random_utils import set_seed
from tests.policies.pi0_pi05.tiny_pi05 import sample_inputs, tiny_pi05  # noqa: F401


def _state(policy):
    return {**dict(policy.named_parameters()), **dict(policy.named_buffers())}


@torch.no_grad()
def test_from_pretrained_matches_saved_policy(tiny_pi05, tmp_path):  # noqa: F811
    from lerobot.policies.pi05 import PI05Policy

    set_seed(0)
    saved = PI05Policy(tiny_pi05(device="cpu")).eval()
    saved.save_pretrained(tmp_path)

    loaded = PI05Policy.from_pretrained(tmp_path, config=tiny_pi05(device="cpu")).eval()
    expected, actual = _state(saved), _state(loaded)
    assert expected.keys() == actual.keys()
    for name, tensor in expected.items():
        assert torch.equal(actual[name], tensor), name
    inputs, noise = sample_inputs(saved, seed=0)
    assert torch.equal(
        loaded.model.sample_actions(*inputs, noise=noise), saved.model.sample_actions(*inputs, noise=noise)
    )


@torch.no_grad()
def test_from_pretrained_initializes_parameters_missing_from_checkpoint(tiny_pi05, tmp_path):  # noqa: F811
    from lerobot.policies.pi05 import PI05Policy

    set_seed(0)
    PI05Policy(tiny_pi05(device="cpu")).save_pretrained(tmp_path)
    weights = tmp_path / "model.safetensors"
    state_dict = {
        key: value.clone() for key, value in load_file(weights).items()
    }  # mmap: copy before overwriting
    missing = "model.action_out_proj.weight"
    del state_dict[missing]
    save_file(state_dict, weights)

    loaded = PI05Policy.from_pretrained(tmp_path, config=tiny_pi05(device="cpu"), strict=False)
    parameters = dict(loaded.named_parameters())
    # Initialized by the fallback, not left as skipped-initialization memory (often all zeros).
    weight = parameters[missing]
    bound = weight.shape[1] ** -0.5  # nn.Linear's default init range
    assert torch.isfinite(weight).all() and 0 < weight.abs().max() <= bound
    assert torch.equal(parameters["model.action_in_proj.weight"], state_dict["model.action_in_proj.weight"])
