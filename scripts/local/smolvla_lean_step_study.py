#!/usr/bin/env python
"""Study: a leaner ONNX export of one SmolVLA denoising step (Jetson Orin).

Rewrites the action expert's step without changing its math, and checks it is bit-exact against
the model's own step in fp32 and fp16 before exporting:
- the cross-attention layers' k/v projections of the prefix K/V, which only depend on the prefix,
  move out of the step (``project_cross_kv``, once per prediction);
- grouped-query attention without materializing the K/V head expansion;
- RoPE tables computed once per step, and a concat instead of slice assignment.
Also exports variants with the suffix padded to 56 and 64 tokens (exact for the 50 real tokens,
since the suffix attends causally). Results are in SMOLVLA_ORIN_OPTIMIZATION.md.

    uv run --no-sync --with onnx --with numpy==1.26.4 python scripts/local/smolvla_lean_step_study.py <out_dir>
    /usr/src/tensorrt/bin/trtexec --onnx=<out_dir>/step_fast.onnx --fp16 --useCudaGraph
"""

import copy
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

from lerobot.policies.smolvla import modeling_smolvla
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.tensorrt_smolvla import (
    KVFillExport,
    StepExport,
    _example_inputs,
    _step_inputs,
    fp16_copy,
)
from lerobot.utils.tensorrt import export_onnx

POLICY = "chasedreaminf/smolvla_so101-pick-color-block-in-cup-2cam-run2-v3"


def rope_tables(positions: Tensor, head_dim: int, max_wavelength: float = 10_000):
    d_half = head_dim // 2
    exponents = (2.0 / head_dim) * torch.arange(d_half, dtype=torch.float32, device=positions.device)
    radians = positions[..., None].to(torch.float32) / (max_wavelength**exponents)[None, None, :]
    radians = radians[..., None, :]
    return torch.cos(radians), torch.sin(radians)


def rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """``apply_rope`` with precomputed tables and a concat instead of slice assignment."""
    dtype = x.dtype
    x1, x2 = x.to(torch.float32).chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


def gqa_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor, n_kv: int) -> Tensor:
    """``eager_attention_forward`` without materializing the K/V head expansion.

    q: [B, L, H, D]; k, v: [B, S, KV, D]; mask: [B, L, S] bool. Query head h uses K/V head
    h // (H / KV), as in the expansion.
    """
    b, n, h, d = q.shape
    groups = h // n_kv
    q = q.to(torch.float32).view(b, n, n_kv, groups, d).permute(0, 2, 3, 1, 4)  # B KV G L D
    k = k.to(torch.float32).permute(0, 2, 3, 1)[:, :, None]  # B KV 1 D S
    scores = torch.matmul(q, k) * d**-0.5  # B KV G L S
    scores = torch.where(mask[:, None, None], scores, torch.finfo(torch.float32).min)
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    out = torch.matmul(probs, v.permute(0, 2, 1, 3)[:, :, None])  # B KV G L D
    return out.permute(0, 3, 1, 2, 4).reshape(b, n, h * d)


def project_cross_kv(model, kv: list[Tensor]) -> list[Tensor]:
    """Apply the expert's k/v projections to the prefix K/V of cross-attention layers (flat list)."""
    vwe = model.vlm_with_expert
    out = list(kv)
    for i, layer in enumerate(vwe.lm_expert.layers):
        if _is_self_attn(vwe, i):
            continue
        attn = layer.self_attn
        for j, proj in ((2 * i, attn.k_proj), (2 * i + 1, attn.v_proj)):
            t = kv[j]
            out[j] = proj(t.to(proj.weight.dtype).view(*t.shape[:2], -1)).view(
                *t.shape[:2], -1, attn.head_dim
            )
    return out


def _is_self_attn(vwe, i: int) -> bool:
    n = vwe.self_attn_every_n_layers
    return "cross" not in vwe.attention_mode or (n > 0 and i % n == 0)


class FastStepExport(nn.Module):
    """One denoising step; cross-attention layers read pre-projected K/V (``project_cross_kv``)."""

    def __init__(self, model, n_tokens: int | None = None):
        super().__init__()
        self.model = model
        self.n_tokens = n_tokens  # pad the suffix with trailing zero rows (never attended: causal)
        vwe = model.vlm_with_expert
        assert vwe.num_vlm_layers == vwe.num_expert_layers

    def forward(self, x_t: Tensor, timestep: Tensor, prefix_pad_masks: Tensor, *kv: Tensor) -> Tensor:
        model = self.model
        vwe = model.vlm_with_expert
        n_kv, head_dim = vwe.num_key_value_heads, vwe.vlm.config.text_config.head_dim
        suffix_embs, _, _ = model.embed_suffix(x_t, timestep)
        if self.n_tokens:
            suffix_embs = torch.nn.functional.pad(
                suffix_embs, (0, 0, 0, self.n_tokens - suffix_embs.shape[1])
            )
        b, n, _ = suffix_embs.shape
        p = prefix_pad_masks.shape[1]
        # Suffix tokens are all valid and each starts its own attention block (att_masks are all
        # ones), so they attend causally: the mask is the prefix padding followed by a lower triangle.
        causal = torch.ones(n, n, dtype=torch.bool, device=x_t.device).tril()
        mask = torch.cat([prefix_pad_masks[:, None, :].expand(b, n, p), causal.expand(b, n, n)], dim=2)
        offsets = torch.sum(prefix_pad_masks.to(torch.int32), dim=-1)[:, None]
        steps = torch.arange(n, device=x_t.device, dtype=torch.int32)[None]
        self_cos, self_sin = rope_tables((offsets + steps).to(torch.int64), head_dim)
        cross_cos, cross_sin = rope_tables(steps.to(torch.int64).expand(b, n), head_dim)

        hidden = suffix_embs
        for i, layer in enumerate(vwe.lm_expert.layers):
            attn = layer.self_attn
            h = layer.input_layernorm(hidden).to(attn.q_proj.weight.dtype)
            q = attn.q_proj(h).view(b, n, -1, head_dim)
            if _is_self_attn(vwe, i):
                k = attn.k_proj(h).view(b, n, -1, head_dim)
                v = attn.v_proj(h).view(b, n, -1, head_dim)
                q, k = rope(q, self_cos, self_sin), rope(k, self_cos, self_sin)
                k = torch.cat([kv[2 * i], k], dim=1)
                v = torch.cat([kv[2 * i + 1], v], dim=1)
                layer_mask = mask
            else:
                q = rope(q, cross_cos, cross_sin)
                k, v = kv[2 * i], kv[2 * i + 1]
                layer_mask = mask[:, :, :p]
            out = gqa_attention(q, k, v, layer_mask, n_kv).to(attn.o_proj.weight.dtype)
            out = attn.o_proj(out)
            out += hidden  # in place, as in the model: keeps o_proj's dtype
            residual = out.clone()
            out = layer.mlp(layer.post_attention_layernorm(out))
            out += residual
            hidden = out
        hidden = vwe.lm_expert.norm(hidden)
        return model.action_out_proj(hidden[:, : model.config.chunk_size].to(torch.float32))


@torch.no_grad()
def main(out_dir: Path):
    policy = SmolVLAPolicy.from_pretrained(POLICY).to("cuda").eval()
    model = policy.model
    images, img_masks, lt, lm, state = _example_inputs(policy, 2, "cuda")
    embs, pad, att = model.embed_prefix(images, img_masks, lt, lm, state=state)
    get_safe_dtype = modeling_smolvla.get_safe_dtype
    modeling_smolvla.get_safe_dtype = lambda dtype, device: torch.float32 if dtype == torch.float64 else dtype

    # Exactness check in fp32 on CPU.
    m32 = copy.deepcopy(model).float().cpu().eval()
    kv32 = KVFillExport(m32)(embs.float().cpu(), pad.cpu(), att.cpu())
    x = torch.randn(1, 50, 32)
    t = torch.tensor([0.7])
    ref = StepExport(m32)(x, t, pad.cpu(), *kv32)
    new = FastStepExport(m32)(x, t, pad.cpu(), *project_cross_kv(m32, kv32))
    print("fp32 max |diff|", (new - ref).abs().max().item(), "ref scale", ref.abs().max().item())
    del m32

    m16 = fp16_copy(model)
    kv = KVFillExport(m16)(embs.half(), pad, att)
    x_t = torch.randn(1, 50, 32, device="cuda")
    t = torch.tensor([0.7], device="cuda")
    ref16 = StepExport(m16)(x_t, t, pad, *kv)
    new16 = FastStepExport(m16)(x_t, t, pad, *project_cross_kv(m16, list(kv)))
    print("fp16 max |diff|", (new16 - ref16).abs().max().item())
    names = _step_inputs(len(m16.vlm_with_expert.lm_expert.layers))
    export_onnx(
        FastStepExport(m16),
        [x_t, t, pad, *project_cross_kv(m16, list(kv))],
        out_dir / "step_fast.onnx",
        names,
        ["v_t"],
    )
    kvp = project_cross_kv(m16, list(kv))
    for n in (56, 64):
        out = FastStepExport(m16, n)(x_t, t, pad, *kvp)
        print(n, "tokens: fp16 max |diff| vs model", (out - ref16).abs().max().item())
        export_onnx(
            FastStepExport(m16, n), [x_t, t, pad, *kvp], out_dir / f"step_fast{n}.onnx", names, ["v_t"]
        )
    modeling_smolvla.get_safe_dtype = get_safe_dtype
    print("exported", out_dir / "step_fast.onnx")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    main(out)
