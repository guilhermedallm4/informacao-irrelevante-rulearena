"""Hook placement across architectures (src/interp/activations.py).

Tiny random models on CPU, one per supported layout: Llama-style pre-norm
(Qwen2), hybrid conv/attention (LFM2), post-norm sandwich (OLMo 3) and
sandwich + layer scalar + logit softcap (Gemma 4). Each test checks an
identity that only holds if the hooks read the right tensors.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interp import activations as A  # noqa: E402

COMMON = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
              num_attention_heads=4, num_key_value_heads=2)


def build(kind):
    torch.manual_seed(0)
    if kind == "qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM
        return Qwen2ForCausalLM(Qwen2Config(**COMMON))
    if kind == "lfm2":
        from transformers import Lfm2Config, Lfm2ForCausalLM
        return Lfm2ForCausalLM(Lfm2Config(**COMMON, layer_types=["conv", "full_attention", "conv"],
                                          block_auto_adjust_ff_dim=False))
    if kind == "olmo3":
        from transformers import Olmo3Config, Olmo3ForCausalLM
        return Olmo3ForCausalLM(Olmo3Config(**COMMON))
    if kind == "gemma4":
        from transformers.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
        m = Gemma4ForCausalLM(Gemma4TextConfig(
            **COMMON, head_dim=8, global_head_dim=8, sliding_window=16, hidden_size_per_layer_input=0,
            layer_types=["sliding_attention", "full_attention", "full_attention"], final_logit_softcapping=3.0))
        for layer, s in zip(m.model.layers, [0.5, 2.0, 1.5]):    # exercise the scalar bookkeeping
            layer.layer_scalar.fill_(s)
        return m
    raise ValueError(kind)


class Tok:
    """Character-level stand-in for a tokenizer."""
    pad_token_id, eos_token_id = 0, 1

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[2 + ord(c) % 60 for c in text])


KINDS = ["qwen2", "lfm2", "olmo3", "gemma4"]
PREFIX, ANSWER = "The total cost is $", "1365"


@pytest.fixture(scope="module", params=KINDS)
def setup(request):
    model = build(request.param).float().eval()
    cap = A.capture(model, Tok(), PREFIX, ANSWER)
    return request.param, model, cap


def test_arch_resolution(setup):
    kind, model, _ = setup
    arch = A.resolve_arch(model)
    assert len(arch.layers) == 3
    if kind == "lfm2":
        assert arch.mixers == ["conv", "attention", "conv"]
    if kind in ("olmo3", "gemma4"):          # sandwich: the write is the post-norm output
        assert all("RMSNorm" in type(m).__name__ for m in arch.attn_write + arch.mlp_write)
    if kind == "gemma4":
        assert arch.softcap == 3.0
        assert arch.write_scale == pytest.approx([1.5, 3.0, 1.5])   # products of the later scalars
        assert arch.embed_scale == pytest.approx(1.5)


def test_residual_is_sum_of_captured_writes(setup):
    """resid[L] = embedding + every block write (with the layer scalars): hooks read the true writes."""
    _, model, cap = setup
    arch = A.resolve_arch(model)
    scale = torch.tensor(arch.write_scale or [1.0] * len(arch.layers)).view(-1, 1, 1)
    rebuilt = cap.resid[0] * arch.embed_scale + ((cap.attn_out + cap.mlp_out) * scale).sum(0)
    torch.testing.assert_close(rebuilt, cap.resid[-1], rtol=1e-4, atol=1e-4)
    assert cap.mlp_act.shape == (3, len(ANSWER), 64)


def test_logit_lens_last_layer_matches_model_output(setup):
    _, model, cap = setup
    lens = A.logit_lens(model, cap)
    tok = Tok()
    ids = torch.tensor([tok(PREFIX).input_ids + tok(ANSWER).input_ids])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()
    n = len(ANSWER)
    logp = torch.log_softmax(logits[-n - 1:-1], dim=-1)
    target = cap.target_ids
    expected = logp.gather(-1, target.view(-1, 1)).squeeze(-1).numpy()
    assert lens["target_logp"][-1] == pytest.approx(expected, abs=1e-4)


def test_dla_sums_to_final_logit_gap(setup):
    """Sum of the attributions = target logit - mean logit (before any softcap)."""
    _, model, cap = setup
    arch = A.resolve_arch(model)
    dla = A.direct_logit_attribution(model, cap)
    total = dla["attn"].sum(0) + dla["mlp"].sum(0) + dla["embed"]
    with torch.no_grad():
        h = arch.final_norm(cap.resid[-1])
        W = model.lm_head.weight.float()
        gap = (h * (W[cap.target_ids] - W.mean(0, keepdim=True))).sum(-1).numpy()
    assert total == pytest.approx(gap, rel=1e-3, abs=1e-3)


def test_left_padding_is_only_numerical_noise(setup):
    _, model, cap = setup
    padded = A.capture(model, Tok(), PREFIX, ANSWER, left_pad=4)
    diff = A.compare(cap, padded)
    assert diff["resid_rel_l2"].max() < 1e-4


def test_moe_models_are_refused():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    torch.manual_seed(0)
    m = Qwen3MoeForCausalLM(Qwen3MoeConfig(**COMMON, num_experts=4, num_experts_per_tok=2,
                                           moe_intermediate_size=16))
    with pytest.raises(NotImplementedError):
        A.resolve_arch(m)
