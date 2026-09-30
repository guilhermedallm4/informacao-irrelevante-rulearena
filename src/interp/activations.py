"""Residual-stream and MLP-neuron capture for decoder-only HF models.

One teacher-forced forward pass over `prefix + answer` records, for every layer
and for each position that predicts an answer token:

  - resid[l]:    residual stream after layer l (l = 0 is the embedding);
  - attn_out[l], mlp_out[l]: what the attention block and the MLP of layer l
                 write into the residual stream;
  - mlp_act[l]:  MLP neuron activations (input of down_proj,
                 act(gate) * up), one value per neuron.

From these it derives a logit lens (the answer token read out from every
layer), the direct logit attribution of each sublayer to the answer token,
and the "convergence layer" where the answer becomes the top-1 prediction and
stays there.

Only the last few positions are kept, so a ~7k-token pass costs a few MB.

Architectures (see `resolve_arch`): the Llama/Qwen/Mistral/Phi-3 layout
(pre-norm), post-norm "sandwich" layouts where each block's output is
normalized before it is added to the residual (OLMo 2/3, Gemma 2-4: the
write is the output of post_attention_layernorm / post_feedforward_layernorm),
Gemma 4's per-layer residual scalar and final logit softcap, and LFM2's
hybrid layers (short convolution or attention as the token mixer; the
"attention" metrics then describe whichever mixer the layer has). Mixture-
of-experts MLPs are refused: they have no single set of neurons per layer.

Works with models split across several GPUs (device_map="auto"): inputs go to
the embedding's device and every captured tensor is gathered on the
unembedding's device, where the logit lens and attribution run.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch


def load_model(model_id: str, dtype=torch.bfloat16, device_map: str = "auto"):
    """Load a causal LM; "auto" spreads it over all visible GPUs if needed."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=device_map, attn_implementation="sdpa")
    model.eval()
    return model, tok


@dataclass
class Arch:
    """Where each quantity lives in a given model (resolved once per model)."""
    embed: torch.nn.Module          # output = residual before layer 1
    layers: list                    # decoder layers; output = residual after the layer
    final_norm: torch.nn.Module     # norm applied before the unembedding
    attn_write: list                # per layer: module whose output the mixer adds to the residual
    mlp_write: list                 # per layer: module whose output the MLP adds to the residual
    neuron_in: list                 # per layer: module whose *input* is the MLP neuron activations
    write_scale: list | None        # per layer: factor applied to that layer's writes by the end
    embed_scale: float              # same factor for the embedding
    softcap: float | None           # final logit softcapping (Gemma)
    mixers: list                    # per layer: "attention" or "conv"


def _text_config(model):
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def _attr(obj, *names):
    """First attribute of `obj` among `names` that exists and is not None."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def resolve_arch(model) -> Arch:
    if getattr(model, "_interp_arch", None) is not None:
        return model._interp_arch
    base = model.model
    if hasattr(base, "language_model"):         # multimodal wrappers (Gemma 3/4, Mistral 3)
        base = base.language_model
    final_norm = _attr(base, "norm", "embedding_norm")     # LFM2 calls it embedding_norm
    if final_norm is None:
        raise NotImplementedError(f"{type(model).__name__}: norma final não encontrada")
    attn_write, mlp_write, neuron_in, scalars, mixers = [], [], [], [], []
    for i, layer in enumerate(base.layers):
        if getattr(layer, "enable_moe_block", False) or getattr(layer, "hidden_size_per_layer_input", 0):
            raise NotImplementedError(f"camada {i}: bloco MoE / per-layer input não suportado")
        mixer = _attr(layer, "self_attn", "conv")          # LFM2: attention or short conv
        mlp = _attr(layer, "mlp", "feed_forward")
        if mixer is None or mlp is None:
            raise NotImplementedError(f"camada {i}: atenção/MLP não encontradas ({type(layer).__name__})")
        down = _attr(mlp, "down_proj", "w2")
        if down is None:
            raise NotImplementedError(f"camada {i}: MLP sem down_proj (MoE?) ({type(mlp).__name__})")
        # Llama-style layers also have a "post_attention_layernorm", but it is the
        # pre-MLP norm. Only sandwich layers have post_feedforward_layernorm, and
        # there both post-norms sit between the block and the residual addition.
        if hasattr(layer, "post_feedforward_layernorm"):
            attn_write.append(layer.post_attention_layernorm)
            mlp_write.append(layer.post_feedforward_layernorm)
        else:
            attn_write.append(mixer)
            mlp_write.append(mlp)
        neuron_in.append(down)
        mixers.append("conv" if mixer is getattr(layer, "conv", None) else "attention")
        s = getattr(layer, "layer_scalar", None)
        scalars.append(float(s.float().item()) if s is not None else 1.0)
    # layer_scalar multiplies the residual at the end of its layer, after that
    # layer's writes: a write from layer l is scaled by the product of l..L
    tail, write_scale = 1.0, [1.0] * len(scalars)
    for i in range(len(scalars) - 1, -1, -1):
        tail *= scalars[i]
        write_scale[i] = tail
    arch = Arch(embed=base.embed_tokens, layers=list(base.layers), final_norm=final_norm,
                attn_write=attn_write, mlp_write=mlp_write, neuron_in=neuron_in,
                write_scale=None if all(s == 1.0 for s in scalars) else write_scale,
                embed_scale=tail, softcap=getattr(_text_config(model), "final_logit_softcapping", None),
                mixers=mixers)
    model._interp_arch = arch
    return arch


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def output_device(model) -> torch.device:
    return model.lm_head.weight.device


@dataclass
class Capture:
    """Activations at the positions predicting each answer token."""
    resid: torch.Tensor      # [L+1, n, d]
    attn_out: torch.Tensor   # [L, n, d]
    mlp_out: torch.Tensor    # [L, n, d]
    mlp_act: torch.Tensor    # [L, n, d_ff]
    target_ids: torch.Tensor  # [n] answer token ids
    n_prefix_tokens: int


def _first(x):
    return x[0] if isinstance(x, tuple) else x


@contextmanager
def _hooks(model, keep: int, store: dict):
    arch = resolve_arch(model)
    dev = output_device(model)
    handles = []

    # .to(dev, copy=True): a slice is a view that would keep the full-sequence
    # tensor alive; copying also gathers layers that live on other GPUs
    def save(name, idx):
        def hook(_mod, _inp, out):
            store.setdefault(name, {})[idx] = _first(out)[0, -keep:].detach().to(dev, copy=True)
        return hook

    def save_input(name, idx):
        def hook(_mod, inp):
            store.setdefault(name, {})[idx] = inp[0][0, -keep:].detach().to(dev, copy=True)
        return hook

    handles.append(arch.embed.register_forward_hook(save("resid", 0)))
    for i, layer in enumerate(arch.layers):
        handles.append(layer.register_forward_hook(save("resid", i + 1)))
        handles.append(arch.attn_write[i].register_forward_hook(save("attn_out", i)))
        handles.append(arch.mlp_write[i].register_forward_hook(save("mlp_out", i)))
        handles.append(arch.neuron_in[i].register_forward_pre_hook(save_input("mlp_act", i)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def capture(model, tok, prefix_text: str, answer_text: str, left_pad: int = 0) -> Capture:
    """Teacher-forced pass over prefix + answer; keep answer-predicting positions.

    The prefix and the answer are tokenized separately so the boundary is
    exact. Position p predicts token p+1, so the n answer tokens are
    predicted by the n positions ending just before the last token.

    left_pad > 0 prepends that many masked padding tokens (with position ids
    that ignore them). The text seen by the model is identical, only the
    tensor shapes change, so the difference to left_pad=0 measures pure
    numerical noise, the same kind that batched generation introduces.
    """
    prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids
    answer_ids = tok(answer_text, add_special_tokens=False).input_ids
    real = prefix_ids + answer_ids
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dev_in = input_device(model)
    ids = torch.tensor([[pad_id] * left_pad + real], device=dev_in)
    kwargs = {}
    if left_pad:
        kwargs["attention_mask"] = torch.tensor([[0] * left_pad + [1] * len(real)], device=dev_in)
        kwargs["position_ids"] = torch.tensor([[0] * left_pad + list(range(len(real)))], device=dev_in)
    n = len(answer_ids)
    store: dict = {}
    with _hooks(model, keep=n + 1, store=store):
        model.model(input_ids=ids, use_cache=False, **kwargs)

    def stack(name):
        d = store[name]
        return torch.stack([d[i][:n] for i in sorted(d)]).float()

    return Capture(resid=stack("resid"), attn_out=stack("attn_out"),
                   mlp_out=stack("mlp_out"), mlp_act=stack("mlp_act"),
                   target_ids=torch.tensor(answer_ids, device=output_device(model)),
                   n_prefix_tokens=len(prefix_ids))


@torch.no_grad()
def logit_lens(model, cap: Capture, top_k: int = 1, alt_ids: torch.Tensor | None = None) -> dict:
    """Read the answer token out of every layer's residual stream.

    Applies the final norm and the unembedding to each layer's residual.
    Returns per layer (rows) and answer position (columns): target
    log-probability, target rank (0 = top-1) and the top-k token ids.
    `alt_ids` ([n] token ids) adds the same statistics for an alternative
    token at each position, e.g. the model's own answer when the target is
    the gold answer.
    """
    arch = resolve_arch(model)
    norm = arch.final_norm
    h = cap.resid.to(norm.weight.device, model.lm_head.weight.dtype)        # [L+1, n, d]
    logits = model.lm_head(norm(h).to(output_device(model))).float()       # [L+1, n, V]
    if arch.softcap:                                                        # as in the model's forward
        logits = torch.tanh(logits / arch.softcap) * arch.softcap
    logp = torch.log_softmax(logits, dim=-1)
    tgt = cap.target_ids.view(1, -1, 1).expand(logp.shape[0], -1, 1)
    target_logp = logp.gather(-1, tgt).squeeze(-1)
    target_logit = logits.gather(-1, tgt).squeeze(-1)
    rank = (logits > target_logit.unsqueeze(-1)).sum(-1)
    top = logits.topk(top_k, dim=-1)
    out = {
        "target_logp": target_logp.cpu().numpy(),          # [L+1, n]
        "target_rank": rank.cpu().numpy(),                 # [L+1, n]
        "top_ids": top.indices.cpu().numpy(),              # [L+1, n, k]
        "top_logp": logp.gather(-1, top.indices).cpu().numpy(),
    }
    if alt_ids is not None:
        alt = alt_ids.to(logits.device).view(1, -1, 1).expand(logp.shape[0], -1, 1)
        alt_logit = logits.gather(-1, alt).squeeze(-1)
        out["alt_logp"] = logp.gather(-1, alt).squeeze(-1).cpu().numpy()
        out["alt_rank"] = (logits > alt_logit.unsqueeze(-1)).sum(-1).cpu().numpy()
    return out


@torch.no_grad()
def direct_logit_attribution(model, cap: Capture) -> dict:
    """How much each sublayer writes toward the answer token.

    Projects attn_out[l] and mlp_out[l] onto the unembedding direction of the
    target token (minus the mean unembedding, so a uniform shift counts as
    zero), scaled by the final RMSNorm of the last residual. The sum over all
    sublayers plus the embedding approximates the final target logit
    relative to the mean logit.
    """
    arch = resolve_arch(model)
    norm = arch.final_norm
    W = model.lm_head.weight
    if not hasattr(model, "_mean_unembed"):                 # [1, d], computed once
        model._mean_unembed = W.float().mean(0, keepdim=True)
    direction = W[cap.target_ids].float() - model._mean_unembed  # [n, d]
    direction = direction * norm.weight.float().to(direction.device)
    final = cap.resid[-1]                                   # [n, d]
    eps = _attr(norm, "variance_epsilon", "eps") or 1e-6
    scale = torch.rsqrt(final.pow(2).mean(-1, keepdim=True) + eps)  # [n, 1]

    def dla(x, factor=None):  # x: [L, n, d]
        out = ((x * scale) * direction).sum(-1)             # [L, n]
        if factor is not None:                              # writes rescaled by later layer scalars
            out = out * torch.tensor(factor, device=out.device, dtype=out.dtype).view(-1, 1)
        return out.cpu().numpy()

    return {"attn": dla(cap.attn_out, arch.write_scale), "mlp": dla(cap.mlp_out, arch.write_scale),
            "embed": dla(cap.resid[:1])[0] * arch.embed_scale}


def convergence_layer(target_rank: np.ndarray) -> int | None:
    """First layer from which every answer token is top-1 through the last layer.

    target_rank: [L+1, n]. Returns None if the final layer itself does not
    rank the answer first (the model would not produce that answer).
    """
    top1 = (target_rank == 0).all(axis=1)                   # [L+1]
    if not top1[-1]:
        return None
    layer = len(top1) - 1
    while layer > 0 and top1[layer - 1]:
        layer -= 1
    return int(layer)


def compare(clean: Capture, other: Capture, top_frac: float = 0.01) -> dict:
    """Per-layer difference between two runs at the answer positions.

    Both captures must use the same answer tokens. Returns arrays indexed by
    layer, averaged over answer positions:
      resid_cos:    cosine similarity of the residual stream (L+1);
      resid_rel_l2: ||h_other - h_clean|| / ||h_clean|| (L+1);
      mlp_rel_l2:   same for MLP neuron activations (L);
      neuron_jaccard: overlap of the top `top_frac` most active neurons (L).
    """
    assert torch.equal(clean.target_ids, other.target_ids), "different answers"

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean(-1)

    def rel(a, b):
        return ((b - a).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-6)).mean(-1)

    k = max(1, int(top_frac * clean.mlp_act.shape[-1]))
    top_c = clean.mlp_act.abs().topk(k, dim=-1).indices     # [L, n, k]
    top_o = other.mlp_act.abs().topk(k, dim=-1).indices
    d_ff = clean.mlp_act.shape[-1]
    mask_c = torch.zeros(*top_c.shape[:2], d_ff, dtype=torch.bool, device=top_c.device)
    mask_o = torch.zeros_like(mask_c)
    mask_c.scatter_(-1, top_c, True)
    mask_o.scatter_(-1, top_o, True)
    jac = (mask_c & mask_o).sum(-1) / (mask_c | mask_o).sum(-1)

    return {
        "resid_cos": cos(clean.resid, other.resid).cpu().numpy(),
        "resid_rel_l2": rel(clean.resid, other.resid).cpu().numpy(),
        "mlp_rel_l2": rel(clean.mlp_act, other.mlp_act).cpu().numpy(),
        "neuron_jaccard": jac.float().mean(-1).cpu().numpy(),
    }
