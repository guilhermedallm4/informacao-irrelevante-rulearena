"""Atribuição por cabeça de atenção nas sondas (sem treinamento).

Para cada sonda: passagem única (teacher forcing) e, na última posição,
  - diferença de logit (correto - errado);
  - DLA por cabeça: projeção da escrita da cabeça (fatia de o_proj) na direção
    W_U[correto] - W_U[errado], com a escala da RMSNorm final;
  - massa de atenção de cada cabeça nos trechos do prompt (regras, exemplo,
    pergunta, resposta do modelo).
Saída: results/edicao_pesos/diagnostico.pt
"""
import json, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.airline.prompt_builder import ONE_SHOT_EXAMPLE

MODEL = "Qwen/Qwen3-14B"

def spans(tok, prefix, question_marker):
    enc = tok(prefix, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    def tok_at(char):
        for i, (a, b) in enumerate(offs):
            if b > char: return i
        return len(offs)
    i_rules = tok_at(prefix.index("# Bag fees"))
    i_ex = tok_at(prefix.index("<example>"))
    i_q = tok_at(prefix.index("</example>") + len("</example>"))
    i_resp = tok_at(prefix.index("<|im_start|>assistant"))
    return enc["input_ids"], [0, i_rules, i_ex, i_q, i_resp, len(offs)]

def main(probes_path, out_path, limit=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0",
                                                 attn_implementation="eager").eval()
    cfg = model.config; H = cfg.num_attention_heads; D = cfg.head_dim; L = cfg.num_hidden_layers
    layers = model.model.layers; norm = model.model.norm; WU = model.lm_head.weight
    store = {}
    def pre(i):
        def hook(mod, inp): store[("z", i)] = inp[0][0, -1].detach().float()
        return hook
    def post(i):
        def hook(mod, inp, out): store[("a", i)] = out[1][0, :, -1, :].detach().float()
        return hook
    for i, l in enumerate(layers):
        l.self_attn.o_proj.register_forward_pre_hook(pre(i))
        l.self_attn.register_forward_hook(post(i))
    norm.register_forward_pre_hook(lambda mod, inp: store.__setitem__("h", inp[0][0, -1].detach().float()))
    probes = [json.loads(l) for l in open(probes_path)]
    if limit: probes = probes[:limit]
    recs = []
    with torch.no_grad():
        for n, p in enumerate(probes):
            ids, sp = spans(tok, p["prefix"], None)
            x = torch.tensor([ids], device="cuda:0")
            out = model(input_ids=x, use_cache=False)
            logits = out.logits[0, -1].float()
            g = tok(p["gold"], add_special_tokens=False).input_ids[0]
            w = tok(p["wrong"], add_special_tokens=False).input_ids[0]
            h = store["h"]
            scale = torch.rsqrt(h.pow(2).mean() + norm.variance_epsilon)
            d = (WU[g].float() - WU[w].float()) * norm.weight.float() * scale
            dla = torch.zeros(L, H)
            att = torch.zeros(L, H, 5)
            for i, l in enumerate(layers):
                z = store[("z", i)].view(H, D)                       # [H, D]
                Wo = l.self_attn.o_proj.weight.float().view(-1, H, D)  # [d_model, H, D]
                head_out = torch.einsum("mhk,hk->hm", Wo, z)          # [H, d_model]
                dla[i] = head_out @ d
                a = store[("a", i)]
                for s in range(5):
                    att[i, :, s] = a[:, sp[s]:sp[s+1]].sum(-1)
            recs.append(dict(kind=p["kind"], gold=p["gold"], instance=p["instance"], item=p["item"], info=p["info"],
                             logit_diff=(logits[g] - logits[w]).item(), top1=tok.decode([logits.argmax().item()]),
                             dla=dla.cpu(), att=att.cpu(), n_tokens=len(ids)))
            if n % 50 == 0: print(n, p["kind"], p["gold"], f"diff={recs[-1]['logit_diff']:.2f} top1={recs[-1]['top1']!r}", flush=True)
    torch.save(recs, out_path)
    print("salvo", out_path, len(recs))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
