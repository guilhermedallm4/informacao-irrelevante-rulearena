"""Diagnóstico de cópia do exemplo 1-shot (sem treinamento).

Passa cada traço-gabarito (ordem do enunciado) com atenção eager e, em cada marca
(ponto de decisão), guarda por cabeça (40 camadas × 40 cabeças):
  - massa de atenção em cada trecho do contexto: antes das regras, regras, exemplo,
    pergunta, resposta já escrita;
  - DLA: projeção da escrita da cabeça (fatia de o_proj) em
    d̂ = (W_U[gold] − W_U[2º melhor]) ⊙ γ · escala da RMSNorm final na posição.
Saída: <saida.pt> com meta por marca, att [n, L, H, 5] e dla [n, L, H] (fp16).
Uso: diag_copia.py <saida.pt> [gpu] [config|-] [reordenar=0|1]
"""
import sys, json, torch, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.airline.gold import load_airline_problems
from src.interp.experiment import sample_problems, chat_prompt
from edicao_pesos.traco_gabarito import traco
from edicao_pesos.edicoes import aplicar
MODEL = "Qwen/Qwen3-14B"
MARCADORES = ["# Bag fees", "<example>", "</example>", "<|im_start|>assistant"]

def main(out_path, gpu=0, cfg_path=None, reordenar=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = f"cuda:{gpu}"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev, attn_implementation="eager").eval()
    if cfg_path:
        for l in aplicar(model, tok, json.load(open(cfg_path))): print("  edição:", l)
    cfg = model.config; H = cfg.num_attention_heads; Dh = cfg.head_dim; NL = cfg.num_hidden_layers
    layers = model.model.layers; norm = model.model.norm; WU = model.lm_head.weight.float()
    P = []; sp = []; st = {}
    def pre(i):
        def hook(mod, inp): st[("z", i)] = inp[0][0, P].detach().float()
        return hook
    def post(i):
        def hook(mod, inp, out):
            a = out[1][0][:, P, :].float()                       # [H, |P|, T]
            st[("a", i)] = torch.stack([a[:, :, sp[s]:sp[s + 1]].sum(-1) for s in range(5)], -1).cpu()   # [H, |P|, 5]
        return hook
    for i, l in enumerate(layers):
        l.self_attn.o_proj.register_forward_pre_hook(pre(i)); l.self_attn.register_forward_hook(post(i))
    norm.register_forward_pre_hook(lambda mod, inp: st.__setitem__("h", inp[0][0, P].detach().float()))
    Wo = [l.self_attn.o_proj.weight.float().view(-1, H, Dh) for l in layers]       # [d_model, H, Dh]
    probs = {c: load_airline_problems(c) for c in (0, 1, 2)}; sample = sample_problems(0.3, 42)
    meta, ATT, DLA = [], [], []
    with torch.no_grad():
        for s in sample:
            p = probs[s["complexity"]][s["problem_idx"]]; texto, marcas, total = traco(p, reordenar)
            prefix = chat_prompt(tok, s["question"], "one_shot"); full = prefix + texto
            enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
            starts = [a for a, b in enc["offset_mapping"]]; T = len(starts)
            def tok_at(ch): return next(i for i, a in enumerate(starts) if a >= ch)
            sp[:] = [0] + [tok_at(prefix.index(m) + (len(m) if m == "</example>" else 0)) for m in MARCADORES] + [T]
            P[:] = []; golds = []
            for ch, nome, texto_ok in marcas:
                pos_tok = tok_at(len(prefix) + ch); assert starts[pos_tok] == len(prefix) + ch
                P.append(pos_tok - 1); golds.append(enc["input_ids"][pos_tok])
            ids = torch.tensor([enc["input_ids"]], device=dev)
            logits = model(input_ids=ids, use_cache=False).logits[0, P].float()          # [|P|, V]
            h = st["h"]; scale = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)   # [|P|, 1]
            g = torch.tensor(golds, device=dev); outros = logits.clone(); outros[torch.arange(len(P)), g] = -1e9
            w = outros.argmax(-1); margem = logits[torch.arange(len(P)), g] - outros.max(-1).values
            d = (WU[g] - WU[w]) * norm.weight.float() * scale                             # [|P|, d_model]
            dla = torch.zeros(len(P), NL, H)
            for i in range(NL):
                z = st[("z", i)].view(len(P), H, Dh)
                head_out = torch.einsum("mhk,phk->phm", Wo[i], z)                        # [|P|, H, d_model]
                dla[:, i] = (head_out * d[:, None, :]).sum(-1).cpu()
            att = torch.stack([st[("a", i)] for i in range(NL)], 0).permute(2, 0, 1, 3)  # [L,H,|P|,5] -> [|P|, L, H, 5]
            ATT.append(att.half()); DLA.append(dla.half())
            for k, (ch, nome, texto_ok) in enumerate(marcas):
                meta.append(dict(instance=s["instance_id"], nome=nome, gold=golds[k], gold_txt=tok.decode([golds[k]]),
                                 wrong=w[k].item(), wrong_txt=tok.decode([w[k].item()]), margem=margem[k].item(), top1_ok=bool(margem[k] > 0)))
            print(s["instance_id"], T, len(meta), f"erros={int((margem < 0).sum())}", flush=True)
            del logits, outros; st.clear(); torch.cuda.empty_cache()
    torch.save(dict(meta=meta, att=torch.cat(ATT), dla=torch.cat(DLA), spans=["pre", "regras", "exemplo", "pergunta", "resposta"]), out_path)
    c = collections.defaultdict(lambda: [0, 0])
    for m in meta: k = (m["nome"], m["gold_txt"]); c[k][1] += 1; c[k][0] += m["top1_ok"]
    for k in sorted(c): print(f"{k[0]:12s} {k[1]!r:6s} acerto top-1 {c[k][0]:4d}/{c[k][1]:4d}")
    print("salvo", out_path, len(meta), "marcas")

if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 0, None if len(a) < 4 or a[3] == "-" else a[3], bool(int(a[4])) if len(a) > 4 else False)
