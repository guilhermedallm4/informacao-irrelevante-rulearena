"""Rodada 6 (edição em up_proj): dados no fluxo residual na ENTRADA da MLP (pós-norma, 5 120 dim).

Modo "tracos": passa os 90 traços-gabarito; em cada marca guarda x (entrada da MLP) nas CAMADAS,
  top-5 e margem do modelo; acumula G_x = Σ x xᵀ (5 120²) sobre todas as posições.
Modo "livre": passa respostas livres (jsonl) e só acumula G_x (alvo zero), com truncamento.
Uso: coletar_residual.py tracos <saida.pt> [config|-] [gpu] [reordenar=0|1]
     coletar_residual.py livre  <saida.pt> <a.jsonl,b.jsonl> [gpu] [max_tokens_resposta]
"""
import sys, json, torch, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.airline.gold import load_airline_problems
from src.interp.experiment import sample_problems, chat_prompt
from edicao_pesos.traco_gabarito import traco
from edicao_pesos.edicoes import aplicar
MODEL = "Qwen/Qwen3-14B"; CAMADAS = [20, 22, 23, 24, 26, 28, 30, 31, 32, 34]

def carregar(gpu, cfg_path=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=f"cuda:{gpu}", attn_implementation="sdpa").eval()
    if cfg_path:
        for l in aplicar(model, tok, json.load(open(cfg_path))): print("  edição:", l)
    store = {}
    for L in CAMADAS:
        model.model.layers[L].mlp.register_forward_pre_hook(lambda m, inp, L=L: store.__setitem__(L, inp[0][0].detach()))
    return tok, model, store

def tracos(out_path, cfg_path, gpu, reordenar=False):
    tok, model, store = carregar(gpu, cfg_path); dev = f"cuda:{gpu}"; dm = model.config.hidden_size
    G = {L: torch.zeros(dm, dm, dtype=torch.float32, device=dev) for L in CAMADAS}
    probs = {c: load_airline_problems(c) for c in (0, 1, 2)}; sample = sample_problems(0.3, 42)
    acts, meta, n_pos = [], [], 0
    with torch.no_grad():
        for s in sample:
            p = probs[s["complexity"]][s["problem_idx"]]; texto, marcas, total = traco(p, reordenar)
            prefix = chat_prompt(tok, s["question"], "one_shot"); full = prefix + texto
            enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
            ids = torch.tensor([enc["input_ids"]], device=dev); starts = [a for a, b in enc["offset_mapping"]]
            logits = model(input_ids=ids, use_cache=False).logits[0].float(); n_pos += ids.shape[1]
            for L in CAMADAS:
                x = store[L].float(); G[L] += x.T @ x
            for ch, nome, texto_ok in marcas:
                ch_abs = len(prefix) + ch
                pos_tok = next(i for i, a in enumerate(starts) if a >= ch_abs); assert starts[pos_tok] == ch_abs
                p_ = pos_tok - 1; gold = enc["input_ids"][pos_tok]; lg = logits[p_]; top = lg.topk(5)
                outros = lg.clone(); outros[gold] = -1e9
                acts.append(torch.stack([store[L][p_] for L in CAMADAS]).half().cpu())
                meta.append(dict(instance=s["instance_id"], nome=nome, gold=gold, gold_txt=tok.decode([gold]), top=top.indices.tolist(),
                                 top_txt=[tok.decode([t]) for t in top.indices.tolist()], margem=(lg[gold] - outros.max()).item(), traco_ok=(total == s["gold"])))
            print(s["instance_id"], ids.shape[1], len(acts), flush=True)
    torch.save(dict(camadas=CAMADAS, G={L: G[L].cpu() for L in CAMADAS}, n_pos=n_pos, acts=torch.stack(acts), meta=meta), out_path)
    c = collections.defaultdict(lambda: [0, 0])
    for m in meta: k = (m["nome"], m["gold_txt"]); c[k][1] += 1; c[k][0] += (m["top"][0] == m["gold"])
    for k in sorted(c): print(f"{k[0]:12s} {k[1]!r:6s} acerto top-1 {c[k][0]:3d}/{c[k][1]:3d}")
    print("salvo", out_path, n_pos, "posições")

def livre(out_path, arquivos, gpu, max_resp):
    tok, model, store = carregar(gpu); dev = f"cuda:{gpu}"; dm = model.config.hidden_size
    G = {L: torch.zeros(dm, dm, dtype=torch.float32, device=dev) for L in CAMADAS}; n_pos = 0; n_resp = 0
    with torch.no_grad():
        for arq in arquivos:
            for r in map(json.loads, open(arq)):
                ids_p = tok(chat_prompt(tok, r["question"], "one_shot"), add_special_tokens=False).input_ids
                ids_r = tok(r["response"], add_special_tokens=False).input_ids[:max_resp]
                ids = torch.tensor([ids_p + ids_r], device=dev); model(input_ids=ids, use_cache=False)
                for L in CAMADAS:
                    x = store[L].float(); G[L] += x.T @ x
                n_pos += ids.shape[1]; n_resp += 1; print(Path(arq).stem, r["instance_id"], ids.shape[1], flush=True)
    torch.save(dict(camadas=CAMADAS, G={L: G[L].cpu() for L in CAMADAS}, n_pos=n_pos, n_resp=n_resp, arquivos=arquivos), out_path)
    print("salvo", out_path, n_pos, "posições de", n_resp, "respostas")

if __name__ == "__main__":
    a = sys.argv
    if a[1] == "tracos": tracos(a[2], None if len(a) < 4 or a[3] == "-" else a[3], int(a[4]) if len(a) > 4 else 0, bool(int(a[5])) if len(a) > 5 else False)
    else: livre(a[2], a[3].split(","), int(a[4]) if len(a) > 4 else 0, int(a[5]) if len(a) > 5 else 3000)
