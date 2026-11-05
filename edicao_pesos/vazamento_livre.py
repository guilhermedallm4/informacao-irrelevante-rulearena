"""Distribuição da resposta r_j = w_j·k de cada chave em TODAS as posições das respostas livres
(modelo-base), com os contextos de maior resposta. Serve para limitar α_j à mão.
Uso: vazamento_livre.py <sufixo> <saida.pt> <a.jsonl,b.jsonl> [gpu] [max_tokens_resposta]"""
import sys, json, torch, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.interp.experiment import chat_prompt
MODEL = "Qwen/Qwen3-14B"; D = "results/edicao_pesos/deltas/"

def main(sufixo, out_path, arquivos, gpu=0, max_resp=3000):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = f"cuda:{gpu}"; tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa").eval()
    cls = json.load(open(D + f"classes_{sufixo}.json")); camadas = sorted(set(c["camada"] for c in cls))
    V = {L: torch.load(D + f"fatores_{sufixo}_L{L}.pt")["V"].to(dev) for L in camadas}
    nomes = {L: torch.load(D + f"fatores_{sufixo}_L{L}.pt")["classes"] for L in camadas}
    store = {}
    for L in camadas:
        model.model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, inp, L=L: store.__setitem__(L, inp[0][0].detach()))
    resp = {L: [] for L in camadas}; top = {L: [[] for _ in nomes[L]] for L in camadas}
    with torch.no_grad():
        for arq in arquivos:
            for r in map(json.loads, open(arq)):
                prefix = chat_prompt(tok, r["question"], "one_shot"); np_ = len(tok(prefix, add_special_tokens=False).input_ids)
                ids_r = tok(r["response"], add_special_tokens=False).input_ids[:max_resp]
                ids = torch.tensor([tok(prefix, add_special_tokens=False).input_ids + ids_r], device=dev)
                model(input_ids=ids, use_cache=False)
                for L in camadas:
                    R = (store[L].float() @ V[L])[np_:]          # só a resposta gerada [T_r, nc]
                    resp[L].append(R.half().cpu())
                    for j in range(R.shape[1]):
                        v, i = R[:, j].topk(min(3, R.shape[0]))
                        for vv, ii in zip(v.tolist(), i.tolist()):
                            top[L][j].append((vv, r["instance_id"], tok.decode(ids_r[max(0, ii-12):ii+1]), tok.decode(ids_r[ii+1:ii+2])))
                print(Path(arq).stem, r["instance_id"], flush=True)
    out = {}
    for L in camadas:
        A = torch.cat(resp[L]).float()
        for j, nome in enumerate(nomes[L]):
            x = A[:, j]; q = torch.quantile(x, torch.tensor([0.5, 0.99, 0.999, 1.0]))
            tj = sorted(top[L][j], key=lambda t: -t[0])[:8]
            out[nome] = dict(camada=L, q50=q[0].item(), q99=q[1].item(), q999=q[2].item(), max=q[3].item(), n=len(x), top=tj)
            print(f"L{L} {nome:22s} mediana {q[0]:+.3f} p99 {q[1]:+.3f} p99.9 {q[2]:+.3f} máx {q[3]:+.3f}")
            for t in tj[:4]: print(f"      {t[0]:+.2f} {t[1]} ...{t[2]!r} → {t[3]!r}")
    torch.save(out, out_path); print("salvo", out_path)

if __name__ == "__main__":
    a = sys.argv; main(a[1], a[2], a[3].split(","), int(a[4]) if len(a) > 4 else 0, int(a[5]) if len(a) > 5 else 3000)
