"""Empurrão efetivo de cada chave up_proj por posição nas respostas livres:
push_j(pos) = (v_j·x)(g·c_j), c_j = u_j ⊙ W_downᵀ d̂_j, relativo a 1 na classe.
Mostra quantis, contextos de maior empurrão e, nas respostas cortadas (laço), o empurrão
médio no último trecho por classe (atribuição do laço).
Uso: vazamento_up.py <sufixo> <saida.pt> <a.jsonl,...> [gpu] [max_tokens_resposta]"""
import sys, json, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.interp.experiment import chat_prompt
from edicao_pesos.edicoes import aplicar
MODEL = "Qwen/Qwen3-14B"; D = "results/edicao_pesos/deltas/"

def main(sufixo, out_path, arquivos, gpu=0, max_resp=3000):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    dev = f"cuda:{gpu}"; tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa").eval()
    cls = json.load(open(D + f"classes_{sufixo}.json")); camadas = sorted(set(c["camada"] for c in cls))
    F = {L: torch.load(D + f"fatores_{sufixo}_L{L}.pt") for L in camadas}
    WU = model.lm_head.weight.float(); gamma = model.model.norm.weight.float()
    C = {}
    for L in camadas:
        Wd = model.model.layers[L].mlp.down_proj.weight.float(); cs = []
        for j, nome in enumerate(F[L]["classes"]):
            c = next(c for c in cls if f"{c['nome']}:{c['gold_txt']!r}" == nome)
            ids = [tok(t, add_special_tokens=False).input_ids[0] for t in (c["gold_txt"], c["wrong_txt"])]
            d = (WU[ids[0]] - WU[ids[1]]) * gamma; d = d / d.norm()
            cs.append(F[L]["U"][:, j].to(dev) * (Wd.T @ d))
        C[L] = torch.stack(cs, 1); F[L]["Vd"] = F[L]["V"].to(dev)
    store = {}
    for L in camadas:
        model.model.layers[L].mlp.register_forward_pre_hook(lambda m, inp, L=L: store.__setitem__(L, inp[0][0].detach()))
    resp = {L: [] for L in camadas}; top = {L: [[] for _ in F[L]["classes"]] for L in camadas}; laco = {L: [[] for _ in F[L]["classes"]] for L in camadas}
    with torch.no_grad():
        for arq in arquivos:
            for r in map(json.loads, open(arq)):
                ids_p = tok(chat_prompt(tok, r["question"], "one_shot"), add_special_tokens=False).input_ids; np_ = len(ids_p)
                ids_r = tok(r["response"], add_special_tokens=False).input_ids[:max_resp]
                ids = torch.tensor([ids_p + ids_r], device=dev); model(input_ids=ids, use_cache=False)
                for L in camadas:
                    x = store[L].float(); g = torch.nn.functional.silu(model.model.layers[L].mlp.gate_proj(store[L]).float())
                    P = ((x @ F[L]["Vd"]) * (g @ C[L]))[np_:]
                    resp[L].append(P.half().cpu())
                    for j in range(P.shape[1]):
                        v, i = P[:, j].topk(min(2, P.shape[0]))
                        for vv, ii in zip(v.tolist(), i.tolist()): top[L][j].append((vv, r["instance_id"], tok.decode(ids_r[max(0, ii-10):ii+1]), tok.decode(ids_r[ii+1:ii+2])))
                        if r.get("finish_reason") == "length": laco[L][j].append((P[-800:, j].abs().mean().item(), r["instance_id"], Path(arq).stem))
                print(Path(arq).stem, r["instance_id"], flush=True)
    out = {}
    for L in camadas:
        A = torch.cat(resp[L]).float()
        for j, nome in enumerate(F[L]["classes"]):
            x = A[:, j]; q = torch.quantile(x, torch.tensor([0.5, 0.99, 0.999, 1.0])); qn = torch.quantile(x, torch.tensor([0.001, 0.0]))
            tj = sorted(top[L][j], key=lambda t: -t[0])[:6]; lj = sorted(laco[L][j], key=lambda t: -t[0])
            out[nome] = dict(camada=L, q50=q[0].item(), q99=q[1].item(), q999=q[2].item(), max=q[3].item(), min=qn[1].item(), n=len(x), top=tj,
                             laco_medio=(sum(t[0] for t in lj) / len(lj) if lj else 0.0), laco=lj[:6])
            print(f"L{L} {nome:22s} p99 {q[1]:+.3f} p99.9 {q[2]:+.3f} máx {q[3]:+.3f} mín {qn[1]:+.3f} | laço: empurrão médio final {out[nome]['laco_medio']:.3f} (n={len(lj)})")
            for t in tj[:3]: print(f"      {t[0]:+.2f} {t[1]} ...{t[2]!r} → {t[3]!r}")
    torch.save(out, out_path); print("salvo", out_path)

if __name__ == "__main__":
    a = sys.argv; main(a[1], a[2], a[3].split(","), int(a[4]) if len(a) > 4 else 0, int(a[5]) if len(a) > 5 else 3000)
