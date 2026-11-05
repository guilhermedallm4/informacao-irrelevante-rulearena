"""Ativações das MLPs (entrada de down_proj) e DLA por camada da MLP nas sondas.

Guarda, para as camadas 20-39, o vetor de neurônios na posição de decisão de
cada sonda (fp16) e o DLA da MLP de cada camada na direção correto-errado.
Também guarda ativações em posições aleatórias de 12 respostas, para medir
efeitos colaterais fora dos pontos de decisão.
"""
import json, sys, torch, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
MODEL = "Qwen/Qwen3-14B"
CAMADAS = list(range(20, 40))

def main(probes_path, out_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0",
                                                 attn_implementation="sdpa").eval()
    L = model.config.num_hidden_layers; norm = model.model.norm; WU = model.lm_head.weight
    store = {}
    for i, l in enumerate(model.model.layers):
        l.mlp.down_proj.register_forward_pre_hook(lambda m, inp, i=i: store.__setitem__(("k", i), inp[0][0].detach()))
        l.mlp.register_forward_hook(lambda m, inp, out, i=i: store.__setitem__(("mlp", i), out[0, -1].detach().float()))
    norm.register_forward_pre_hook(lambda m, inp: store.__setitem__("h", inp[0][0, -1].detach().float()))
    probes = [json.loads(l) for l in open(probes_path)]
    acts, dla, meta = [], [], []
    rnd_acts = []
    rng = random.Random(0)
    with torch.no_grad():
        for n, p in enumerate(probes):
            ids = tok(p["prefix"], add_special_tokens=False, return_tensors="pt").input_ids.to("cuda:0")
            logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
            g = tok(p["gold"], add_special_tokens=False).input_ids[0]; w = tok(p["wrong"], add_special_tokens=False).input_ids[0]
            scale = torch.rsqrt(store["h"].pow(2).mean() + norm.variance_epsilon)
            d = (WU[g].float() - WU[w].float()) * norm.weight.float() * scale
            dla.append(torch.stack([store[("mlp", i)] @ d for i in range(L)]).cpu())
            acts.append(torch.stack([store[("k", i)][-1] for i in CAMADAS]).half().cpu())
            meta.append(dict(kind=p["kind"], gold=p["gold"], instance=p["instance"], item=p["item"],
                             diff=(logits[g] - logits[w]).item(), top1=tok.decode([logits.argmax().item()])))
            if n % 75 == 0:   # posições aleatórias da resposta (últimos 400 tokens), para efeitos colaterais
                T = ids.shape[1]; pos = rng.sample(range(max(0, T - 400), T - 1), 40)
                rnd_acts.append(torch.stack([store[("k", i)][pos] for i in CAMADAS]).half().cpu())  # [nc, 40, dff]
                print(n, meta[-1], flush=True)
    torch.save(dict(camadas=CAMADAS, acts=torch.stack(acts), dla_mlp=torch.stack(dla), meta=meta,
                    rnd_acts=torch.cat(rnd_acts, dim=1)), out_path)
    print("salvo", out_path)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
