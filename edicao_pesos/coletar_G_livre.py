"""Rodada 4: G_L = KᵀK sobre as posições das respostas LIVRES do modelo (prompt + resposta).

Os traços-gabarito só cobrem o texto ideal; as chaves w_j ajustadas só neles vazaram
para frases que o modelo escreve sozinho ("within the Main Cabin allowance", "1st bag").
Aqui acumula-se G (alvo zero) sobre respostas reais: linha de base, v4 e as respostas
colapsadas da rodada 3 (que contêm exatamente os contextos de vazamento).
Uso: coletar_G_livre.py <saida.pt> <a.jsonl,b.jsonl,...> [gpu] [max_tokens_resposta]
"""
import sys, json, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.interp.experiment import chat_prompt
MODEL = "Qwen/Qwen3-14B"; CAMADAS = [20, 22, 24, 26, 28, 30, 32, 34]

def main(out_path, arquivos, gpu=0, max_resp=3000):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = f"cuda:{gpu}"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa").eval()
    dff = model.config.intermediate_size
    G = {L: torch.zeros(dff, dff, dtype=torch.float32, device=dev) for L in CAMADAS}
    store = {}
    for L in CAMADAS:
        model.model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, inp, L=L: store.__setitem__(L, inp[0][0].detach()))
    n_pos = 0; n_resp = 0
    with torch.no_grad():
        for arq in arquivos:
            for r in map(json.loads, open(arq)):
                prefix = chat_prompt(tok, r["question"], "one_shot")
                ids_p = tok(prefix, add_special_tokens=False).input_ids
                ids_r = tok(r["response"], add_special_tokens=False).input_ids[:max_resp]
                ids = torch.tensor([ids_p + ids_r], device=dev)
                model(input_ids=ids, use_cache=False)
                for L in CAMADAS:
                    x = store[L].float(); G[L] += x.T @ x
                n_pos += ids.shape[1]; n_resp += 1
                print(Path(arq).stem, r["instance_id"], ids.shape[1], flush=True)
    torch.save(dict(camadas=CAMADAS, G={L: G[L].cpu() for L in CAMADAS}, n_pos=n_pos, n_resp=n_resp, arquivos=arquivos), out_path)
    print("salvo", out_path, n_pos, "posições de", n_resp, "respostas")

if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2].split(","), int(a[3]) if len(a) > 3 else 0, int(a[4]) if len(a) > 4 else 3000)
