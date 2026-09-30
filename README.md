# Informação irrelevante e seguimento de regras em LLMs (RuleArena airline)

Frases que não têm nada a ver com o problema ("I was counting 1+1.", "I stopped
to breathe.", "What is the value?") mudam a resposta de LLMs que calculam a
tarifa de bagagem da American Airlines? E o que acontece dentro do modelo
quando isso ocorre?

Este repositório tem o código, os testes e os resultados de um estudo com
**15 modelos abertos, de 1,2B a 72B parâmetros**: Qwen2.5, Qwen3, Gemma 4,
OLMo 3, LFM (Liquid AI), phi-4 e Mistral. Cada modelo foi avaliado em 90
problemas do [RuleArena](https://github.com/skyriver-2000/RuleArena) (Zhou et
al., ACL 2025), com medidas de comportamento e de ativações internas.

## Principais achados

1. **Só um modelo domina a tarefa.** O Gemma 4 31B acerta **59%** dos
   problemas. Os outros 14 ficam entre 0% e 9%, inclusive o Qwen2.5-72B (7%).
   O que a escala melhora nesses modelos é a *aproximação*: a fração de
   respostas a até ±10% do gabarito vai de 6% (LFM2.5-1.2B) a 70%
   (Qwen2.5-72B). Os erros são de **aplicação de regras**, como cobrar a taxa
   de tamanho em vez da de peso. As contas que os modelos escrevem fecham com
   o total que eles declaram.

2. **Frases irrelevantes mudam a resposta de quase todos os modelos, e mais
   do que o ruído numérico.** Com decodificação gulosa, só gerar a mesma
   pergunta em outro lote já muda de 6% a 93% das respostas, conforme o modelo.
   A frase no prompt aumenta esse desvio em todos os modelos, menos no Gemma 4.
   Nos modelos que quase nunca acertam, a frase só **troca um erro por outro**:
   o erro mediano com e sem frase é praticamente o mesmo.

3. **Nos modelos intermediários, as frases desfazem acertos.** Somando os
   modelos que acertam alguns problemas (Gemma 4 à parte), as frases desfazem
   **36% dos acertos (84 de 234), contra 15% no controle (12 de 78), com
   p = 0,0006 (Fisher)**. No Gemma 4 a taxa é a mesma com e sem frase (3%,
   p = 1,0).

4. **Estabilidade e robustez às frases são propriedades diferentes.** O
   Qwen2.5-72B é tão estável ao ruído de lote quanto o Gemma 4 (controle de 9%
   contra 8%), mas as frases ainda triplicam o desvio dele (26–34%,
   p ≤ 0,005). No Gemma 4 elas não têm efeito. A escala do Qwen2.5 compra
   estabilidade e aproximação, mas não imunidade às frases nem a conta exata.

5. **Frases no meio do raciocínio separam as famílias.** Qwen3, Mistral e
   Gemma 4 resistem a frases sem números inseridas no meio de uma conta.
   Qwen2.5, phi-4, LFM e OLMo se abalam com qualquer frase. "I was counting
   1+1." é sempre a mais perturbadora: no Qwen3-8B ela provocou 27 loops em
   que o modelo seguiu somando literalmente (1+1, 2+2, 4+4…) até o limite de
   tokens.

6. **A resposta só se forma nas últimas camadas, em todas as arquiteturas.**
   Pelo logit lens, a resposta vira top-1 e não muda mais a partir de 88–100%
   da profundidade: em modelos pré-norma (Qwen, Mistral, phi-4), pós-norma
   (OLMo, Gemma) e no híbrido de convolução e atenção (LFM).

7. **A camada em que o modelo escolhe o dígito errado fica mais profunda com
   a escala, mas depende da família.** No Qwen3 ela sobe de forma consistente
   (67% → 78% → 83% da profundidade, de 8B a 32B). No Qwen2.5 fica estável do
   7B ao 32B (76–81%) e só sobe no 72B (90%). O Gemma 4, o melhor modelo,
   decide em 73%: decidir mais tarde não é o que faz um modelo acertar.

8. **O deslocamento interno causado pela frase não prevê quais respostas
   mudam.** Ele depende sobretudo da **distância** entre o ponto em que a
   frase é inserida e a resposta. Com essa distância controlada, respostas
   que mudam e que não mudam têm o mesmo deslocamento. A resposta muda mais
   quanto mais texto ainda falta gerar, o que é coerente com uma perturbação
   pequena que se amplifica ao longo da geração.

9. **Os resultados se replicam entre máquinas e reproduzem o artigo
   original.** O Qwen3-8B dá os mesmos números agregados numa RTX 5090 e numa
   RTX PRO 6000 (controle de 40–41%, frases no prompt de 64–76%), embora
   respostas individuais mudem. O Qwen2.5-72B dá valores próximos aos do
   artigo do RuleArena (17% contra 19% no nível 1, 1-shot). Nas mesmas
   condições, o Gemma 4 (59%) supera os valores que o artigo reporta para o
   Claude-3.5 Sonnet (23%) e o GPT-4o (18%) e fica no nível do o1-preview
   (55%). Como o Gemma 4 é posterior ao benchmark, **não descartamos
   contaminação** (ver Limitações).

## Desenho do estudo

**Amostra.** 30% estratificado do RuleArena airline: 30 problemas de cada
nível de complexidade (0, 1, 2), sorteados com seed 42. O prompt é 1-shot:
a política de bagagem completa (~5–6 mil tokens), um exemplo resolvido e a
pergunta.

**Dois cenários de inserção:**

| Cenário | Onde entra a frase |
|---|---|
| *prompt* | no meio da lista de itens do passageiro |
| *reasoning* | no meio de uma soma do próprio raciocínio do modelo (ex.: `$0 (checking) + ` ▸ frase), e o modelo continua dali |

**Controles.** O controle *prompt* gera a mesma pergunta de novo, noutro lote,
e mede o ruído numérico da geração em lotes. O controle *reasoning* faz o
mesmo corte no raciocínio, sem frase. Todo efeito é comparado com o controle
do mesmo cenário (teste exato de Fisher).

**Ativações.** O raciocínio limpo é reapresentado em cada condição (*teacher
forcing*), e as ativações são lidas nas posições que produzem os dígitos da
resposta: fluxo residual, logit lens, atribuição direta ao logit (DLA) por
atenção e MLP, e neurônios da MLP. Como todas as condições leem o mesmo
texto, qualquer diferença vem da frase. Uma condição com padding mascarado
(`clean_padded`) mede o piso de ruído numérico. A **camada da escolha** é a
camada a partir da qual o dígito errado que o modelo escolheu passa o dígito
correto e não perde mais a frente.

**Decodificação.** Gulosa, com no máximo 4 096 tokens gerados (32 768 nos
VibeThinker). Os ajustes do `generation_config` de cada modelo que mudam a
escolha gulosa (por exemplo, `repetition_penalty=1.05` no Qwen2.5) são
zerados.

## Resultados

Tabelas completas em [`results/irrelevant_info/resumo/`](results/irrelevant_info/resumo/),
geradas por `python scripts/resumir_resultados.py`.

### Acurácia (pergunta sem frase, 90 problemas)

| Modelo | Acertos (90) | C0 / C1 / C2 | a ±10% | Erro mediano | Sem resposta |
|---|---|---|---|---|---|
| Qwen3-8B (RTX 5090, referência) | 2 (2.2%) | 2 / 0 / 0 | 40% | 14% | 0 |
| LFM2.5-1.2B-Instruct | 0 (0.0%) | 0 / 0 / 0 | 6% | 53% | 20 |
| LFM2-2.6B | 0 (0.0%) | 0 / 0 / 0 | 9% | 36% | 2 |
| Qwen2.5-7B-Instruct | 0 (0.0%) | 0 / 0 / 0 | 16% | 25% | 0 |
| Qwen3-8B (rtxpro6000) | 3 (3.3%) | 3 / 0 / 0 | 44% | 13% | 0 |
| Olmo-3-7B-Instruct | 0 (0.0%) | 0 / 0 / 0 | 14% | 29% | 15 |
| Qwen3-14B | 2 (2.2%) | 2 / 0 / 0 | 38% | 13% | 0 |
| Qwen2.5-14B-Instruct | 2 (2.2%) | 1 / 1 / 0 | 42% | 14% | 0 |
| phi-4 | 3 (3.3%) | 2 / 1 / 0 | 51% | 10% | 0 |
| Mistral-Small-24B-Instruct-2501 | 4 (4.4%) | 0 / 3 / 1 | 41% | 12% | 0 |
| gemma-4-31B-it | 53 (58.9%) | 19 / 18 / 16 | 98% | 0% | 0 |
| Qwen3-32B | 5 (5.6%) | 4 / 1 / 0 | 53% | 9% | 0 |
| Qwen2.5-32B-Instruct | 8 (8.9%) | 4 / 3 / 1 | 53% | 9% | 0 |
| Olmo-3.1-32B-Instruct | 6 (6.7%) | 3 / 2 / 1 | 34% | 9% | 32 |
| Qwen2.5-72B-Instruct | 6 (6.7%) | 5 / 1 / 0 | 70% | 7% | 0 |

"Sem resposta": a geração limpa não chegou a "The total cost is $X". No OLMo,
isso é raciocínio legítimo cortado pelo limite de 4 096 tokens, então a
acurácia dele está subestimada.

### Comportamento (% de respostas que mudaram em relação à pergunta sem frase)

| Modelo | Controle (prompt) | Frases no prompt | p máx. | Controle (raciocínio) | "1+1" no raciocínio | Outras 2 no raciocínio | p (outras) |
|---|---|---|---|---|---|---|---|
| Qwen3-8B (RTX 5090, referência) | 40% | 64–67% | 0.002 | 20% | 53% (25 loops) | 23–24% | 0.72/0.59 |
| LFM2.5-1.2B-Instruct | 86% | 97–100% | 0.031 | 49% | 91% (46 loops) | 91–96% | 0.00/0.00 |
| LFM2-2.6B | 72% | 84–92% | 0.069 | 22% | 52% (4 loops) | 41–58% | 0.00/0.01 |
| Qwen2.5-7B-Instruct | 39% | 72–83% | 0.000 | 21% | 46% (0 loops) | 36–46% | 0.05/0.00 |
| Qwen3-8B (rtxpro6000) | 41% | 70–76% | 0.000 | 20% | 52% (27 loops) | 24–26% | 0.59/0.48 |
| Olmo-3-7B-Instruct | 93% | 96–99% | 0.719 | 51% | 68% (5 loops) | 67–71% | 0.07/0.02 |
| Qwen3-14B | 57% | 61–71% | 0.650 | 12% | 23% (2 loops) | 22–22% | 0.11/0.11 |
| Qwen2.5-14B-Instruct | 26% | 50–78% | 0.001 | 10% | 37% (9 loops) | 20–22% | 0.04/0.09 |
| phi-4 | 41% | 66–68% | 0.002 | 21% | 37% (0 loops) | 39–42% | 0.01/0.00 |
| Mistral-Small-24B-Instruct-2501 | 22% | 54–67% | 0.000 | 16% | 22% (0 loops) | 19–24% | 0.69/0.19 |
| gemma-4-31B-it | 8% | 7–11% | 1.000 | 1% | 3% (0 loops) | 2–2% | 1.00/1.00 |
| Qwen3-32B | 56% | 69–74% | 0.090 | 10% | 34% (4 loops) | 21–21% | 0.06/0.06 |
| Qwen2.5-32B-Instruct | 23% | 38–51% | 0.052 | 10% | 31% (0 loops) | 18–26% | 0.01/0.20 |
| Olmo-3.1-32B-Instruct | 88% | 90–93% | 1.000 | 41% | 48% (4 loops) | 47–59% | 0.09/0.71 |
| Qwen2.5-72B-Instruct | 9% | 26–34% | 0.005 | 6% | 23% (0 loops) | 17–18% | 0.02/0.03 |

"p máx.": o maior dos três p-valores (Fisher) das frases no prompt contra o
controle do mesmo cenário.

### Acertos desfeitos pela frase

| Modelo | Acertos | Desfeitos no controle | Desfeitos com frase |
|---|---|---|---|
| Qwen3-8B (rtxpro6000) | 3 | 1/6 | 8/18 |
| Qwen3-14B | 2 | 1/4 | 4/12 |
| Qwen2.5-14B-Instruct | 2 | 1/4 | 7/12 |
| phi-4 | 3 | 0/6 | 4/18 |
| Mistral-Small-24B-Instruct-2501 | 4 | 2/8 | 10/24 |
| Qwen3-32B | 5 | 0/10 | 8/30 |
| Qwen2.5-32B-Instruct | 8 | 1/16 | 11/48 |
| Olmo-3.1-32B-Instruct | 6 | 5/12 | 22/36 |
| Qwen2.5-72B-Instruct | 6 | 1/12 | 10/36 |
| **Intermediários, somados** | | **12/78 (15%)** | **84/234 (36%)**, p = 0,0006 |
| gemma-4-31B-it | 53 | 3/106 (3%) | 11/318 (3%), p = 1,0 |

### Ativações

| Modelo | Camadas | Convergência | Camada da escolha | Gabarito no top-5 | Deslocamento na última camada: ruído / prompt / raciocínio | Raciocínio, distância fixa |
|---|---|---|---|---|---|---|
| LFM2.5-1.2B-Instruct | 16 | 94% | 81% | 0% | 0.018 / 0.037 / 0.069 | 0.049 |
| LFM2-2.6B | 30 | 97% | 73% | 25% | 0.021 / 0.037 / 0.085 | 0.047 |
| Qwen2.5-7B-Instruct | 28 | 96% | 79% | 63% | 0.021 / 0.040 / 0.129 | 0.130 |
| Qwen3-8B (rtxpro6000) | 36 | 94% | 67% | 78% | 0.012 / 0.023 / 0.163 | 0.207 |
| Olmo-3-7B-Instruct | 32 | 94% | 53% | 0% | 0.015 / 0.039 / 0.051 | 0.038 |
| Qwen3-14B | 40 | 95% | 78% | 61% | 0.011 / 0.021 / 0.101 | 0.108 |
| Qwen2.5-14B-Instruct | 48 | 98% | 76% | 57% | 0.019 / 0.050 / 0.106 | 0.102 |
| phi-4 | 40 | 88% | 55% | 16% | 0.017 / 0.040 / 0.165 | 0.181 |
| Mistral-Small-24B-Instruct-2501 | 40 | 95% | 72% | 76% | 0.012 / 0.027 / 0.065 | 0.073 |
| gemma-4-31B-it | 60 | 100% | 73% | 16% | 0.009 / 0.017 / 0.016 | 0.016 |
| Qwen3-32B | 64 | 98% | 83% | 62% | 0.013 / 0.029 / 0.096 | 0.102 |
| Qwen2.5-32B-Instruct | 64 | 97% | 81% | 50% | 0.016 / 0.035 / 0.134 | 0.143 |
| Olmo-3.1-32B-Instruct | 64 | 93% | 42% | 6% | 0.016 / 0.034 / 0.045 | 0.054 |
| Qwen2.5-72B-Instruct | 80 | 98% | 90% | 57% | 0.013 / 0.026 / 0.047 | 0.056 |

Convergência e camada da escolha em profundidade relativa (camada ÷ número
de camadas). "Gabarito no top-5": nos erros, fração em que o dígito correto
está entre os 5 mais prováveis na última camada. Essa coluna **não é
comparável** entre tokenizadores de 1 dígito por token (Qwen, Gemma, Mistral)
e de blocos de até 3 dígitos (OLMo, LFM, phi-4). "Distância fixa": frase
inserida a 1–3 mil caracteres do fim da resposta.

## Limitações

- **Acurácia perto de zero na maioria dos modelos.** Nesses modelos o
  "desvio" mede troca entre respostas erradas, não perda de acertos. A
  análise de acertos desfeitos se apoia em poucos problemas por modelo.
- **Contaminação.** O Gemma 4 é posterior à publicação do RuleArena (dez.
  2024). Em 52 dos 53 acertos, as parcelas que ele escreve somam o total que
  ele declara, o que descarta respostas decoradas de forma simples, mas não
  o treino com problemas parecidos. O teste decisivo, com variantes
  contrafactuais inéditas, ainda não foi feito.
- **Logit lens.** É uma leitura aproximada das camadas intermediárias
  (Belrose et al., 2023). As diferenças de profundidade entre modelos
  pedem confirmação com tuned lens ou *activation patching*.
- **Teacher forcing.** As ativações medem o efeito da frase na *leitura* de
  uma resposta fixa, não o caminho da geração livre.
- **Um prompt, uma seed, decodificação gulosa.** Os modelos de raciocínio
  (VibeThinker) foram avaliados de forma gulosa, e não com a amostragem que
  os autores recomendam.
- **Modelos ausentes.** Llama 3.1 8B e 3.3 70B (restritos) não foram
  avaliados. Os VibeThinker 1.5B e 3B estavam em execução quando esta versão
  foi publicada e não entram nas tabelas.

## Reprodução

Requisitos: Linux, Python ≥ 3.10 e GPU NVIDIA (CUDA 12.8+) ou AMD (ROCm 6.x).
Os modelos até 32B cabem numa GPU de 96 GB; o 72B precisa de duas.

```bash
git clone --recurse-submodules <url-deste-repositório>
cd <repositório>
bash setup_maquina.sh              # .venv, PyTorch, dependências, kernel Jupyter e testes
source .venv/bin/activate
python -m pytest tests -q          # não precisa de GPU

# um modelo
bash rodar_informacao_irrelevante.sh Qwen/Qwen3-8B --gpu 0 --lote 16
# ou a fila inteira de fila_modelos.yaml (um modelo por GPU; os de 2 GPUs na ordem)
bash rodar_fila.sh --listar
bash rodar_fila.sh
```

Detalhes de execução, arquiteturas suportadas e armadilhas conhecidas estão em
[LEIA-ME_MAQUINA_GRANDE.md](LEIA-ME_MAQUINA_GRANDE.md). A revisão de
literatura está em
[REVISAO_LITERATURA_INFORMACAO_IRRELEVANTE.md](REVISAO_LITERATURA_INFORMACAO_IRRELEVANTE.md).

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `src/interp/` | inserção das frases, geração, captura de ativações, resumos e fila de modelos |
| `src/airline/`, `src/evaluator_v2_1.py` | prompt do RuleArena, gabarito e avaliação da resposta final |
| `external/rulearena/` | RuleArena (submódulo, commit `59f60f6b`, licença MIT) |
| `notebooks/informacao_irrelevante_ativacoes.ipynb` | notebook do estudo (roda um modelo) |
| `notebooks/execucoes/` | o notebook executado para cada modelo, com todas as saídas e gráficos |
| `notebooks/acompanhamento.ipynb` | painel das execuções e da fila (não usa GPU) |
| `results/irrelevant_info/<execução>/` | `manifest.json` e `generations.jsonl` de cada modelo |
| `results/irrelevant_info/resumo/` | tabelas-resumo (acurácia, comportamento, acertos desfeitos, ativações) |
| `results/irrelevant_info/_arquivado_repetition_penalty_1.05/` | execução inicial do Qwen2.5-72B, feita com a penalidade de repetição herdada do `generation_config` (bug corrigido); guardada só como registro |
| `tests/` | testes da inserção das frases, dos hooks por arquitetura e da fila (sem GPU) |

**Fora do repositório:** as capturas de ativações (`captures/*.pt`, ~23 GB).
Elas são reproduzíveis com o notebook, e as tabelas de `resumo/` guardam o
que se tirou delas.

## Referência do benchmark

Zhou, R. et al. *RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in
Real-World Scenarios.* ACL 2025. [arXiv:2412.08972](https://arxiv.org/abs/2412.08972)

## Licença

A definir. O RuleArena, incluído como submódulo, é distribuído sob licença MIT.
