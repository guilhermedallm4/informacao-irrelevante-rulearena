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

## Edição manual dos pesos do Qwen3-14B (sem treinamento)

Experimento complementar (novembro de 2026): é possível fazer o Qwen3-14B
acertar mais problemas **só trocando valores nos pesos**, sem gradiente, sem
dados de treino e sem mudar o prompt? Código em [`edicao_pesos/`](edicao_pesos/),
resultados em [`results/edicao_pesos/`](results/edicao_pesos/).

### De onde vêm os erros

Comparando as 90 respostas limpas do Qwen3-14B com o gabarito item a item,
os erros não são de conta, são de regra, e três deles dominam:

| Regra aplicada errado | Problemas afetados (de 90) |
|---|---|
| Mala de 63–65 polegadas cobrada como "acima de 65" ($200 em vez de $30) | 53 |
| Peso de 51–53 lb cobrado na faixa de 54–70 lb ($100 em vez de $30) | 31 |
| 3ª mala em rota dos EUA cobrada a $200 em vez de $150 | 20 |

Nos três casos o modelo repete o texto do exemplo resolvido do prompt
("over 65 inches", "Third Bag: $200") em vez de consultar a tabela.

### Como os pesos foram editados

1. **Sondas.** Das próprias respostas do modelo foram extraídos os pontos de
   decisão: o prefixo cortado logo antes do dígito decisivo (por exemplo, em
   "For dimensions over 6" o próximo token deve ser "2", não "5"). Cada sonda
   tem o token correto e o errado, o que dá uma medida token a token do
   efeito de cada edição sem gerar respostas inteiras.
2. **Onde está o erro.** A atribuição direta ao logit por cabeça de atenção
   (1 600 cabeças) explica pouco: as cabeças somam −0,8 de uma diferença de
   logit de −7, e as mesmas cabeças ajudam num tipo de erro e atrapalham
   noutro. O empurrão para o dígito errado vem das MLPs das camadas 33–39.
   Já nas camadas 20–27 as ativações das MLPs separam "63–65" de "acima de
   65" com AUC 0,98: o modelo *sabe* o tamanho da mala e erra a regra depois.
3. **A edição.** Em cada camada escolhida, a matriz `down_proj` da MLP recebe
   uma correção de posto baixo, em forma fechada:

   ```
   W_down  <-  W_down + α · d̂ · wᵀ
   d̂ = (W_U[token correto] − W_U[token errado]) ⊙ γ_norm, normalizado
   w = (KᵀK + λI)⁻¹ Kᵀ y
   ```

   `K` são as ativações dos neurônios (entrada de `down_proj`) em **todas**
   as 1,36 milhão de posições de 180 respostas (as 90 da linha de base e as
   90 de uma primeira versão editada), `y` vale 1 nos pontos de decisão do
   erro e 0 em todo o resto; as posições vizinhas que não podem mudar (malas
   acima de 65 polegadas, pesos de 54–70 lb, as outras linhas "Bag: $") pesam
   100× na soma. `λ` = 0,1 da norma média das ativações e `α` foram escolhidos
   à mão por varredura. É mínimos quadrados fechados, sem gradiente nem
   iterações. Camadas e alvos:

   | Camada | Alvo | Token correto × errado |
   |---|---|---|
   | 23 | "For dimensions over 6" ▸ (63–65 in) | 2 × 5 |
   | 23 | "For dimensions over 62 inches up to " ▸ | 6 × 1 |
   | 24 | "For weights over 5" ▸ (51–53 lb) | 0 × 3 |
   | 31 | "Third Bag: $" ▸ (rota dos EUA) | 1 × 2 |
   | 31 | "Third Bag: $1" ▸ | 5 × 0 |

   A correção é guardada em forma fatorada (`U`, `V` em fp32), porque somada
   entrada a entrada a um peso em bf16 ela é arredondada a zero.
4. **Avaliação.** Mesmo protocolo do estudo (1-shot, decodificação gulosa,
   4 096 tokens, lote 16). Como a geração em lote sozinha muda 57% das
   respostas deste modelo, a linha de base foi gerada de novo nesta máquina.

### Resultados

Linha de base e modelo editado, mesmo protocolo, mesma máquina (90 problemas,
pergunta sem frase):

| Modelo | Acertos (90) | C0 / C1 / C2 | a ±10% | Erro mediano |
|---|---|---|---|---|
| Qwen3-14B (execução publicada acima) | 2 (2.2%) | 2 / 0 / 0 | 27 (30%) | 14.9% |
| Qwen3-14B, linha de base refeita | 3 (3.3%) | 2 / 1 / 0 | 26 (29%) | 14.8% |
| Qwen3-14B editado, α = 500 / 500 / 300 | **6 (6.7%)** | 5 / 1 / 0 | **48 (53%)** | **9.1%** |
| Qwen3-14B editado, α = 800 / 800 / 500 | 5 (5.6%) | 4 / 1 / 0 | 36 (40%) | 10.5% (16 respostas cortadas em 4 096 tokens) |
| Rodada 3: 17 decisões, chaves só dos traços-gabarito | 0 (0%) | 0 / 0 / 0 | 13 (14%) | 67 respostas em laço |
| Rodada 4: 17 decisões, chaves com alvo zero nas respostas livres | 5 (5.6%) | 2 / 3 / 0 | 43 (48%) | 9.1% (6 em laço) |
| Rodada 5: rodada 2 + 12 decisões da rodada 4 (`config_v7B.json`) | 7 (7.8%) | 7 / 0 / 0 | 47 (52%) | **7.4%** (8 em laço) |
| Rodada 6, ciclo 6: 23 decisões em `up_proj`, filtradas pela porta da MLP (`config_c6.json`) | **10 (11.1%)** | 9 / 1 / 0 | 45 (50%) | 10.0% (0 em laço) |
| Rodada 6, ciclo 8: 25 decisões em `up_proj`, α calibrado por classe | 0 (0%) | 0 / 0 / 0 | 25 (28%) | 19.8% (12 em laço) |

Com α maior a correção passa do ponto: 16 das 90 respostas entram em laço
até o limite de tokens. A configuração 500 / 500 / 300 é a final da rodada 2;
as rodadas 3–6 estão descritas abaixo.

Quatro problemas novos passaram a ser acertados (c0_000, c0_003, c0_054,
c1_077) e um foi perdido (c1_095). No nível 0, as respostas a ±10% do
gabarito foram de 9 para 20 em 30 e o erro mediano de 17,9% para 7,9%. O
modelo editado fica, em aproximação, no nível do Qwen2.5-32B (53% a ±10%).

Token a token, nos pontos de decisão (sondas tiradas das respostas da linha
de base e da primeira versão editada, 1 em cada 4):

| Ponto de decisão | n | Sem edição | Editado |
|---|---|---|---|
| 63–65 in: "over 6" ▸ **2** | 106 | 47% | **96%** |
| 63–65 in: "over 62 inches up to " ▸ **6** | 72 | 69% | 69% |
| 51–53 lb: "over 5" ▸ **0** | 55 | 58% | **89%** |
| 3ª mala EUA: "Third Bag: $" ▸ **1** | 15 | 0% | **100%** |
| 3ª mala EUA: "Third Bag: $1" ▸ **5** | 6 | 100% | 100% |
| vizinho: >65 in ▸ 5 (não deve mudar) | 95 | 97% | 98% |
| vizinho: 54–70 lb ▸ 3 (não deve mudar) | 41 | 63% | 68% |
| vizinho: "Fourth Bag: $" ▸ 2 (não deve mudar) | 29 | 83% | 79% |

Tipos de erro por item nas 90 respostas (problemas afetados):

| Regra | Linha de base | Editado |
|---|---|---|
| Mala de 63–65 in na faixa errada | 51 | **14** |
| 3ª mala a $200 em vez de $150 | 21 | **5** |
| Peso 51–53 lb na faixa errada | 32 | 24 |
| Peso 54–70 lb na faixa errada | 24 | 29 |
| Mala >65 in com taxa errada ($150 em vez de $200 nos EUA/Canadá) | 26 | 25 |

### Rodadas 3–5: todas as decisões, meta de 60%

Pedido seguinte: estender a edição manual a todas as regras restantes e
tentar chegar a 60% de acerto exato. Não chegou: o melhor resultado destas
rodadas é 7/90 (7,8%); a rodada 6, abaixo, chega a 10/90. O que foi feito e
por que o teto é baixo:

1. **Censo por teacher forcing.** Para cada um dos 90 problemas foi escrito o
   *traço-gabarito*: a resposta ideal no formato exato do exemplo do prompt,
   com cada ponto de decisão marcado (fatia da lista de taxas, limite de peso
   70 lb das malas gratuitas de Business/First, faixa 63–65 / >65 in, faixa
   de peso, taxa de cada faixa por região, totais). 68 dos 90 traços batem
   com o gabarito; os outros 22 diferem porque o gabarito oficial reordena as
   malas para aplicar a franquia de Business/First às mais caras, coisa que
   nenhum modelo faz. Uma passagem do modelo sobre o traço dá, em 4 285
   pontos de decisão, o top-1 e a margem de logit sem gerar nada
   (`traco_gabarito.py`, `medir_traco.py`).
2. **17 classes, mesma forma fechada.** Cada classe (decisão, token correto)
   com ≥2 erros recebe uma correção de posto 1, w_j = A⁻¹ b_j com
   A = G + λI + β K_MᵀK_M (G = KᵀK de todas as posições; K_M os pontos
   marcados com peso β = 100), camada escolhida por validação cruzada
   (AUC contra os vizinhos da mesma decisão) entre 20 e 34, α_j calibrado a
   partir da inclinação medida da margem com α = 100
   (`construir_edicao_v5.py`, `calibrar_v5.py`). Nos traços, isso leva quase
   todas as decisões a 100%:

   | Decisão (token correto) | n | Sem edição | Rodada 4 |
   |---|---|---|---|
   | 1ª mala gratuita: "First Bag: $" ▸ 0 | 74 | 88% | **100%** |
   | 2ª mala a $100: "Second Bag: $" ▸ 1 | 13 | 23% | **100%** |
   | 3ª mala a $150: "Third Bag: $" ▸ 1 | 43 | 21% | **86%** |
   | 63–65 in: "over 6" ▸ 2 | 313 | 70% | **99%** |
   | >65 in: "over 6" ▸ 5 (não deve mudar) | 315 | 99% | 99% |
   | 70 lb (Business/First): "Within" | 57 | 33% | **100%** |
   | 50 lb: "Over" (não deve mudar) | 571 | 92% | 94% |
   | "but under" / "and over" | 571 | 92% | **98%** |
   | $450 na Ásia: "the fee is $" ▸ 4 | 35 | 63% | 74% |

3. **Rodada 3 vazou e colapsou (0/90).** As chaves ajustadas só com as
   posições dos traços (683 mil) disparavam no texto que o modelo escreve
   por conta própria e que não aparece no traço ideal ("within the Main
   Cabin allowance", "1st bag and 1st bag…", "2222…"): 67 das 90 respostas
   entraram em laço. Medir só nos traços não detecta isso.
4. **Rodada 4: alvo zero nas respostas livres.** G passou a incluir também as
   ativações de 360 respostas reais (linha de base, rodada 2 e as próprias
   respostas colapsadas da rodada 3, 2,9 milhões de posições;
   `coletar_G_livre.py`). As chaves passam a responder ≈1 só nos contextos
   certos do texto livre ("Third Bag: $", "For dimensions over 6") e ≈0,05
   no percentil 99 das demais posições (`vazamento_livre.py`). Resultado:
   5/90, 43 a ±10%, 6 laços, todos ligados à chave "Within" (α = 665).
5. **Rodada 5: combinação à mão.** Chaves da rodada 2 (camadas 23/24/31)
   mais 12 classes da rodada 4 com α reduzido à mão ("Within" 250, 2ª mala
   500, "and" 400, sem as classes já cobertas pela rodada 2): **7/90**, 47 a
   ±10%, erro mediano 7,4%, 8 laços. Sem as chaves de peso_lim
   (`config_v7A.json`) não há laço mas o acerto numa fatia de 32 problemas
   cai de 7 para 5.

Por que não chega a 60%: o acerto exato exige todos os itens certos, e um
problema tem de 4 a 10 malas. Mesmo na melhor configuração só 12% dos itens
saem sem nenhum erro (2% na linha de base), e 36 dos 90 problemas ainda têm
pelo menos um item errado. Dos 54 sem erro de item, só 7 acertam o total: os
outros erram fora dos itens (qual item vai de bagagem de mão, lista de taxas
por classe/rota escrita antes dos itens, soma final) e 22 exigem a
reordenação de malas que o gabarito oficial faz. Os erros restantes mais
frequentes na rodada 5 são o limite de 70 lb das malas gratuitas (23
problemas, a chave que mais faz laço quando forte), $150 em vez de $200 para
malas >65 in nos EUA (19), faixa 54–70 lb (10) e 2ª mala Main Plus a $0 em
vez de $45 (7, decisão que o modelo acerta no traço e erra no texto livre).
Cada uma dessas é uma chave a mais com o mesmo risco de vazamento, e o ganho
em acerto exato por chave tem sido de 1–2 problemas, dentro do ruído de lote.

### Rodada 6: edição em `up_proj`, filtrada pela porta da MLP

Nas rodadas anteriores a chave lia a entrada de `down_proj` (os 17 408
neurônios já filtrados pela porta) e escrevia a direção do dígito certo no
fluxo residual. Ela dispara em qualquer posição cujo padrão de neurônios se
pareça com o do ponto de decisão, e daí vêm o vazamento e os laços. A rodada 6
muda o lugar da correção: a chave v_j lê o fluxo residual na **entrada** da
MLP (5 120 dimensões, pós-norma) e o padrão u_j é injetado em `up_proj`,
antes da porta:

```
W_up  <-  W_up + α · u_j · v_jᵀ
v_j = (G_x + λI + β K_MᵀK_M)⁻¹ b_j                       (chave no fluxo residual)
u_j = ḡ ⊙ W_downᵀ (W_down diag(ḡ²) W_downᵀ + μI)⁻¹ d̂       (padrão sobre os neurônios)
```

O efeito na posição i é W_down (g_i ⊙ u_j) · (v_j·x_i), onde g_i =
silu(W_gate x_i) é a porta original do modelo e ḡ a porta média na classe.
A correção só chega ao fluxo residual pelos neurônios que a porta do próprio
modelo deixou abertos naquele contexto: `gate_proj` e `down_proj` não mudam,
e o filtro de contexto vem de graça. G_x soma x xᵀ em todas as posições dos
traços-gabarito e das respostas livres (alvo zero), K_M são as marcas
(alvo 1, peso β = 100), a camada de cada classe é escolhida por validação
cruzada entre 22 e 34, e cada u_j é normalizado para projeção média 1 em d̂
(`coletar_residual.py`, `construir_edicao_up.py`; `UpProjEditado` em
`edicoes.py`).

Quatro versões dos fatores diferem nos dados: **up1** (17 classes) usa como
positivos só as marcas dos traços-gabarito e como alvo zero os traços mais
4,2 milhões de posições de respostas livres; **up2** é o mesmo sem as camadas
23, 24 e 31, para somar à rodada 2; **up3** (23 classes) acrescenta como
positivos 7 450 marcas localizadas por regex nas respostas livres do próprio
modelo (`medir_livre.py`), de modo que a chave é ajustada no texto que o
modelo realmente escreve; **up4** (25 classes) acrescenta ainda as marcas e
as posições das respostas dos ciclos 6 e 7. Os α por classe foram escolhidos
à mão (`ajustar_config.py`) ou calibrados pela inclinação da margem medida
com α = 100 (`calibrar_v5.py`). Oito ciclos de geração, 90 problemas cada,
avaliados com `avaliar_ciclo.py`, que também detecta laços sem corte:

| Ciclo | Configuração | Acertos | C0 / C1 / C2 | a ±10% | Erro mediano | Laços |
|---|---|---|---|---|---|---|
| 1 | up1, α calibrado | 1 | 1 / 0 / 0 | 46 | 9,7% | 0 |
| 2 | rodada 2 + up1 | 6 | 5 / 0 / 1 | 26 | 16,8% | 6 |
| 3 | rodada 2 + up2 | 3 | 3 / 0 / 0 | 38 | 12,2% | 1 |
| 4 | up1, α retocado à mão | 7 | 7 / 0 / 0 | 42 | 10,8% | 3 |
| 5 | up3, α à mão, "Within" 450 | 6 | 5 / 1 / 0 | 30 | 15,3% | 0 |
| **6** | **up3, "Within" zerado (`config_c6.json`)** | **10** | 9 / 1 / 0 | 45 | 10,0% | **0** |
| 7 | ciclo 6 com três α retocados | 7 | 6 / 1 / 0 | 47 | 8,8% | 0 |
| 8 | up4, α calibrado por classe (15 de 25 no teto de 500) | 0 | 0 / 0 / 0 | 25 | 19,8% | 12 |

O ciclo 6 é o melhor resultado de todo o experimento em acerto exato: 10/90
contra 3/90 da linha de base, sem nenhuma resposta em laço. Três ressalvas:

- O ganho está nos problemas simples (9 dos 10 são de complexidade 0, 30
  problemas, 4 a 5 malas). Na aproximação ele não supera a rodada 2 (45
  contra 48 a ±10%, erro mediano 10,0% contra 9,1%).
- Por item, o ciclo 6 corrige outras regras que a rodada 2 (lista de taxas
  por mala, 2ª mala, faixas de peso das malas gratuitas) mas é pior nas
  faixas de tamanho: 25 problemas com erro em 63–65 in (14 na rodada 2) e
  37 em >65 in (25). As duas edições consertam coisas diferentes, e somá-las
  (ciclos 2 e 3) trouxe de volta os laços.
- O ciclo 8 mostra o limite: calibrar cada classe para margem 3 na geração
  livre pôs 15 das 25 chaves no teto de 500 e o modelo colapsou (12 laços,
  0/90), mesmo com a porta filtrando. A chave "Within" (limite de 70 lb das
  malas gratuitas) continua sendo a que derruba o modelo quando forte; zerá-la
  foi o que levou o ciclo 5 ao 6.

### O que se aprende

- **Três regras corrigidas sem treino.** Os três erros mais frequentes
  caíram de 51, 21 e 32 problemas para 14, 5 e 24, com correções de posto 1
  ou 2 em três matrizes, sem mexer em nenhum outro peso. A informação
  necessária (o tamanho da mala) já estava nas ativações intermediárias;
  faltava ligá-la ao dígito certo.
- **O acerto exato sobe pouco porque os erros são muitos e independentes.**
  Um problema tem de 4 a 10 malas, e basta um item errado. As regras
  restantes (taxa de $200 para malas grandes nos EUA, faixa de 54–70 lb,
  2ª mala, malas gratuitas de Business/First com limite de 70 lb) seguem
  erradas e limitam o acerto exato a 6/90. A aproximação, que depende de
  cada item, quase dobrou.
- **A primeira tentativa vazou.** Uma correção ajustada só com as sondas e
  480 posições aleatórias fixava o primeiro dígito, mas (a) o modelo seguia
  com "over 62 inches up to 115" em metade dos casos, (b) a correção de
  peso disparava também em "For weights over " e gerava "over 00 lbs" e
  (c) a da 3ª mala vazava para "Fourth Bag" ($150 em vez de $200). Com
  alvo zero em todas as 1,36 milhão de posições e peso 100× nos vizinhos,
  (b) e (c) sumiram; (a) melhorou na geração livre (131 linhas "62–65 $30"
  contra 13 na linha de base, nenhuma "62–115"), embora a sonda isolada
  desse ponto não tenha mudado.
- **Edição em bf16 some.** Somada entrada a entrada ao peso em bf16, a
  correção é arredondada a zero (cada entrada é menor do que a resolução do
  bf16 nos valores da matriz). É preciso mantê-la fatorada em fp32.
- **A porta da MLP é um filtro de contexto pronto.** Mover a correção de
  `down_proj` para `up_proj` faz o empurrão passar pela `gate_proj` original:
  só os neurônios que o modelo já abre naquele contexto transmitem a chave.
  Com isso, 23 chaves com α até 500 geraram zero laços (ciclos 5–7), onde a
  mesma quantidade de chaves em `down_proj` gerava 6–8. O ganho em acerto
  exato (10/90) vem daí, mas a calibração automática que leva todas as chaves
  ao teto colapsa do mesmo jeito (ciclo 8): a porta reduz o vazamento, não o
  elimina.
- **Teacher forcing não prevê a geração livre.** Nos traços-gabarito a
  rodada 3 fechava 17 decisões; na geração livre colapsou. O que a geração
  livre acrescenta é o texto do próprio modelo, que o traço ideal não tem.
  Incluir as respostas reais (inclusive as colapsadas) como alvo zero no
  ajuste resolveu o vazamento em forma fechada, sem gradiente.
- **O acerto exato não escala com chaves.** Cada chave corrige uma decisão;
  o acerto exato exige todas, e os erros fora dos itens (bagagem de mão,
  lista de taxas, reordenação das malas) não têm um ponto de decisão único
  para editar. A medida que responde à edição é a aproximação (26 → 48 a
  ±10%) e o erro mediano (14,8% → 7,4%), não o acerto exato (3 → 7, e 10
  na rodada 6 com aproximação pior).
- **Ruído de lote.** A linha de base refeita acerta 3 em vez de 2 e só 44
  das 90 respostas são idênticas à execução publicada, como já medido pelo
  controle do estudo. Diferenças de 1–2 acertos entre execuções não
  significam nada; a aproximação (±10%) e os tipos de erro por item são
  medidas mais estáveis.

Reprodução: `edicao_pesos/sondas.py` (sondas), `diagnostico.py` e
`coletar_mlp.py` (atribuição por cabeça e por MLP), `coletar_posicoes.py`
(estatísticas de todas as posições, rodada 2), `construir_edicao_v3.py`
(correções fechadas), `varrer_alpha.py` e `avaliar_sondas.py` (varredura e
avaliação token a token), `gerar.py` (90 problemas) e `comparar.py`. Os
fatores da rodada 2 estão em `results/edicao_pesos/deltas/fatores_v4_L{23,24,31}.pt`
(configuração `edicao_pesos/config_v4_500_500_300.json`). Rodadas 3–5:
`traco_gabarito.py` e `medir_traco.py` (traços e censo por teacher forcing),
`coletar_G_livre.py` (G das respostas livres), `construir_edicao_v5.py` e
`calibrar_v5.py` (17 classes e α), `vazamento_livre.py` (resposta das chaves
no texto livre), `juntar.py` (fatias por GPU); fatores em
`deltas/fatores_v6_L{22,26,30,32,34}.pt` com `deltas/classes_v6.json`, e a
configuração final (7/90) em `edicao_pesos/config_v7B.json`. Rodada 6:
`coletar_residual.py` (x na entrada da MLP e G_x nos traços e nas respostas
livres), `medir_livre.py` (marcas nas respostas livres do modelo),
`construir_edicao_up.py` (chaves e padrões em `up_proj`), `vazamento_up.py`
(empurrão de cada chave por posição e atribuição dos laços),
`ajustar_config.py` e `calibrar_v5.py` (α por classe), `avaliar_ciclo.py`
(acertos, ±10%, laços); fatores em `deltas/fatores_up3_L{22,26,28,30,32,34}.pt`
com `deltas/classes_up3.json` e a configuração do ciclo 6 (10/90) em
`edicao_pesos/config_c6.json`. Respostas de cada rodada em
`results/edicao_pesos/editado_*.jsonl` e `results/edicao_pesos/ciclo{1..8}.jsonl`.

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
| `edicao_pesos/`, `results/edicao_pesos/` | edição manual dos pesos do Qwen3-14B: scripts, configurações, fatores finais, gerações e avaliações token a token |

**Fora do repositório:** as capturas de ativações (`captures/*.pt`, ~23 GB).
Elas são reproduzíveis com o notebook, e as tabelas de `resumo/` guardam o
que se tirou delas.

## Referência do benchmark

Zhou, R. et al. *RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in
Real-World Scenarios.* ACL 2025. [arXiv:2412.08972](https://arxiv.org/abs/2412.08972)

## Licença

A definir. O RuleArena, incluído como submódulo, é distribuído sob licença MIT.
