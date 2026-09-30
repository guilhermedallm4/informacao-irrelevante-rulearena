# Estudo de informação irrelevante em outra máquina (modelos maiores)

Este pacote leva o estudo de `notebooks/informacao_irrelevante_ativacoes.ipynb`
para a máquina com **2× NVIDIA RTX PRO 6000 Blackwell (96 GB cada, 192 GB no
total; driver 595.84, CUDA 13.2)**, para repeti-lo com modelos maiores.
O estudo mede como frases irrelevantes ("I was counting 1+1.", "I stopped to
breathe.", "What is the value?") afetam a resposta e as ativações internas do
modelo em 90 problemas do RuleArena airline.

## O que vem no pacote

| Caminho | Conteúdo |
|---|---|
| `src/interp/` | inserção das frases, geração, captura de ativações (residual, MLP, logit lens, DLA, gabarito) |
| `notebooks/informacao_irrelevante_ativacoes.ipynb` | notebook do estudo, já executado com o Qwen3-8B (serve de referência) |
| `setup_maquina.sh` | cria a `.venv`, instala PyTorch para a GPU da máquina e as dependências, registra o kernel e roda os testes |
| `rodar_informacao_irrelevante.sh` | executa o estudo para um modelo, em segundo plano e retomável |
| `fila_modelos.yaml`, `rodar_fila.sh`, `src/interp/fila.py` | fila de modelos: um por GPU, os de 2 GPUs no fim (seção [Fila de modelos](#fila-de-modelos)) |
| `notebooks/acompanhamento.ipynb` | painel somente leitura: fila, status das execuções e resultados parciais |
| `requirements-interp.txt` | versões testadas das dependências |
| `external/rulearena/` | benchmark RuleArena no commit fixado `59f60f6b` (não precisa de internet para os dados) |
| `results/irrelevant_info/Qwen3-8B_one_shot_frac0.30_seed42/` | gerações e manifesto da execução de referência com o Qwen3-8B, para comparação |
| `.git/` | histórico do repositório, no branch `adaptacao-rtx5090` |

Fica de fora: a `.venv`, o cache de modelos do HuggingFace, as capturas de
ativações do Qwen3-8B (~900 MB, reproduzíveis) e a FOLIO v2 (dataset restrito,
cada pessoa baixa com o próprio token).

## Requisitos

- Linux, Python ≥ 3.10, `git`.
- GPU NVIDIA (driver com suporte a CUDA 12.8 ou mais novo) ou AMD com ROCm 6.x.
  O `setup_maquina.sh` detecta qual é. Na máquina das RTX PRO 6000 (CUDA 13.2)
  ele instala o PyTorch com CUDA 13.0, igual ao ambiente onde o estudo foi
  testado.
- Disco: ~10 GB para o ambiente, mais os pesos dos modelos no cache do
  HuggingFace (tabela abaixo) e ~3 GB de resultados por modelo.

## Passo a passo

```bash
# 1. Extrair
tar -xzf counterfactual_grounding_pacote_*.tar.gz
cd counterfactual_grounding

# 2. Instalar (cria .venv, instala PyTorch para a GPU, roda os testes)
bash setup_maquina.sh

# 3. Login no HuggingFace (só para modelos restritos, como Llama)
source .venv/bin/activate
hf auth login

# 4. Teste rápido: 6 problemas, ~15 min, numa GPU só
bash rodar_informacao_irrelevante.sh Qwen/Qwen3-32B --teste --gpu 0

# 5. Execução completa: 90 problemas, um modelo por GPU, em paralelo
bash rodar_informacao_irrelevante.sh Qwen/Qwen3-14B --gpu 0 --lote 16
bash rodar_informacao_irrelevante.sh Qwen/Qwen3-32B --gpu 1

# 6. Depois que as duas terminarem: o 72B ocupa as duas GPUs
bash rodar_informacao_irrelevante.sh Qwen/Qwen2.5-72B-Instruct --gpu 0,1 --lote 4
```

Para rodar vários modelos sem ficar lançando um por um, use a
[fila de modelos](#fila-de-modelos) em vez dos passos 5 e 6.

`--gpu` escolhe as GPUs visíveis para aquela execução. Sem ela, a execução
enxerga as duas e divide o modelo entre elas, o que desperdiça uma GPU num
modelo que caberia em uma. Não rode duas execuções na mesma GPU.

Cada execução grava:
- o notebook executado em `notebooks/execucoes/informacao_irrelevante_<modelo>_frac<fração>.ipynb`;
- o log em `results/irrelevant_info/<modelo>_frac<fração>.log`;
- gerações e capturas em `results/irrelevant_info/<modelo>_one_shot_frac<fração>_seed42/`.

Para acompanhar:

```bash
kill -0 $(cat results/irrelevant_info/Qwen3-32B_frac0.30.pid) && echo rodando || echo terminou
wc -l results/irrelevant_info/Qwen3-32B_one_shot_frac0.30_seed42/generations.jsonl.partial   # de 810
nvidia-smi
```

O processo continua se a sessão SSH cair. Se ele for interrompido (queda de
energia, `kill`), rodar o mesmo comando de novo retoma das gerações já salvas.
Quando termina, o código de saída do `nbconvert` fica em
`results/irrelevant_info/<modelo>_frac<fração>.exit` (0 = sucesso).

Outras opções do script:
- `--rotulo NOME`: acrescenta um sufixo aos nomes da execução (pasta de
  resultados, log, pid, notebook), para não misturar com uma execução anterior
  do mesmo modelo. Ex.: `--rotulo rtxpro6000` grava em
  `Qwen3-8B_one_shot_frac0.30_seed42_rtxpro6000/`.
- `--max-tokens N`: limite de tokens gerados (padrão 4096). Modelos de
  raciocínio longo precisam de mais: o VibeThinker usa ~15 mil tokens por
  resposta e roda com 32 768 (campo `max_tokens` na fila).
- `--primeiro-plano`: espera a execução terminar e sai com o código do
  `nbconvert`, em vez de devolver o terminal. A fila usa este modo.

## Fila de modelos

`rodar_fila.sh` roda em sequência os modelos de `fila_modelos.yaml`,
aproveitando as duas GPUs:

1. **Fase 1:** os modelos de 1 GPU, um em cada GPU. Quando uma GPU fica livre,
   o próximo modelo de 1 GPU (pela `ordem`) começa nela.
2. **Fase 2:** os modelos de 2 GPUs, um de cada vez, quando não sobrar
   nenhum modelo de 1 GPU **com ordem menor** (pendente ou rodando) e as duas
   GPUs estiverem livres.

A `ordem` vale também entre as fases. Um modelo de 2 GPUs funciona como uma
barreira: os de 1 GPU com ordem maior esperam por ele. Se algum deles já
estiver rodando e for o que ocupa as GPUs, a fila o **pausa logo depois do
próximo salvamento** (um lote de geração ou uma captura) e o retoma depois.
Assim quase nada se perde, e a pausa não conta como falha. Na fila atual, o
Qwen2.5-72B (200) roda antes dos VibeThinker (240 e 250).

Antes de lançar numa GPU, o escalonador confere com `nvidia-smi` que ela está
livre (menos de 2 GB em uso), para não colidir com outros processos da
máquina.

```bash
bash rodar_fila.sh --listar          # mostra a fila e o estado, sem executar nada
bash rodar_fila.sh                   # inicia em segundo plano (continua se o SSH cair)
tail -f results/irrelevant_info/fila.log
bash rodar_fila.sh --parar           # encerra o escalonador e as execuções dele
```

Com `--teste`, a fila inteira roda com 6 problemas por modelo, com estado e log
separados (`fila_estado_teste.json`, `fila_teste.log`). `--fila ARQUIVO`
usa outra fila (útil para testar com poucos modelos).

**A fila (`fila_modelos.yaml`):** cada modelo tem `id` (HuggingFace), `gpus`
(1 ou 2), `lote` e `ordem` (menor roda antes), e opcionalmente `rotulo`. O
Qwen3-8B usa `rotulo: rtxpro6000` para não misturar com a execução de
referência da RTX 5090.

**Sucesso e falha:** uma execução só conta como concluída com código de saída 0
**e** resultados completos (`generations.jsonl` e uma captura com gabarito por
problema). Se falhar, o motivo (fim do log) fica registrado, e ela é tentada de
novo **uma** vez; a geração retoma do que foi salvo. Se falhar de novo, fica
como "falhou" e a fila segue. `bash rodar_fila.sh --refazer-falhas` devolve os
modelos que falharam para a fila.

**Verificações antes de rodar:** para cada modelo, o escalonador baixa o
`config.json` do HuggingFace.
- Repositório restrito sem acesso (Llama sem `hf auth login` ou sem licença
  aceita): fica como "sem acesso", e a fila segue sem ele. Depois de obter o
  acesso, reiniciar a fila o coloca de volta.
- Modelo MoE (ex.: Qwen3-30B-A3B): fica como "recusado", porque os hooks
  exigem MLP densa.

**Estado e retomada:** o estado fica em
`results/irrelevant_info/fila_estado.json` (pendente / rodando, com GPU, PID e
início / concluído / falhou, com o motivo / sem acesso / recusado). Cada
execução roda na própria sessão e sobrevive ao escalonador. Se o escalonador
parar (kill, queda), rodar `bash rodar_fila.sh` de novo:
- acompanha as execuções que ainda estão rodando, sem relançá-las;
- registra as que terminaram enquanto ele estava parado (pelo arquivo `.exit`);
- retoma as que foram interrompidas;
- pula os modelos já concluídos, inclusive os que já estiverem completos no
  disco.

`--parar` é diferente: encerra também as execuções em andamento e as devolve
para "pendente". A próxima partida retoma das gerações salvas.

**Acompanhar:** a seção 0 de `notebooks/acompanhamento.ipynb` mostra o que está
em cada GPU, o que falta, o que terminou e o que falhou, com o motivo. O
notebook só lê arquivos: não carrega modelo nem usa GPU.

**Disco:** os modelos da fila somam ~600 GB de pesos no cache do HuggingFace
(`~/.cache/huggingface`), mais ~3 GB de resultados por modelo (bem mais no
72B e no 70B).

## Modelos sugeridos (2 × 96 GB)

A lista completa em uso, com 11 modelos (Qwen3, Qwen2.5, Mistral Small, phi-4
e Llama), está em `fila_modelos.yaml`. Alguns deles:

| Modelo | Pesos (bf16) | Camadas | GPUs | Lote sugerido | Observação |
|---|---:|---:|---|---:|---|
| `Qwen/Qwen3-8B` | 16 GB | 36 | 1 | 16 | referência (já executado na RTX 5090) |
| `Qwen/Qwen3-14B` | 30 GB | 40 | 1 | 16 | |
| `Qwen/Qwen3-32B` | 66 GB | 64 | 1 | 8 | mesma família, 4× o tamanho |
| `Qwen/Qwen2.5-72B-Instruct` | 145 GB | 80 | 2 | 4 | dividido entre as duas GPUs |
| `meta-llama/Llama-3.3-70B-Instruct` | 141 GB | 80 | 2 | 4 | restrito: aceitar a licença no HuggingFace; números viram tokens de até 3 dígitos |

Use `--lote N` para mudar o tamanho do lote. Um lote que não cabe na
memória é dividido ao meio automaticamente, então um valor alto só custa
algumas tentativas.

Quando o modelo usa as duas GPUs (`--gpu 0,1`), ele é dividido em camadas
entre elas (`device_map="auto"`). As camadas rodam uma GPU de cada vez, então
isso serve para caber na memória, não para ganhar velocidade.

**Comparação com o Qwen3-8B:** a execução de referência foi numa RTX 5090.
Rodar o Qwen3-8B de novo nesta máquina separa o efeito do tamanho do modelo do
efeito do hardware, que no relatório de validação mudou 22–38% das predições.
É um bom primeiro teste completo; na RTX 5090 ele levou ~4 h com lote 8.

## Configuração avançada

O notebook lê variáveis de ambiente, que o script já define:

| Variável | Padrão | Efeito |
|---|---|---|
| `IRR_MODEL` | `Qwen/Qwen3-8B` | modelo do HuggingFace |
| `IRR_FRACTION` | `0.30` | fração de cada nível de complexidade (0.30 → 90 problemas) |
| `IRR_BATCH` | `8` | tamanho do lote de geração |
| `IRR_MAX_NEW_TOKENS` | `4096` | limite de tokens gerados |
| `IRR_BAND` | reescalado | faixa de camadas sombreada nos gráficos, ex. `40,46`. Sem ela, usa a mesma profundidade relativa de 22–26 no Qwen3-8B |

## Arquiteturas suportadas

`src/interp/activations.py` (`resolve_arch`) localiza, em cada modelo, o que
entra no residual. As fórmulas das métricas são as mesmas em todos.

| Layout | Modelos | O que é lido como escrita no residual |
|---|---|---|
| pré-norma (Llama) | Qwen2.5, Qwen3, Mistral, phi-4, Llama | saída da atenção e da MLP |
| pós-norma "sanduíche" | OLMo 2/3, Gemma 2–4 | saída de `post_attention_layernorm` e `post_feedforward_layernorm` |
| híbrido | LFM2 / LFM2.5 (Liquid) | saída da convolução curta **ou** da atenção (o "mixer" da camada) e da MLP |

Particularidades tratadas:
- **Gemma 4:** `layer_scalar`, que multiplica o residual ao fim de cada
  camada e entra na DLA, e *softcap* dos logits, que entra no logit lens.
- **Wrappers multimodais** (`model.model.language_model`).
- **LFM2:** a norma final se chama `embedding_norm`, e nas camadas de
  convolução as métricas "de atenção" descrevem a convolução.

`tests/test_arch.py` verifica, em modelos minúsculos de cada layout, que:
- o residual é a soma das escritas capturadas;
- o logit lens da última camada é igual à saída do modelo;
- a DLA soma a diferença entre o logit do alvo e o logit médio;
- o padding à esquerda é só ruído numérico.

**Tokenização de números:** Qwen e Gemma usam um token por dígito. OLMo 3,
LFM2, phi-4 e Llama 3 agrupam até 3 dígitos por token. A análise do gabarito
funciona por token, então nesses modelos o "ponto de decisão" pode ser um
bloco de dígitos.

**Decodificação gulosa pura:** `generate` parte do `generation_config.json`
de cada modelo, e alguns trazem ajustes que mudam a escolha gulosa (Qwen2.5:
`repetition_penalty=1.05`; LFM2.5-2.6B: 1.1). `experiment.GREEDY` zera esses
ajustes para todos os modelos. As execuções anteriores a essa correção estão
em `results/irrelevant_info/_arquivado_repetition_penalty_1.05/`.

## Limitações conhecidas

- **Modelos MoE** (ex.: `Qwen3-30B-A3B`, `LFM2.5-8B-A1B`) não são suportados:
  não há um conjunto único de neurônios de MLP por camada. A fila os recusa.
- **Modelos de raciocínio** (QwQ, Qwen3 com `enable_thinking=True`,
  LFM2.5-2.6B, que sempre abre `<think>`) ficam de fora. Os prompts desativam
  o modo de raciocínio do Qwen3 e do Gemma 4.
- **Modelos restritos** (Llama, Gemma 3) exigem aceitar a licença no site do
  HuggingFace com a conta do token; sem isso, a fila os marca como
  "sem acesso".
- **Tempo:** com o Qwen3-8B numa RTX 5090, a geração levou ~4 h. Modelos
  maiores geram mais devagar por token; conte com várias horas por modelo
  para a execução completa.
- **Monitoramento:** use o PID do arquivo `.pid`. Não use
  `pgrep -f jupyter-nbconvert`: o próprio comando de busca contém esse texto e
  se encontra, então parece que o processo nunca termina.
- **Não edite os scripts `.sh` no lugar com execuções rodando:** o `bash` lê
  o script aos poucos, e uma execução em andamento continuaria lendo o arquivo
  alterado a partir da posição antiga. Grave a versão nova em outro arquivo e
  troque com `mv` (as execuções em andamento ficam com a versão antiga).
- **Numeração das GPUs:** as execuções usam `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  para que a GPU 0 do CUDA seja a GPU 0 do `nvidia-smi`, que é onde a fila
  confere a memória livre.

## Resultado de referência (Qwen3-8B, RTX 5090)

- A resposta mudou em 40% dos problemas só por gerar noutro lote. Uma frase na
  pergunta elevou isso para 64–67% (p ≤ 0,002).
- No meio da conta, só "I was counting 1+1." teve efeito: 53% de desvio, com
  25 de 90 gerações presas num loop aritmético até o limite de tokens.
- A resposta converge na camada 34, e as camadas que mais a amplificam são a
  34, a 35 e a 36. A escolha entre o dígito certo e o errado se fixa por volta
  da camada 24.

O notebook de referência tem todos os números e gráficos.
