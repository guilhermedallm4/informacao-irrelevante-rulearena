# Revisão de literatura: informação irrelevante, escala e mecanismos internos

Relatório de apoio ao estudo de informação irrelevante
(`notebooks/informacao_irrelevante_ativacoes.ipynb`). Este estudo insere frases
neutras ("I was counting 1+1.", "I stopped to breathe.", "What is the value?")
na pergunta ou no meio de uma conta do raciocínio do modelo, em 90 problemas do
RuleArena airline. Mede o efeito no comportamento (a resposta muda?) e nas
ativações internas (fluxo residual, logit lens, atribuição direta ao logit,
neurônios da MLP e camada em que o modelo escolhe o dígito errado).

Para cada trabalho: o que estuda, como estuda, o que encontra e como se
relaciona com o nosso estudo. As referências foram conferidas em 26/09/2026.
Três trabalhos muito recentes foram lidos só pelo resumo e estão marcados
como tal.

## Resumo

| # | Trabalho | Tema | Relação principal com o nosso estudo |
|---|---|---|---|
| 1 | Shi et al., ICML 2023 | distração por contexto irrelevante | referência do fenômeno comportamental |
| 2 | Mirzadeh et al., ICLR 2025 | distratores "no-op" em matemática | efeito em todas as escalas; crítica sobre a qualidade dos distratores |
| 3 | Gema et al., 2025 | escala inversa com raciocínio mais longo | loops aritméticos depois de "1+1" |
| 4 | Lanham et al., 2023 | fidelidade do raciocínio escrito × escala | frase no raciocínio pesa menos no 72B |
| 5 | Yuan et al., 2025 | não determinismo em bf16 | nosso controle de 40% |
| 6 | Thinking Machines, 2025 | kernels invariantes ao lote | como eliminar o ruído do controle |
| 7 | Stolfo et al., EMNLP 2023 | aritmética: atenção + MLPs tardias | onde procurar o cálculo |
| 8 | Nikankin et al., ICLR 2025 | aritmética por "saco de heurísticas" | erros de "quase acerto" |
| 9 | Lad et al., NeurIPS 2025 | estágios de inferência | convergência nas camadas finais |
| 10 | Niu et al., 2025 | *contextual entrainment* | "1+1" puxando a conta |
| 11 | arXiv 2609.17804 (resumo) | distração localizada em cabeças de atenção | mecanismo candidato |
| 12 | arXiv 2606.24077 (resumo) | *entrainment* de frases inteiras | mecanismo candidato |
| 13 | Hu et al., 2025 | profundidade efetiva × escala | contraponto à hipótese de profundidade |
| 14 | Wendler et al., ACL 2024 | fases do logit lens em 7B/13B/70B | profundidade relativa estável |
| 15 | Halawi et al., 2023 | camada crítica sob contexto enganoso | análogo da nossa "camada da escolha" |
| 16 | Belrose et al., 2023 | tuned lens × logit lens | cuidado metodológico |
| 17 | arXiv 2606.29196 (resumo) | profundidade de representações muda com a escala | a direção depende do que se mede |

---

## 1. Distração por contexto irrelevante

### 1.1 Shi et al. (2023). *Large Language Models Can Be Easily Distracted by Irrelevant Context.* ICML 2023.

- **Estudo:** cria o GSM-IC, versão do GSM8K em que cada problema recebe uma
  frase com informação irrelevante. Compara técnicas de prompting
  (chain-of-thought, least-to-most, self-consistency, exemplos com
  distratores).
- **Resultados:** o desempenho cai de forma acentuada com a frase
  irrelevante, e **uma única frase já basta** para degradar bastante. As
  técnicas de prompting continuam suscetíveis. Self-consistency, exemplos
  que contêm distratores e a instrução de ignorar informação irrelevante
  melhoram a robustez.
- **Relação com o nosso estudo:** é a referência do fenômeno. Os
  distratores deles são **temáticos**: falam de pessoas e quantidades do
  problema. Os nossos são **neutros**, sem números do problema, e mesmo
  assim o Qwen3-8B desvia 64–67% das respostas, contra 40% do controle.

### 1.2 Mirzadeh et al. (2025). *GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models.* ICLR 2025.

- **Estudo:** gera variantes simbólicas do GSM8K, trocando nomes e números.
  A variante **GSM-NoOp** acrescenta orações que parecem relevantes mas não
  mudam a conta.
- **Resultados:** a acurácia varia entre instâncias do mesmo problema, e o
  GSM-NoOp causa **quedas de até 65%** em todos os modelos testados,
  inclusive os maiores. Os modelos tendem a converter a oração irrelevante
  numa operação.
- **Crítica posterior:** análises independentes (Ivanova, blog do ICLR 2025;
  "Revisiting GSM-Symbolic", LessWrong) argumentam que parte do efeito vem
  da qualidade dos distratores, e que uma auditoria deles reduz muito a
  queda.
- **Relação com o nosso estudo:** mostra que o efeito não some com a
  escala. A crítica reforça a escolha de frases **claramente neutras** e de
  um **controle explícito**, os dois presentes no nosso desenho.

### 1.3 Gema et al. (2025). *Inverse Scaling in Test-Time Compute.* arXiv:2507.14417.

- **Estudo:** constrói tarefas (contagem com distratores, regressão com
  atributos espúrios, dedução com restrições) e varia o comprimento do
  raciocínio de modelos de raciocínio.
- **Resultados:** raciocinar por **mais tempo piora** o desempenho nessas
  tarefas. Modelos Claude ficam cada vez mais distraídos por informação
  irrelevante, enquanto os da série o da OpenAI resistem aos distratores mas
  se prendem à formulação do problema.
- **Relação com o nosso estudo:** a escala aqui é o tempo de raciocínio, não
  o número de parâmetros. Tem paralelo com os 25 de 90 casos em que o
  Qwen3-8B entrou num loop aritmético depois de "I was counting 1+1." e só
  parou no limite de tokens.

## 2. Raciocínio escrito e escala

### 2.1 Lanham et al. (2023). *Measuring Faithfulness in Chain-of-Thought Reasoning.* arXiv:2307.13702 (Anthropic).

- **Estudo:** intervém no raciocínio escrito (inserir erros, truncar,
  parafrasear, trocar por texto de preenchimento) e mede se a resposta final
  muda.
- **Resultados:** a dependência da resposta em relação ao raciocínio escrito
  varia muito entre tarefas. Em modelos **maiores e mais capazes**, a
  resposta tende a depender **menos** do texto do raciocínio (escala inversa
  de fidelidade).
- **Relação com o nosso estudo:** o cenário *reasoning* é uma intervenção do
  mesmo tipo: inserimos uma frase no meio de uma conta. Nos testes
  preliminares, a frase no raciocínio desloca o residual do Qwen2.5-72B ~3
  vezes menos do que nos modelos de 7–8B. Lanham et al. oferecem uma
  explicação candidata: modelos maiores "leem" menos o próprio raciocínio.
  Isso ainda precisa ser confirmado com a amostra completa.

## 3. Não determinismo da decodificação gulosa

### 3.1 Yuan et al. (2025). *Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning.* arXiv:2506.09501.

- **Estudo:** mede quanto mudam as respostas ao variar tamanho do lote,
  número de GPUs, tipo de GPU e precisão numérica, com decodificação gulosa.
- **Resultados:** em **bf16**, essas mudanças de configuração alteram as
  respostas. Com o DeepSeek-R1-Distill-Qwen-7B, a acurácia varia até 9% e o
  comprimento da resposta até 9 000 tokens. O efeito é maior em cadeias de
  raciocínio longas, porque um arredondamento diferente num token inicial
  leva a uma cadeia diferente. FP32 é quase perfeitamente reprodutível. Os
  autores propõem o LayerCast: pesos em 16 bits, contas em FP32.
- **Relação com o nosso estudo:** explica o controle. No Qwen3-8B, gerar a
  mesma pergunta em outro lote mudou 40% das respostas, e 2 de 4 problemas
  deram respostas diferentes entre a RTX 5090 e a RTX PRO 6000. Qualquer
  comparação entre modelos precisa do controle de cada um.

### 3.2 Thinking Machines Lab (2025). *Defeating Nondeterminism in LLM Inference.* Blog técnico.

- **Estudo:** identifica a causa do não determinismo com temperatura 0:
  kernels (matmul, RMSNorm, atenção) cujo resultado para uma sequência
  depende do tamanho do lote em que ela é processada.
- **Resultados:** versões **invariantes ao lote** desses kernels dão
  reprodutibilidade total (1 000 execuções idênticas), com código público
  demonstrado justamente no Qwen3-8B.
- **Relação com o nosso estudo:** é o caminho para eliminar o ruído do
  controle numa próxima rodada, junto com o LayerCast. Com o controle perto
  de zero, o efeito das frases fica isolado.

## 4. Mecanismo interno da aritmética

### 4.1 Stolfo, Belinkov e Sachan (2023). *A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis.* EMNLP 2023.

- **Estudo:** análise de mediação causal (intervenções nas ativações) em
  modelos respondendo a perguntas aritméticas.
- **Resultados:** as camadas iniciais processam números e operadores; a
  **atenção** leva essa informação até a última posição; **MLPs tardias**
  calculam e escrevem o resultado no residual.
- **Relação com o nosso estudo:** justifica medir DLA separada para atenção
  e MLP, e ler as ativações nas posições que produzem os dígitos da
  resposta.

### 4.2 Nikankin et al. (2025). *Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics.* ICLR 2025.

- **Estudo:** identifica o circuito aritmético de vários LLMs e classifica
  neurônios individuais das MLPs.
- **Resultados:** os modelos não usam um algoritmo robusto nem memorização,
  e sim um **"saco de heurísticas"**: neurônios nas camadas intermediárias e
  tardias que reagem a padrões dos operandos (por exemplo, um operando numa
  faixa de valores). A combinação desses neurônios explica a maior parte da
  acurácia.
- **Relação com o nosso estudo:** combina com os erros de "quase acerto": o
  dígito correto costuma estar no top-5 no ponto de decisão, mas raramente
  no top-2. Também motiva a análise de neurônios da MLP (Jaccard e mudanças
  maiores que 3σ).

### 4.3 Lad, Gurnee e Tegmark (2025). *The Remarkable Robustness of LLMs: Stages of Inference?* NeurIPS 2025.

- **Estudo:** remove e troca camadas de oito modelos e observa o efeito.
- **Resultados:** os modelos são robustos à remoção de camadas
  intermediárias e sensíveis à das iniciais e finais. Os autores propõem
  quatro estágios universais: destokenização, engenharia de atributos,
  *ensembling* da predição e **afiação do residual nas últimas camadas**.
- **Relação com o nosso estudo:** nos três modelos testados, a resposta só
  converge acima de 95% da profundidade, coerente com o estágio de afiação.

## 5. Mecanismo da distração

### 5.1 Niu et al. (2025). *Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs.* arXiv:2505.09338.

- **Estudo:** mede a tendência de repetir tokens do contexto e procura o
  circuito responsável, com mascaramento diferenciável de cabeças de
  atenção.
- **Resultados:** tokens que aparecem no prompt recebem logits mais altos
  **mesmo quando são irrelevantes ou aleatórios** (*contextual entrainment*).
  O efeito depende de "cabeças de *entrainment*", e desligá-las o reduz
  bastante.
- **Relação com o nosso estudo:** é um mecanismo candidato para o efeito de
  "I was counting 1+1." no meio da conta. Os tokens "1" e "+" entram no
  contexto justamente onde o modelo está somando.

### 5.2 *A Four-Stage Decomposition of Word-Problem Solving and Mechanistic Fragility in LLM Math Reasoning.* arXiv:2609.17804 (setembro de 2026). **Lido só o resumo.**

- **Estudo e resultados (segundo o resumo):** decompõe a resolução de
  problemas em quatro estágios (abstração do esquema, planejamento da
  operação, ligação dos operandos, cálculo). A falha causada por um
  distrator se localiza no **planejamento da operação**, implementado por
  cabeças de atenção cujo papel causal é validado.
- **Relação com o nosso estudo:** sugere onde procurar com *activation
  patching*.

### 5.3 *Sentence-Level Contextual Entrainment in Large Language Models.* arXiv:2606.24077 (2026). **Lido só o resumo.**

- **Estudo e resultados (segundo o resumo):** estende o *entrainment* de
  tokens para orações inteiras. Uma oração irrelevante degrada modelos que
  resolviam o problema de forma confiável.
- **Relação com o nosso estudo:** as nossas frases são orações inteiras
  neutras, o caso estudado ali.

## 6. Profundidade da decisão e escala do modelo

Esta seção responde à pergunta: **a resposta de modelos maiores se define em
camadas relativamente mais profundas?** Nenhum trabalho encontrado mede a
camada em que o modelo decide o resultado de uma conta sob distratores,
comparando escalas. Os mais próximos:

### 6.1 Hu et al. (2025). *What Affects the Effective Depth of Large Language Models?* arXiv:2512.14064.

- **Estudo:** mede quantas camadas contribuem de fato para a computação
  (profundidade efetiva) na família Qwen2.5 (1.5B a 32B), comparando modelos
  base e de raciocínio longo e tarefas de dificuldade diferente.
- **Resultados:** o número de camadas efetivas cresce com o tamanho, mas a
  **proporção fica estável**. Modelos maiores repetem o mesmo padrão de uso
  em mais camadas. Os modelos também não usam mais camadas em problemas mais
  difíceis, e o treino de raciocínio longo não aumenta a profundidade
  efetiva.
- **Relação com o nosso estudo:** é o contraponto direto à hipótese de
  profundidade. A nossa camada de **convergência** concorda com eles: 96–98%
  da profundidade no 7B, no 8B e no 72B. Só a **camada da escolha** (onde o
  dígito errado passa o certo) parece mais tardia no 72B (89% contra ~70%),
  mas com só 5–6 problemas por modelo. Se isso se confirmar na amostra
  completa, será um resultado em tensão com Hu et al.

### 6.2 Wendler et al. (2024). *Do Llamas Work in English? On the Latent Language of Multilingual Transformers.* ACL 2024.

- **Estudo:** logit lens no Llama-2 de 7B, 13B e 70B, em tarefas de tradução.
- **Resultados:** três fases consistentes em todos os tamanhos: alta
  entropia no início, "inglês latente" no meio e o idioma-alvo no fim. As
  fases aparecem nas **mesmas profundidades relativas**.
- **Relação com o nosso estudo:** mais um indício de que as etapas do
  processamento escalam em proporção ao número de camadas.

### 6.3 Halawi, Denain e Steinhardt (2023). *Overthinking the Truth: Understanding how Language Models Process False Demonstrations.* arXiv:2307.09476.

- **Estudo:** logit lens em modelos que recebem exemplos (few-shot) com
  rótulos corretos ou falsos.
- **Resultados:** até uma **camada crítica**, as duas condições se comportam
  igual. Depois dela, a predição com exemplos falsos vira para a errada, e
  as camadas finais pioram uma resposta que estava certa ("overthinking").
  Poucas "cabeças de indução falsas" explicam o efeito, confirmado por lesão.
- **Relação com o nosso estudo:** é o análogo mais próximo, em conceito, da
  nossa "camada da escolha" sob contexto enganoso, embora o contexto e a
  tarefa sejam outros e a escala não seja o foco.

### 6.4 Belrose et al. (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112.

- **Estudo:** propõe o tuned lens (uma sonda afim treinada por camada) e o
  compara ao logit lens em modelos de até 20B.
- **Resultados:** o logit lens é **enviesado** nas camadas intermediárias
  (favorece certos tokens e é mal calibrado), e em alguns modelos falha. O
  tuned lens é mais preditivo e menos enviesado.
- **Relação com o nosso estudo:** a comparação de profundidade entre modelos
  usa o logit lens, e o viés dele pode variar entre modelos. Para afirmar uma
  diferença de profundidade, convém confirmar com tuned lens ou, melhor, com
  *activation patching*.

### 6.5 *Representational Depth of Evaluation Awareness Shifts With Scale in Open-Weight Language Models.* arXiv:2606.29196 (2026). **Lido só o resumo.**

- **Estudo e resultados (segundo o resumo):** na família Qwen2.5 e no
  Gemma 2, a camada em que um conceito específico ("estou sendo avaliado") é
  mais legível **muda muito com a escala**: fica nas camadas finais dos
  modelos pequenos e nas iniciais dos grandes.
- **Relação com o nosso estudo:** a profundidade relativa pode mudar com a
  escala, e a direção depende do que se mede. Por isso a comparação precisa
  ser feita dentro da mesma família.

---

## 7. Onde o nosso estudo se posiciona

Nesta busca não encontramos um trabalho que combine:

1. frases **semanticamente neutras**, sem relação com o problema;
2. inserção **no meio de uma conta do raciocínio do próprio modelo**, seguida
   de continuação livre;
3. **controle explícito do ruído de lote** para cada modelo e cada cenário;
4. ativações medidas com **teacher forcing**, que isola o efeito da frase na
   leitura da resposta;
5. comparação da **camada da escolha** entre escalas **dentro da mesma
   família** (Qwen2.5: 7B, 14B, 32B, 72B; Qwen3: 8B, 14B, 32B), com outras
   arquiteturas como contraste (OLMo 3, Gemma 4, LFM2 híbrido
   convolução + atenção).

A busca foi feita em 26/09/2026 e não é exaustiva. Antes de afirmar
novidade, recomenda-se uma revisão sistemática.

**Hipóteses que a fila completa pode testar:**

- **H1** (Lanham et al.): a frase inserida no raciocínio pesa menos em
  modelos maiores, tanto no desvio de comportamento quanto no deslocamento
  do residual.
- **H2** (em tensão com Hu et al.): a camada da escolha sobe, em
  profundidade relativa, com o tamanho, dentro de cada família.
- **H3** (Niu et al.): o efeito de "I was counting 1+1." no raciocínio é
  maior que o das outras frases porque os tokens numéricos são copiados
  para a conta.

## Fontes

- Shi et al. 2023: <https://arxiv.org/abs/2302.00093> · <https://proceedings.mlr.press/v202/shi23a.html>
- Mirzadeh et al. 2025: <https://arxiv.org/pdf/2410.05229>
- Críticas ao GSM-Symbolic: <https://desirivanova.com/post/gsm-symbolic/> · <https://www.lesswrong.com/posts/Ze4C99Dasj74YKCFh/revisiting-gsm-symbolic-models-seem-to-reason-okay-actually>
- Gema et al. 2025: <https://arxiv.org/abs/2507.14417>
- Lanham et al. 2023: <https://arxiv.org/abs/2307.13702>
- Yuan et al. 2025: <https://arxiv.org/html/2506.09501v1>
- Thinking Machines Lab 2025: <https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/>
- Stolfo et al. 2023: <https://arxiv.org/abs/2305.15054>
- Nikankin et al. 2025: <https://arxiv.org/abs/2410.21272>
- Lad et al. 2025: <https://arxiv.org/pdf/2406.19384>
- Niu et al. 2025: <https://arxiv.org/pdf/2505.09338>
- arXiv 2609.17804: <https://arxiv.org/abs/2609.17804>
- arXiv 2606.24077: <https://arxiv.org/pdf/2606.24077>
- Hu et al. 2025: <https://arxiv.org/abs/2512.14064>
- Wendler et al. 2024: <https://arxiv.org/pdf/2402.10588>
- Halawi et al. 2023: <https://arxiv.org/abs/2307.09476>
- Belrose et al. 2023: <https://arxiv.org/abs/2303.08112>
- arXiv 2606.29196: <https://arxiv.org/pdf/2606.29196>
