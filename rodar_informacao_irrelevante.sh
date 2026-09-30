#!/usr/bin/env bash
# Executa o estudo de informação irrelevante para um modelo, em segundo plano.
#
# Uso (na raiz do repositório):
#   bash rodar_informacao_irrelevante.sh <modelo> [--teste] [--lote N] [--gpu LISTA]
#                                        [--rotulo NOME] [--max-tokens N] [--primeiro-plano]
#
#   --teste           2% da amostra (6 problemas) em vez de 30% (90)
#   --lote N          tamanho do lote de geração (padrão 8)
#   --gpu LISTA       GPUs visíveis para esta execução, ex. 0, 1 ou 0,1
#                     (padrão: todas; o modelo é dividido entre elas se precisar)
#   --rotulo NOME     sufixo dos nomes da execução, para não misturar com uma
#                     execução anterior do mesmo modelo (ex. rtxpro6000)
#   --max-tokens N    limite de tokens gerados (padrão 4096; modelos de raciocínio
#                     longo, como o VibeThinker, precisam de mais)
#   --primeiro-plano  espera o fim da execução e sai com o código do nbconvert
#                     (usado pela fila, rodar_fila.sh)
#
# Exemplos (2 GPUs de 96 GB):
#   bash rodar_informacao_irrelevante.sh Qwen/Qwen3-32B --teste --gpu 0
#   bash rodar_informacao_irrelevante.sh Qwen/Qwen3-14B --gpu 0 --lote 16   # em paralelo com
#   bash rodar_informacao_irrelevante.sh Qwen/Qwen3-32B --gpu 1             # ... este
#   bash rodar_informacao_irrelevante.sh Qwen/Qwen2.5-72B-Instruct --gpu 0,1 --lote 4
#
# O processo continua rodando se a sessão SSH cair. Se ele for interrompido,
# rodar o mesmo comando de novo retoma de onde parou (as gerações são salvas
# lote a lote). Ao terminar, o código de saída fica em <nome>.exit, ao lado do log.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

MODEL="${1:?informe o modelo, ex.: Qwen/Qwen3-32B}"
shift
ARGS=("$@")
FRACTION=0.30
BATCH=8
GPUS="${CUDA_VISIBLE_DEVICES:-}"
TAG=""
MAX_TOKENS=4096
FOREGROUND=0
while [ $# -gt 0 ]; do
    case "$1" in
        --teste) FRACTION=0.02 ;;
        --lote) BATCH="$2"; shift ;;
        --gpu) GPUS="$2"; shift ;;
        --rotulo) TAG="$2"; shift ;;
        --max-tokens) MAX_TOKENS="$2"; shift ;;
        --primeiro-plano) FOREGROUND=1 ;;
        *) echo "opção desconhecida: $1" >&2; exit 2 ;;
    esac
    shift
done

NAME="$(basename "$MODEL")${TAG:+_$TAG}_frac${FRACTION}"
RUN_NB="notebooks/execucoes/informacao_irrelevante_${NAME}.ipynb"
LOG="results/irrelevant_info/${NAME}.log"
PIDFILE="results/irrelevant_info/${NAME}.pid"
EXITFILE="results/irrelevant_info/${NAME}.exit"
mkdir -p notebooks/execucoes results/irrelevant_info

if [ -f "$PIDFILE" ] && [ "$(cat "$PIDFILE")" != "$$" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Já existe uma execução para $NAME (PID $(cat "$PIDFILE")). Log: $LOG" >&2
    exit 3
fi

if [ "$FOREGROUND" = 0 ]; then
    # Relança este script em primeiro plano, numa sessão própria, desacoplado do terminal
    nohup setsid bash "$0" "$MODEL" "${ARGS[@]}" --primeiro-plano > /dev/null 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "Iniciado: $MODEL | fração $FRACTION | lote $BATCH | GPUs ${GPUS:-todas} | PID $(cat "$PIDFILE")"
    echo "  notebook: $RUN_NB"
    echo "  log:      $LOG"
    echo
    echo "Acompanhar:"
    echo "  kill -0 \$(cat $PIDFILE) && echo rodando || echo terminou"
    echo "  cat $EXITFILE   # código de saída, quando terminar (0 = sucesso)"
    echo "  nvidia-smi"
    exit 0
fi

# ── Primeiro plano: roda o nbconvert e registra o código de saída ─────────
echo $$ > "$PIDFILE"
rm -f "${EXITFILE:?}"
if [ -f "$LOG" ]; then
    mv "$LOG" "$LOG.anterior"      # guarda o log da tentativa anterior
fi
source .venv/bin/activate

# Cada execução usa sua própria cópia do notebook, que guarda as saídas daquele modelo
cp notebooks/informacao_irrelevante_ativacoes.ipynb "$RUN_NB"

if [ -n "$GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi
# Mesma numeração de GPUs que o nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID

set +e
IRR_MODEL="$MODEL" IRR_FRACTION="$FRACTION" IRR_BATCH="$BATCH" IRR_RUN_TAG="$TAG" \
    IRR_MAX_NEW_TOKENS="$MAX_TOKENS" \
    jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=counterfactual-grounding \
    --ExecutePreprocessor.timeout=-1 "$RUN_NB" > "$LOG" 2>&1 < /dev/null
CODE=$?
set -e
echo "$CODE" > "$EXITFILE"
echo "[rodar_informacao_irrelevante] nbconvert terminou com código $CODE em $(date '+%F %T')" >> "$LOG"
exit "$CODE"
