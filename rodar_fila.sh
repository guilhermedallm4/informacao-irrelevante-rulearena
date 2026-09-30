#!/usr/bin/env bash
# Roda a fila de modelos de fila_modelos.yaml, um modelo por GPU, os de 2 GPUs no fim.
#
# Uso (na raiz do repositório):
#   bash rodar_fila.sh [--teste] [--fila ARQUIVO] [--refazer-falhas]   # inicia em segundo plano
#   bash rodar_fila.sh --listar [--teste] [--fila ARQUIVO]              # mostra a fila, sem executar
#   bash rodar_fila.sh --parar  [--teste]                               # encerra o escalonador e as execuções
#
#   --teste           6 problemas por modelo; estado e log separados (fila_estado_teste.json)
#   --fila ARQUIVO    outra fila (padrão: fila_modelos.yaml)
#   --refazer-falhas  devolve à fila os modelos marcados como "falhou"
#
# O escalonador continua se a sessão SSH cair. Pará-lo (kill ou queda) não
# interrompe as execuções em andamento; rodar este script de novo retoma a fila:
# acompanha o que ainda está rodando, pula o que terminou e retoma o resto.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source .venv/bin/activate

SUFFIX=""
for a in "$@"; do
    if [ "$a" = "--teste" ]; then SUFFIX="_teste"; fi
done
for a in "$@"; do
    case "$a" in
        --listar|--parar|-h|--help) exec python -m src.interp.fila "$@" ;;
    esac
done

LOG="results/irrelevant_info/fila${SUFFIX}.log"
PIDFILE="results/irrelevant_info/fila${SUFFIX}.pid"
mkdir -p results/irrelevant_info

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null \
        && grep -q "src.interp.fila" "/proc/$(cat "$PIDFILE")/cmdline" 2>/dev/null; then
    echo "O escalonador já está rodando (PID $(cat "$PIDFILE")). Log: $LOG" >&2
    exit 1
fi

nohup setsid python -m src.interp.fila "$@" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "Escalonador iniciado (PID $(cat "$PIDFILE"))"
echo "  log:     $LOG"
echo "  estado:  results/irrelevant_info/fila_estado${SUFFIX}.json"
echo
echo "Acompanhar:  bash rodar_fila.sh --listar ${SUFFIX:+--teste}   (ou o notebook notebooks/acompanhamento.ipynb)"
echo "             tail -f $LOG"
echo "Parar tudo:  bash rodar_fila.sh --parar ${SUFFIX:+--teste}"
