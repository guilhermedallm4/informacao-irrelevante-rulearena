#!/usr/bin/env bash
# Prepara o ambiente do estudo numa máquina nova.
#
# Uso (na raiz do repositório):
#   bash setup_maquina.sh
#
# Variáveis opcionais:
#   PYTHON=python3.11      interpretador para criar a .venv (padrão: python3)
#   TORCH_INDEX=<url>      índice do PyTorch; o padrão é escolhido pela GPU:
#                          NVIDIA com driver CUDA >= 13 -> cu130 (o ambiente
#                          testado), senão cu128; AMD -> rocm6.4
#   TORCH_VERSION=2.13.0   versão do PyTorch (padrão: 2.13.0 no cu130, a testada;
#                          a mais recente nos outros índices). Vazio = mais recente
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PYTHON="${PYTHON:-python3}"
echo "== Python: $($PYTHON --version)"

if [ -z "${TORCH_INDEX:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_DRV=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+" | grep -oE "[0-9]+$" || echo 12)
        if [ "$CUDA_DRV" -ge 13 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu130"
            TORCH_VERSION="${TORCH_VERSION-2.13.0}"
        else
            TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        fi
        echo "== GPU NVIDIA detectada (driver suporta CUDA $CUDA_DRV):"
        nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    elif command -v rocm-smi >/dev/null 2>&1; then
        TORCH_INDEX="https://download.pytorch.org/whl/rocm6.4"
        echo "== GPU AMD detectada (ROCm):"
        rocm-smi --showproductname --showmeminfo vram || true
    else
        echo "ERRO: nenhuma GPU encontrada (nem nvidia-smi nem rocm-smi)." >&2
        exit 1
    fi
fi
TORCH_SPEC="torch${TORCH_VERSION:+==$TORCH_VERSION}"
echo "== PyTorch: $TORCH_SPEC de $TORCH_INDEX"

# 1. Ambiente virtual
if [ ! -d .venv ]; then
    "$PYTHON" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 2. PyTorch para esta GPU, depois o resto
pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"
pip install -r requirements-interp.txt

# 3. Benchmark RuleArena no commit fixado (já vem no pacote; clona se faltar)
if [ ! -d external/rulearena/airline ]; then
    git clone https://github.com/skyriver-2000/rulearena external/rulearena
    git -C external/rulearena checkout 59f60f6b4eb93a9b6d187ec781349d4f15618903
fi

# 4. Kernel do Jupyter usado pelos notebooks
python -m ipykernel install --user --name counterfactual-grounding \
    --display-name "Python (counterfactual_grounding .venv)"

# 5. Verificação
python - <<'EOF'
import torch, transformers
n = torch.cuda.device_count()
total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 1e9
print(f"torch {torch.__version__} | transformers {transformers.__version__}")
print(f"GPUs: {n} | VRAM total: {total:.0f} GB")
for i in range(n):
    print(f"  cuda:{i} {torch.cuda.get_device_name(i)}")
assert n > 0, "o PyTorch não enxerga nenhuma GPU"
EOF
python -m pytest tests -q

echo
echo "== Pronto. Próximos passos:"
echo "   source .venv/bin/activate"
echo "   hf auth login            # só é necessário para modelos restritos (ex.: Llama)"
echo "   bash rodar_informacao_irrelevante.sh Qwen/Qwen3-32B --teste --gpu 0   # 6 problemas, ~15 min"
