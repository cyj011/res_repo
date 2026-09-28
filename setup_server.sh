#!/usr/bin/env bash
set -Eeuo pipefail

# Install the project into an isolated virtual environment. RTX 5090 is a
# Blackwell GPU, so use a recent PyTorch CUDA 12.8 wheel with Blackwell support.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

echo "[1/5] Checking Python..."
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
}
"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 9):
    raise SystemExit("Python 3.9 or newer is required for the RTX 5090 PyTorch wheel")
print(f"Python {sys.version.split()[0]}")
PY

echo "[2/5] Creating virtual environment: ${VENV_DIR}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"
PIP=("${PYTHON}" -m pip)

echo "[3/5] Upgrading pip tooling..."
"${PIP[@]}" install --upgrade pip setuptools wheel

echo "[4/5] Installing PyTorch ${TORCH_VERSION} CUDA 12.8 wheels..."
"${PIP[@]}" install \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

echo "[5/5] Installing remaining project dependencies..."
"${PIP[@]}" install \
    -r <(grep -vE '^(torch|torchvision)([<=>].*)?$' "${ROOT_DIR}/requirements.txt")

echo
echo "Verifying installation..."
cd "${ROOT_DIR}"
"${PYTHON}" - <<'PY'
import torch
import torchvision
from approach.ResEmoteNet import ResEmoteNet

print(f"torch       : {torch.__version__}")
print(f"torchvision : {torchvision.__version__}")
print(f"CUDA build  : {torch.version.cuda}")
print(f"CUDA ready  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 0       : {torch.cuda.get_device_name(0)}")
    print(f"GPU capability: {torch.cuda.get_device_capability(0)}")
    # Exercise a real CUDA kernel so an unsupported wheel fails during setup.
    from approach.ResEmoteNet import ResEmoteNet
    model = ResEmoteNet().cuda().eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 64, 64, device="cuda"))
    print("CUDA model test: passed")
else:
    print("WARNING: CUDA is unavailable. Check the NVIDIA driver and PyTorch installation.")
print(f"parameters  : {sum(p.numel() for p in ResEmoteNet().parameters()):,}")
PY

echo
echo "Installation complete. Activate the environment with:"
echo "source ${VENV_DIR}/bin/activate"
echo "PyTorch wheel: torch==${TORCH_VERSION}, torchvision==${TORCHVISION_VERSION}, ${TORCH_INDEX_URL}"
echo "Then train on physical GPU 0 with:"
echo "CUDA_VISIBLE_DEVICES=0 ${PYTHON} ${ROOT_DIR}/train_files/ResEmoteNet_folder_train.py --data-root /mnt/data/yanyi2025/cyj/fer2013_img --gpu 0 --output-dir /mnt/data/yanyi2025/cyj/fer2013/resemotenet"
