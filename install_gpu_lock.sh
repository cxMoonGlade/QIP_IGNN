#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# torch+vision+audio +cu128: extra index
# pyg-lib, torch_scatter, ... +pt28cu128: PyG wheel page for this torch/CUDA combo
pip install -r "${ROOT}/requirements_lock.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
