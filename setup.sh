#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/4] cloning upstream repositories..."
mkdir -p third_party
[ -d third_party/gaussian-splatting ] || git clone --depth 1 https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting
[ -d third_party/vggt-omega ] || git clone --depth 1 https://github.com/facebookresearch/vggt-omega.git third_party/vggt-omega
[ -d third_party/sam-3d-objects ] || git clone --depth 1 https://github.com/facebookresearch/sam-3d-objects.git third_party/sam-3d-objects
[ -d third_party/sam2 ] || git clone --depth 1 https://github.com/facebookresearch/sam2.git third_party/sam2

echo "[2/4] installing python dependencies (torch excluded: keep the container's build)..."
python -m pip install numpy pyyaml tqdm pillow imageio matplotlib huggingface_hub einops lpips
python -m pip install -e .

echo "[3/4] building 3DGS rasterizers (needs nvcc matching the container CUDA)..."
python -m pip install -e third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
python -m pip install -e third_party/gaussian-splatting/submodules/simple-knn

echo "[4/4] installing model packages..."
python -m pip install -e third_party/vggt-omega
python -m pip install -e third_party/sam2 || echo "sam2 install failed (check its README)"
python -m pip install -e third_party/sam-3d-objects || echo "sam-3d-objects install failed; depth-gated pipeline still works"

echo ""
echo "setup done. Next steps:"
echo "  hf download facebook/vggt-omega --include 'checkpoints/*' --exclude '*.onnx'"
echo "  python tests/test_gate.py"
