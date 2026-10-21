#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/4] cloning upstream repositories..."
mkdir -p third_party
clone_repo() {
  local url=$1 dir=$2
  if [ -d "$dir/.git" ]; then
    echo "skip $dir (already present)"
    return 0
  fi
  rm -rf "$dir"
  for attempt in 1 2 3; do
    echo "clone $dir (attempt $attempt)"
    if git clone --depth 1 -c http.version=HTTP/1.1 "$url" "$dir"; then
      return 0
    fi
    echo "clone failed, retrying..."
  done
  return 1
}
clone_repo https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting
clone_repo https://github.com/facebookresearch/vggt-omega.git third_party/vggt-omega
clone_repo https://github.com/facebookresearch/sam-3d-objects.git third_party/sam-3d-objects
clone_repo https://github.com/facebookresearch/sam2.git third_party/sam2

echo "[1b/4] fetching 3DGS CUDA submodules..."
if [ -f third_party/gaussian-splatting/submodules/diff-gaussian-rasterization/setup.py ]; then
  echo "submodules already present"
else
  git -C third_party/gaussian-splatting submodule update --init --recursive || true
fi
clone_repo https://github.com/graphdeco-inria/diff-gaussian-rasterization.git \
  third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
clone_repo https://github.com/graphdeco-inria/simple-knn.git \
  third_party/gaussian-splatting/submodules/simple-knn

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
