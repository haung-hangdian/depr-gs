#!/usr/bin/env bash
# DEPR-GS one-shot environment setup for Linux (tested path: AutoDL + PyTorch image).
#
# Usage:
#   bash setup.sh                 # full install
#   bash setup.sh --dry-run       # print what would run, execute nothing
#   bash setup.sh --self-test     # check toolchain only (git/python/pip/gcc/torch/nvcc)
#   bash setup.sh --no-sam2 --no-sam3d   # skip optional model packages
#
# Env overrides:
#   PIP_INDEX_URL  default https://pypi.tuna.tsinghua.edu.cn/simple
#   SKIP_VGGT=1    skip vggt-omega (only if you bring your own geometry)
#   HF_TOKEN       HuggingFace token for gated checkpoints (not downloaded here)
set -Eeuo pipefail

cd "$(dirname "$0")"

info() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup][warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[setup][error]\033[0m %s\n' "$*" >&2; exit 1; }

DRY_RUN=0 SELF_TEST=0 SKIP_CHECK=0 NO_SAM2=0 NO_SAM3D=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --self-test) SELF_TEST=1 ;;
    --skip-check) SKIP_CHECK=1 ;;
    --no-sam2) NO_SAM2=1 ;;
    --no-sam3d) NO_SAM3D=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) warn "unknown option ignored: $arg" ;;
  esac
done

PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYBIN="${PYBIN:-}"
if [ -z "$PYBIN" ]; then
  if command -v python >/dev/null 2>&1; then PYBIN=python
  elif command -v python3 >/dev/null 2>&1; then PYBIN=python3
  else fail "python not found; install a python 3.10+ environment first"
  fi
fi

run() { if [ "$DRY_RUN" -eq 1 ]; then echo "  [dry-run] $*"; else "$@"; fi; }

# pip always runs with proxies stripped: AutoDL's github turbo breaks PyPI access.
PIP() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u all_proxy -u ALL_PROXY "$PYBIN" -m pip --no-input "$@"
}

GIT_MIRRORS=(
  "https://github.com/"
  "https://gitclone.com/github.com/"
  "https://gh-proxy.com/https://github.com/"
  "https://ghfast.top/https://github.com/"
)

clone_repo() {
  local name=$1 path=$2 repo=$3 optional=${4:-0}
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  [dry-run] clone $repo -> $path"; return 0
  fi
  if [ -e "$path/.git" ]; then
    info "skip clone: $name already present"; return 0
  fi
  for mirror in "${GIT_MIRRORS[@]}"; do
    for attempt in 1 2; do
      info "clone $name (attempt $attempt via $mirror)"
      if git clone --depth 1 -c http.version=HTTP/1.1 "${mirror}${repo}" "$path"; then
        return 0
      fi
      rm -rf "$path"
      warn "clone $name failed, retrying"
    done
  done
  if [ "$optional" -eq 1 ]; then
    warn "optional repo $name skipped (clone failed)"
    return 0
  fi
  fail "unable to clone $name; on AutoDL try: source /etc/network_turbo && bash setup.sh"
}

self_test() {
  local miss=0
  check() {
    local label=$1; shift
    if "$@" >/dev/null 2>&1; then echo "  [ok]   $label"
    else echo "  [miss] $label"; miss=$((miss + 1)); fi
  }
  info "toolchain self-test"
  check git git --version
  check python "$PYBIN" --version
  check pip "$PYBIN" -m pip --version
  check gcc gcc --version
  check "torch (cuda)" "$PYBIN" -c "import torch; print(torch.__version__, torch.version.cuda)"
  check nvcc nvcc --version
  if [ "$SELF_TEST" -eq 1 ]; then
    echo "self-test done ($miss item(s) missing)"; exit 0
  fi
  if [ "$miss" -gt 0 ] && [ "$SKIP_CHECK" -ne 1 ] && [ "$DRY_RUN" -ne 1 ]; then
    fail "missing toolchain items above; fix them or rerun with --skip-check"
  fi
}

if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi
if [ -d /usr/local/cuda ] && [ -z "${CUDA_HOME:-}" ]; then
  export CUDA_HOME=/usr/local/cuda
fi

self_test

info "[1/5] clone upstream repositories"
clone_repo gaussian-splatting third_party/gaussian-splatting \
  graphdeco-inria/gaussian-splatting
clone_repo vggt-omega third_party/vggt-omega facebookresearch/vggt-omega \
  "$([ "${SKIP_VGGT:-0}" = 1 ] && echo 1 || echo 0)"
clone_repo vggt third_party/vggt facebookresearch/vggt 1
clone_repo sam-3d-objects third_party/sam-3d-objects \
  facebookresearch/sam-3d-objects 1
clone_repo sam2 third_party/sam2 facebookresearch/sam2 1

info "[2/5] fetching 3DGS CUDA submodules"
if [ -f third_party/gaussian-splatting/submodules/diff-gaussian-rasterization/setup.py ] \
   && [ -f third_party/gaussian-splatting/submodules/simple-knn/setup.py ]; then
  info "submodules already present"
else
  run git -C third_party/gaussian-splatting submodule update --init --recursive \
    || warn "submodule update failed; falling back to direct clones"
fi
clone_repo diff-gaussian-rasterization \
  third_party/gaussian-splatting/submodules/diff-gaussian-rasterization \
  graphdeco-inria/diff-gaussian-rasterization
clone_repo simple-knn \
  third_party/gaussian-splatting/submodules/simple-knn \
  graphdeco-inria/simple-knn

info "[3/5] installing python dependencies (container torch is kept untouched)"
if [ "$DRY_RUN" -eq 1 ]; then
  echo "  [dry-run] PIP install -U setuptools wheel"
  echo "  [dry-run] PIP install numpy pyyaml tqdm pillow scipy imageio matplotlib huggingface_hub einops"
  echo "  [dry-run] PIP install --no-deps lpips"
  echo "  [dry-run] PIP install -e . --no-build-isolation"
else
  PIP install --index-url "$PIP_INDEX_URL" -U setuptools wheel
  PIP install --index-url "$PIP_INDEX_URL" \
    numpy pyyaml tqdm pillow scipy imageio matplotlib huggingface_hub einops
  # --no-deps keeps pip from replacing the container's CUDA torch build
  PIP install --index-url "$PIP_INDEX_URL" --no-deps lpips \
    || warn "lpips skipped; PSNR/SSIM eval still works without it"
  PIP install -e . --use-pep517 --no-build-isolation
fi

info "[4/5] building 3DGS rasterizers (needs nvcc matching the container CUDA)"
if [ "$DRY_RUN" -eq 1 ]; then
  echo "  [dry-run] PIP install -e submodules/diff-gaussian-rasterization --use-pep517 --no-build-isolation"
  echo "  [dry-run] PIP install -e submodules/simple-knn --use-pep517 --no-build-isolation"
else
  PIP install -e third_party/gaussian-splatting/submodules/diff-gaussian-rasterization \
    --use-pep517 --no-build-isolation
  PIP install -e third_party/gaussian-splatting/submodules/simple-knn \
    --use-pep517 --no-build-isolation
fi

info "[5/5] installing model packages (failures here do not block training)"
if [ "$DRY_RUN" -eq 1 ]; then
  echo "  [dry-run] PIP install -e third_party/{vggt-omega,vggt,sam2,sam-3d-objects} --use-pep517 --no-build-isolation --no-deps"
else
  if [ "${SKIP_VGGT:-0}" -ne 1 ] && [ -e third_party/vggt-omega/.git ]; then
    PIP install -e third_party/vggt-omega --use-pep517 --no-build-isolation --no-deps \
      || warn "vggt-omega install failed; check its README for extra deps"
  fi
  if [ "${SKIP_VGGT:-0}" -ne 1 ] && [ -e third_party/vggt/.git ]; then
    PIP install -e third_party/vggt --use-pep517 --no-build-isolation --no-deps \
      || warn "vggt install failed; only needed if you use --model vggt"
  fi
  if [ "$NO_SAM2" -ne 1 ] && [ -e third_party/sam2/.git ]; then
    PIP install -e third_party/sam2 --use-pep517 --no-build-isolation --no-deps \
      || warn "sam2 install failed; masked eval needs it"
  fi
  if [ "$NO_SAM3D" -ne 1 ] && [ -e third_party/sam-3d-objects/.git ]; then
    PIP install -e third_party/sam-3d-objects --use-pep517 --no-build-isolation --no-deps \
      || warn "sam-3d-objects install failed; object-prior route is optional"
  fi
fi

info "verification"
if [ "$DRY_RUN" -eq 1 ]; then
  echo "  [dry-run] import checks + tests/test_gate.py"
else
  "$PYBIN" -c "from diff_gaussian_rasterization import GaussianRasterizationSettings; print('rasterizer OK')"
  "$PYBIN" -c "from simple_knn._C import distCUDA2; print('simple-knn OK')"
  "$PYBIN" tests/test_gate.py
fi

info "setup finished"
cat <<EOF
Next steps:
  # put your checkpoint at checkpoints/vggt_omega_1b_512.pt (no HF at runtime)
  python scripts/run_vggt_depth.py --images data/<scene>/images --out data/<scene>/vggt --ckpt checkpoints/vggt_omega_1b_512.pt
  # original VGGT weights: --model vggt --ckpt checkpoints/model.pt
  python scripts/vggt_to_colmap.py --vggt data/<scene>/vggt --out data/<scene>/sparse/0
  python train_depr.py -s data/<scene> -m output/<scene>_entropy --gate entropy --depth-path data/<scene>/vggt
EOF
