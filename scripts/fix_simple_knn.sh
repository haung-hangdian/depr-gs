#!/usr/bin/env bash
# One-shot repair for an empty or broken simple-knn checkout.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

# AutoDL's github turbo speeds up cloning; harmless elsewhere.
source /etc/network_turbo 2>/dev/null || true

SK_DIR=third_party/gaussian-splatting/submodules/simple-knn
# graphdeco-inria/simple-knn was removed from GitHub; camenduru keeps a mirror,
# and the INRIA GitLab repo is the canonical upstream.
MIRRORS=(
  "https://github.com/camenduru/simple-knn.git"
  "https://gh-proxy.com/https://github.com/camenduru/simple-knn.git"
  "https://ghfast.top/https://github.com/camenduru/simple-knn.git"
  "https://gitlab.inria.fr/bkerbl/simple-knn.git"
  "https://gitclone.com/github.com/camenduru/simple-knn.git"
)

if [ -f "$SK_DIR/setup.py" ]; then
  echo "[fix] checkout looks complete, rebuilding extension in place"
else
  echo "[fix] removing incomplete checkout at $SK_DIR"
  rm -rf "$SK_DIR"
  for m in "${MIRRORS[@]}"; do
    echo "[fix] cloning via $m"
    if git clone --depth 1 -c http.version=HTTP/1.1 "$m" "$SK_DIR"; then
      break
    fi
    rm -rf "$SK_DIR"
  done
  [ -f "$SK_DIR/setup.py" ] || {
    echo "[fix] all mirrors failed; rerun after: source /etc/network_turbo"
    exit 1
  }
fi

(cd "$SK_DIR" && python setup.py build_ext --inplace)

python -c "import torch, sys; sys.path.insert(0, 'third_party/gaussian-splatting/submodules/simple-knn'); from simple_knn._C import distCUDA2; print('simple-knn OK')"
