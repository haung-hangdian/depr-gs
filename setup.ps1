$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Test-Path third_party)) { New-Item -ItemType Directory third_party | Out-Null }

git clone --depth 1 https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting
git clone --depth 1 https://github.com/facebookresearch/vggt-omega.git third_party/vggt-omega
git clone --depth 1 https://github.com/facebookresearch/sam-3d-objects.git third_party/sam-3d-objects
git clone --depth 1 https://github.com/facebookresearch/sam2.git third_party/sam2

conda create -n depr-gs python=3.10 -y
conda activate depr-gs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .

Set-Location third_party/gaussian-splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
Set-Location $Root

Set-Location third_party/vggt-omega
pip install -e .
Set-Location $Root

pip install -e third_party/sam2
pip install -e third_party/sam-3d-objects

Write-Host "Next: download checkpoints (HF gated):"
Write-Host "  hf download facebook/vggt-omega --include 'checkpoints/*' --exclude '*.onnx'"
Write-Host "  hf download facebook/sam-3d-objects --include 'checkpoints/*'"
Write-Host "  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
