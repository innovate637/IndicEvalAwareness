#!/bin/bash
#SBATCH --job-name=p2-probe128
#SBATCH --partition=h100-full
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --account=professors-limited
#SBATCH --qos=professor-fullgpu-limited
#SBATCH --output=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/logs/probe128_%j.out
set -uo pipefail
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
/localstorage/home/dhruvkumar/IndicEvalAwareness/.venv-p2/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    y = (x @ x).float().sum().item()
    print("bf16 matmul on GPU OK, checksum finite:", y == y)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU mem free {free/1024**3:.1f} GiB / total {total/1024**3:.1f} GiB")
PY
