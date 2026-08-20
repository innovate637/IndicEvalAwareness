#!/bin/bash
#SBATCH --job-name=p2-probe
#SBATCH --partition=h100-full
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --account=professors-limited
#SBATCH --qos=professor-fullgpu-limited
#SBATCH --output=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/logs/probe_%j.out
set -uo pipefail
echo "== nvidia-smi =="
nvidia-smi
echo "== torch =="
source /localstorage/home/dhruvkumar/miniforge3/etc/profile.d/conda.sh
conda activate vllm
python - <<'PY'
import torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
try: print(torch.cuda.get_device_name(0))
except Exception as e: print("ERROR:", type(e).__name__, str(e)[:200])
PY
