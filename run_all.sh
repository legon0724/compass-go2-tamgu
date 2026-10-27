#!/bin/bash
# DGX 컨테이너 안에서 실행: bash run_all.sh
set -e
mkdir -p results
echo "=== 환경 확인 ==="
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), hasattr(torch, '_int_mm'))" | tee results/torch_env.txt
nvidia-smi | tee results/nvidia_smi.txt
pip list > results/pip_list.txt 2>/dev/null
date "+%Y-%m-%d %H:%M:%S %z" > results/run_time.txt
echo "=== 측정 시작 ==="
python run_dgx_benchmark.py 2>&1 | tee results/run_log.txt
echo "=== 끝. results 폴더를 USB로 옮기세요 ==="
ls -l results
