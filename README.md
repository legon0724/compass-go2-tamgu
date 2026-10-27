# CoMPaSS 고2 탐구: DGX Spark에서 INT8 양자화 행렬곱 측정

양자화 기반 저비용 AI 가속기의 개발도상국 보급 전략 탐구에서 사용한 측정 코드와 원시 결과.

## 파일
| 경로 | 내용 |
|---|---|
| `run_dgx_benchmark.py` | 측정 코드. FP32(TF32 끔), TF32, INT8→INT32(`torch._int_mm`) 행렬곱을 CUDA Event로 7회 측정해 중앙값 저장 |
| `run_all.sh` | 환경 기록(nvidia-smi, PyTorch/CUDA 버전, 패키지 목록, 실행 시각) 후 측정 실행, 전체 출력 저장 |
| `results/` | 2026-10-26 19:52(KST) DGX Spark(GB10) 측정 원시 결과와 로그 |
| `capture/` | 실행 터미널 출력 캡처, 캡처 방식 메모, SHA-256 해시, 종료 코드 |
| `make_report_figures.py` | 보고서 그림 1~5를 JSON에서 직접 생성 |
| `data/cpu_results_20260812.json` | 비교용 CPU 측정 원시 결과 사본(원본: grade2-club-individual-research) |
| `figures/` | 생성된 그림 |

## 재현
```
docker run --gpus all --rm -v "$PWD":/workspace -w /workspace nvcr.io/nvidia/pytorch:25.09-py3 bash run_all.sh
python make_report_figures.py
```

## 측정 환경
NVIDIA DGX Spark(GB10, compute capability 12.1), NGC PyTorch 25.09(2.9.0a0+50eac811a6), CUDA 13.0, 드라이버 580.173.02, 시드 20260813+N.
실행은 원격 접속으로 수행했고 터미널 출력은 자동 캡처로 저장했다(`capture/CAPTURE_NOTES.txt`).

## 주요 결과 (N=8192, ms)
| FP32 | TF32 | INT8 커널 | 입력 양자화 | INT8 합계 | NMAE |
|---:|---:|---:|---:|---:|---:|
| 56.01 | 27.78 | 18.81 | 23.20 | 42.02 | 1.86% |

INT8 커널은 FP32보다 2.98배, TF32보다 1.48배 빨랐지만, 입력 양자화를 포함하면 FP32 대비 1.33배, TF32 대비 0.66배였다.

## 기록 정리
- 2026년 8월 1차 측정의 원시 파일은 장비 초기화로 남지 않았다. 원시 파일로 검증할 수 없는 값은 쓰지 않기로 하고, 같은 `run_dgx_benchmark.py`로 10월 26일에 다시 측정했다.
- 8월 커밋의 `make_dgx_chart.py`는 수치를 코드에 직접 적은 그래프 코드여서 삭제하고, JSON을 읽는 `make_report_figures.py`로 바꿨다.
