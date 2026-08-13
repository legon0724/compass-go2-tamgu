import argparse
import csv
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch


def synchronize():
    torch.cuda.synchronize()


def time_cuda(fn, inner_loops, repeats):
    samples = []
    for _ in range(repeats):
        # CUDA 이벤트로 GPU 실행 시간만 잰다.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner_loops):
            result = fn()
        end.record()
        synchronize()
        samples.append(start.elapsed_time(end) / inner_loops)
        del result
    return samples


def quantize_symmetric(x):
    scale = x.abs().amax() / 127.0
    if scale == 0:
        scale = torch.tensor(1.0, device=x.device)
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale


def nmae(reference, measured):
    denominator = reference.abs().mean().clamp_min(1e-12)
    return (reference.sub(measured).abs().mean() / denominator * 100).item()


def get_nvidia_smi():
    fields = [
        "name",
        "driver_version",
        "memory.total",
        "power.limit",
        "compute_cap",
    ]
    command = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, encoding="utf-8")
        return [line.strip() for line in output.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def loop_count(n):
    # 작은 행렬은 여러 번 돌려 오차를 줄인다.
    if n <= 1024:
        return 100
    if n <= 2048:
        return 30
    if n <= 4096:
        return 10
    return 3


def run_case(n, repeats, seed):
    torch.manual_seed(seed + n)
    a = torch.randn((n, n), device="cuda", dtype=torch.float32)
    b = torch.randn((n, n), device="cuda", dtype=torch.float32)
    qa, scale_a = quantize_symmetric(a)
    qb, scale_b = quantize_symmetric(b)
    loops = loop_count(n)

    # 첫 실행은 느릴 수 있어서 예열한다.
    for _ in range(3):
        _ = a @ b
        _ = torch._int_mm(qa, qb)
    synchronize()

    torch.backends.cuda.matmul.allow_tf32 = False
    fp32_samples = time_cuda(lambda: a @ b, loops, repeats)

    torch.backends.cuda.matmul.allow_tf32 = True
    tf32_samples = time_cuda(lambda: a @ b, loops, repeats)

    int8_samples = time_cuda(lambda: torch._int_mm(qa, qb), loops, repeats)

    # 양자화 시간은 행렬곱과 따로 측정한다.
    quant_samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        _qa, _sa = quantize_symmetric(a)
        _qb, _sb = quantize_symmetric(b)
        synchronize()
        quant_samples.append((time.perf_counter() - start) * 1000)
        del _qa, _qb, _sa, _sb

    torch.backends.cuda.matmul.allow_tf32 = False
    reference = a @ b
    int_result = torch._int_mm(qa, qb)
    dequantized = int_result.float() * (scale_a * scale_b)
    error = nmae(reference, dequantized)

    fp32_ms = statistics.median(fp32_samples)
    tf32_ms = statistics.median(tf32_samples)
    int8_ms = statistics.median(int8_samples)
    quant_ms = statistics.median(quant_samples)
    operations = 2 * n * n * n

    row = {
        "n": n,
        "inner_loops": loops,
        "repeats": repeats,
        "fp32_ms": fp32_ms,
        "tf32_ms": tf32_ms,
        "int8_int32_ms": int8_ms,
        "quantization_ms": quant_ms,
        "fp32_vs_int8_speedup": fp32_ms / int8_ms,
        "tf32_vs_int8_speedup": tf32_ms / int8_ms,
        "nmae_percent": error,
        "fp32_tflops": operations / (fp32_ms / 1000) / 1e12,
        "tf32_tflops": operations / (tf32_ms / 1000) / 1e12,
        "int8_tops": operations / (int8_ms / 1000) / 1e12,
        "input_memory_fp32_mib": 2 * n * n * 4 / 1024**2,
        "input_memory_int8_mib": 2 * n * n / 1024**2,
    }
    del a, b, qa, qb, reference, int_result, dequantized
    torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192]
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU를 찾지 못했습니다.")
    if not hasattr(torch, "_int_mm"):
        raise SystemExit("현재 PyTorch에서는 torch._int_mm을 사용할 수 없습니다.")

    args.output.mkdir(parents=True, exist_ok=True)
    environment = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "selected_gpu": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "nvidia_smi": get_nvidia_smi(),
        "seed": args.seed,
        "sizes": args.sizes,
        "repeats": args.repeats,
        "method": "symmetric per-tensor INT8, INT32 accumulation, CUDA Event median",
    }

    rows = []
    for n in args.sizes:
        print(f"N={n} 측정 중", flush=True)
        rows.append(run_case(n, args.repeats, args.seed))

    payload = {"environment": environment, "results": rows}
    json_path = args.output / "dgx_results.json"
    csv_path = args.output / "dgx_results.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"저장: {json_path}")
    print(f"저장: {csv_path}")


if __name__ == "__main__":
    main()
