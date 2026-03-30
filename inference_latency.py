"""
benchmark.py
============
Measures inference latency and peak VRAM for:
  - IOHMambaPredictor  : single recurrent step  → input (1, 4)
  - IOHPredictor       : full window recompute  → input (1, 900, 4)

Run with:
    python benchmark.py

Requires CUDA. Will raise clearly if CUDA is unavailable.
"""

import time
import torch

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
WARMUP_ITERS   = 50
MEASURE_ITERS  = 1000
BATCH_SIZE     = 1 
SEQ_LEN        = 500 
INPUT_DIM      = 4
STATIC_DIM     = 5


def reset_vram_stats(device):
    """Zero out the peak-memory counter so measurements are isolated."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


def peak_vram_mb(device):
    """Return peak allocated VRAM in MB since the last reset."""
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def measure_latency(fn, warmup, iters, device):
    """
    Run fn() warmup times (discarded), then time iters calls.
    Returns mean latency in milliseconds.
    """
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    return elapsed / iters * 1000   # ms per call


def bench_transformer(device):
    model = IOHPredictor(
        input_dim=INPUT_DIM,
        model_dim_1=32,
        model_dim_2=64,
        num_heads=4
    ).to(device).eval()

    dummy_seq    = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM, device=device)
    dummy_static = torch.randn(BATCH_SIZE, STATIC_DIM,         device=device)

    def fn():
        with torch.no_grad():
            _ = model(dummy_seq, dummy_static)

    reset_vram_stats(device)
    latency_ms = measure_latency(fn, WARMUP_ITERS, MEASURE_ITERS, device)
    vram_mb    = peak_vram_mb(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return latency_ms, vram_mb, params


def bench_mamba(device):
    model = IOHMambaPredictor(
        input_dim=INPUT_DIM,
        model_dim_1=32,
        model_dim_2=64
    ).to(device).eval()

    dummy_step   = torch.randn(BATCH_SIZE, INPUT_DIM,  device=device)
    dummy_static = torch.randn(BATCH_SIZE, STATIC_DIM, device=device)

    state_1, state_2 = model.init_states(BATCH_SIZE, device)

    def fn():
        nonlocal state_1, state_2
        with torch.no_grad():
            _, state_1, state_2 = model.step(
                dummy_step, dummy_static, state_1, state_2
            )

    reset_vram_stats(device)
    latency_ms = measure_latency(fn, WARMUP_ITERS, MEASURE_ITERS, device)
    vram_mb    = peak_vram_mb(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return latency_ms, vram_mb, params


def main():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gpu_name = torch.cuda.get_device_name(device)
    budget_ms = 2000.0 

    print("=" * 65)
    print("  IOH Model Inference Benchmark")
    print(f"  GPU            : {gpu_name}")
    print(f"  Batch size     : {BATCH_SIZE}  (deployment: one patient)")
    print(f"  Warmup iters   : {WARMUP_ITERS}")
    print(f"  Measure iters  : {MEASURE_ITERS}")
    print(f"  Real-time budget (0.5 Hz): {budget_ms:.0f} ms per step")
    print("=" * 65)

    print("\n[1/2] Benchmarking Transformer (full window, input [1, 900, 4]) …")
    t_lat, t_vram, t_params = bench_transformer(device)
    t_headroom = budget_ms - t_lat

    print(f"  Parameters         : {t_params:,}")
    print(f"  Inference latency  : {t_lat:.3f} ms")
    print(f"  Peak VRAM          : {t_vram:.2f} MB")
    print(f"  Headroom vs budget : {t_headroom:.1f} ms")

    print("\n[2/2] Benchmarking Mamba (recurrent step, input [1, 4]) …")
    m_lat, m_vram, m_params = bench_mamba(device)
    m_headroom = budget_ms - m_lat

    print(f"  Parameters         : {m_params:,}")
    print(f"  Inference latency  : {m_lat:.3f} ms")
    print(f"  Peak VRAM          : {m_vram:.2f} MB")
    print(f"  Headroom vs budget : {m_headroom:.1f} ms")

    speedup       = t_lat  / m_lat
    vram_reduction = (1 - m_vram / t_vram) * 100

    print(f"  {'Metric':<30} {'Transformer':>12} {'Mamba':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Parameters':<30} {t_params:>12,} {m_params:>12,}")
    print(f"  {'Inference latency (ms)':<30} {t_lat:>12.3f} {m_lat:>12.3f}")
    print(f"  {'Peak VRAM (MB)':<30} {t_vram:>12.2f} {m_vram:>12.2f}")
    print(f"  {'Headroom at 0.5 Hz (ms)':<30} {t_headroom:>12.1f} {m_headroom:>12.1f}")
    print(f"  {'-'*54}")
    print(f"  Mamba latency speedup      : {speedup:.1f}×")
    print(f"  Mamba VRAM reduction       : {vram_reduction:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()