"""
benchmark.py — Latency and throughput profiler for all E2E planner architectures.

Constructs synthetic inputs matching each model's expected shapes, runs warmup
passes, then times N forward passes using CUDA events for GPU-accurate measurement.

Usage (run from the drive_e2e workspace root):
    python benchmark.py                        # all models, batch size 1
    python benchmark.py --models mlp transformer  # specific models
    python benchmark.py --batch_size 8 --n 200    # different batch size / iters
    python benchmark.py --device cpu              # CPU timing

Output:
    model              params     batch   latency (ms)     FPS      real-time?
    ─────────────────────────────────────────────────────────────────────────
    mlp                289K       1         0.3 ± 0.0     3333      YES (10Hz)
    ...
"""

import argparse
import time
import torch
import torch.nn as nn


# ── dataset constants (must match dataset.py) ─────────────────────────────────
PAST_STEPS   = 41
FUTURE_STEPS = 50
IMG_H, IMG_W = 224, 224
REALTIME_HZ  = 10          # Bench2Drive tick rate — need FPS >= this for real-time


# ── synthetic batch constructors ──────────────────────────────────────────────

def _kin_batch(B: int, device: torch.device) -> dict:
    """Kinematic-only batch (MLP, Transformer)."""
    return {
        "past_traj":    torch.randn(B, PAST_STEPS, 2,  device=device),
        "speed":        torch.randn(B, PAST_STEPS,     device=device),
        "acceleration": torch.randn(B, PAST_STEPS, 3,  device=device),
        "command":      torch.zeros(B, dtype=torch.long, device=device),
    }

def _cam_batch(B: int, n_cams: int, device: torch.device, load_depth: bool = False) -> dict:
    """Camera + kinematic batch (all vision models)."""
    batch = _kin_batch(B, device)
    batch["images"] = torch.randn(B, n_cams, 3, IMG_H, IMG_W, device=device)
    if load_depth:
        batch["depth"] = torch.randn(B, 1, 1, IMG_H, IMG_W, device=device)
    return batch


MODEL_CONFIGS = {
    "mlp": {
        "params": {"past_steps": PAST_STEPS, "future_steps": FUTURE_STEPS},
        "batch_fn": lambda B, dev: _kin_batch(B, dev),
        "module": "models.mlp_planner.MLPPlanner",
    },
    "transformer": {
        "params": {"past_steps": PAST_STEPS, "future_steps": FUTURE_STEPS},
        "batch_fn": lambda B, dev: _kin_batch(B, dev),
        "module": "models.transformer_planner.TransformerPlanner",
    },
    "resnet": {
        "params": {},
        "batch_fn": lambda B, dev: _cam_batch(B, 1, dev),
        "module": "models.resnet_planner.ResNetPlanner",
    },
    "front_cam": {
        "params": {},
        "batch_fn": lambda B, dev: _cam_batch(B, 1, dev),
        "module": "models.front_cam_planner.FrontCamPlanner",
    },
    "front_cam_depth": {
        "params": {},
        "batch_fn": lambda B, dev: _cam_batch(B, 1, dev, load_depth=True),
        "module": "models.front_cam_depth_planner.FrontCamDepthPlanner",
    },
    "vision_transformer": {
        "params": {"front_cam_only": False},
        "batch_fn": lambda B, dev: _cam_batch(B, 6, dev, load_depth=True),
        "module": "models.vision_transformer_planner.VisionTransformerPlanner",
    },
}

KNOWN_PARAMS = {
    "mlp":               289_000,
    "transformer":     1_400_000,
    "front_cam":      29_600_000,
    "resnet":         33_200_000,
    "front_cam_depth":35_700_000,
    "vision_transformer": 41_800_000,
}


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1e6:.1f}M"
    return f"{n/1e3:.0f}K"


def _import_model(dotted: str):
    module_path, cls_name = dotted.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def benchmark_model(name: str, cfg: dict, B: int, n: int, warmup: int,
                    device: torch.device) -> dict:
    # Build model
    ModelCls = _import_model(cfg["module"])
    model = ModelCls(**cfg["params"]).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    batch = cfg["batch_fn"](B, device)

    use_cuda = device.type == "cuda"

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(batch)
    if use_cuda:
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(n):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(batch)
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))   # ms
            else:
                t0 = time.perf_counter()
                _ = model(batch)
                latencies.append((time.perf_counter() - t0) * 1000)

    import statistics
    mean_ms = statistics.mean(latencies)
    std_ms  = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    fps     = 1000.0 / mean_ms * B      # account for batch size

    return {
        "name":    name,
        "params":  n_params,
        "batch":   B,
        "mean_ms": mean_ms,
        "std_ms":  std_ms,
        "fps":     fps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Models to benchmark (default: all)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Inference batch size (default: 1 for latency)")
    parser.add_argument("--n",      type=int, default=100, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=20,  help="Warmup iterations")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        hw_str = f"{props.name}  ({props.total_memory // 1024**3} GB VRAM)"
    else:
        import platform
        hw_str = f"CPU — {platform.processor()}"

    print(f"\nHardware : {hw_str}")
    print(f"Precision: fp32   |  Batch size: {args.batch_size}   |  Iters: {args.n} (warmup {args.warmup})")
    print(f"Real-time threshold: {REALTIME_HZ} Hz  ({1000/REALTIME_HZ:.0f} ms/frame)\n")

    col_w = [22, 8, 7, 18, 10, 13]
    header = (f"{'model':<{col_w[0]}} {'params':>{col_w[1]}} {'batch':>{col_w[2]}} "
              f"{'latency (ms)':>{col_w[3]}} {'FPS':>{col_w[4]}} {'real-time?':>{col_w[5]}}")
    sep = "─" * sum(col_w + [len(col_w) - 1])
    print(header)
    print(sep)

    results = []
    for name in args.models:
        cfg = MODEL_CONFIGS[name]
        try:
            r = benchmark_model(name, cfg, args.batch_size, args.n, args.warmup, device)
            results.append(r)
            rt = "YES" if r["fps"] >= REALTIME_HZ else "NO"
            lat_str = f"{r['mean_ms']:6.1f} ± {r['std_ms']:4.1f}"
            print(f"{name:<{col_w[0]}} {_fmt_params(r['params']):>{col_w[1]}} "
                  f"{r['batch']:>{col_w[2]}} {lat_str:>{col_w[3]}} "
                  f"{r['fps']:>{col_w[4]}.1f} {rt:>{col_w[5]}}")
        except Exception as e:
            print(f"{name:<{col_w[0]}} {'ERROR':>{col_w[1]}}  {e}")

    print(sep)
    print(f"\nNote: latency measured per forward pass. FPS = batch_size / mean_latency.")
    print(f"      Real-time = can sustain {REALTIME_HZ} Hz (Bench2Drive tick rate).\n")


if __name__ == "__main__":
    main()
