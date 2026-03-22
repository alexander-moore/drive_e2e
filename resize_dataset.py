"""
resize_dataset.py — Pre-resize bench2drive RGB camera images to 224x224.

Copies anno/ and camera/rgb_*/ from src to dst, resizing all JPEGs.
Non-RGB camera dirs (depth, semantic, instance, lidar, radar) are skipped
since the current multicam_video_resnet model doesn't use them.

Usage:
    python resize_dataset.py \
        --src /home/farm/data/bench2drive \
        --dst /home/farm/data/bench2drive_224 \
        --size 224 \
        --workers 16
"""

import argparse
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

RGB_CAMS = [
    "rgb_front", "rgb_front_left", "rgb_front_right",
    "rgb_back", "rgb_back_left", "rgb_back_right",
]


def process_scenario(args):
    src_scenario, dst_scenario, size = args
    src_scenario = Path(src_scenario)
    dst_scenario = Path(dst_scenario)

    # Copy anno dir as-is
    src_anno = src_scenario / "anno"
    dst_anno = dst_scenario / "anno"
    if src_anno.exists():
        shutil.copytree(src_anno, dst_anno, dirs_exist_ok=True)

    # Resize rgb camera images
    for cam in RGB_CAMS:
        src_cam = src_scenario / "camera" / cam
        dst_cam = dst_scenario / "camera" / cam
        if not src_cam.exists():
            continue
        dst_cam.mkdir(parents=True, exist_ok=True)
        for jpg in sorted(src_cam.glob("*.jpg")):
            dst_jpg = dst_cam / jpg.name
            if dst_jpg.exists():
                continue
            img = Image.open(jpg).convert("RGB")
            img = img.resize((size, size), Image.BILINEAR)
            img.save(dst_jpg, quality=95)

    return src_scenario.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",     default="/home/farm/data/bench2drive")
    parser.add_argument("--dst",     default="/home/farm/data/bench2drive_224")
    parser.add_argument("--size",    type=int, default=224)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    scenarios = sorted([
        d for d in src.iterdir()
        if d.is_dir() and not d.name.endswith(".tar.gz")
    ])
    print(f"Resizing {len(scenarios)} scenarios: {src} → {dst} @ {args.size}x{args.size}")
    print(f"Using {args.workers} workers\n")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_scenario, (str(s), str(dst / s.name), args.size)): s.name
            for s in scenarios
        }
        for fut in as_completed(futures):
            name = fut.result()
            done += 1
            if done % 50 == 0 or done == len(scenarios):
                print(f"  [{done}/{len(scenarios)}] {name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
