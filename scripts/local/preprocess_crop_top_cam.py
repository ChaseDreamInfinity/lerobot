"""Crop the top camera in a LeRobotDataset and write a new dataset.

Wrist camera is copied unchanged. Top frames are cropped to a fixed
(x, y, w, h) window. The new dataset is a sibling repo so the original
is preserved for clean ablation.

Usage:
    uv run python scripts/local/preprocess_crop_top_cam.py \\
        --src=chasedreaminf/record-pick-block-in-cup-2cam-run1_20260505_191315 \\
        --dst=chasedreaminf/record-pick-block-in-cup-2cam-topcrop_20260512 \\
        --crop="288,0,352,352" \\
        --push

Note: at inference time, the live top camera frame must be cropped to the
same (x, y, w, h) before being fed to the policy.
"""

from __future__ import annotations

import argparse
import copy

import torch
import torchvision.transforms.v2.functional as TVF

from lerobot.datasets.lerobot_dataset import LeRobotDataset


TOP_KEY = "observation.images.top"
META_KEYS_TO_STRIP = {
    "episode_index",
    "frame_index",
    "timestamp",
    "index",
    "task_index",
}


def parse_crop(s: str) -> tuple[int, int, int, int]:
    parts = [int(p) for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--crop must be 'x,y,w,h', got {s!r}")
    return tuple(parts)  # type: ignore[return-value]


def build_dst_features(src_features: dict, w: int, h: int) -> dict:
    """Update the top-camera feature shape; everything else carries over."""
    dst = copy.deepcopy(src_features)
    if TOP_KEY not in dst:
        raise KeyError(f"Source dataset has no '{TOP_KEY}' feature.")
    top = dst[TOP_KEY]
    if "shape" not in top:
        raise KeyError(f"'{TOP_KEY}' has no 'shape' in its feature dict.")
    shape = list(top["shape"])
    if len(shape) != 3:
        raise ValueError(f"Expected 3D shape [C,H,W] for top, got {shape}.")
    c = shape[0]
    top["shape"] = [c, h, w]
    return dst


def crop_image_chw(img: torch.Tensor, x: int, y: int, w: int, h: int) -> torch.Tensor:
    """Crop a [C, H, W] tensor. torchvision uses (top, left, height, width)."""
    return TVF.crop(img, top=y, left=x, height=h, width=w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source dataset repo_id")
    p.add_argument("--dst", required=True, help="Destination dataset repo_id")
    p.add_argument(
        "--crop",
        default="288,0,352,352",
        help="Top-cam crop as 'x,y,w,h' in original-frame coords. Default 288,0,352,352.",
    )
    p.add_argument("--root", default=None, help="Local dst root (default: $HF_LEROBOT_HOME/{dst}).")
    p.add_argument("--push", action="store_true", help="Push to HF after writing.")
    p.add_argument("--private", action="store_true", help="Push as private (default: public).")
    p.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Optional cap on episodes (smoke test).",
    )
    args = p.parse_args()

    x, y, w, h = parse_crop(args.crop)
    print(f"Crop: x={x}, y={y}, w={w}, h={h}")

    print(f"Loading source: {args.src}")
    src = LeRobotDataset(args.src)
    src_features = src.meta.features
    total_eps = src.meta.total_episodes
    total_frames = len(src)
    print(f"  {total_eps} episodes, {total_frames} frames, fps={src.fps}")
    print(f"  top original shape: {src_features[TOP_KEY]['shape']}")

    dst_features = build_dst_features(src_features, w=w, h=h)
    print(f"  top new shape:      {dst_features[TOP_KEY]['shape']}")

    print(f"Creating destination: {args.dst}")
    dst = LeRobotDataset.create(
        repo_id=args.dst,
        fps=src.fps,
        features=dst_features,
        root=args.root,
        robot_type=src.meta.robot_type,
        use_videos=True,
        vcodec="libsvtav1",
    )

    cur_ep_idx: int | None = None
    frames_in_ep = 0
    eps_saved = 0
    print(f"Processing frames...")
    for global_idx in range(total_frames):
        item = src[global_idx]
        ep_idx = int(item["episode_index"])

        if args.limit_episodes is not None and ep_idx >= args.limit_episodes:
            break

        if cur_ep_idx is not None and ep_idx != cur_ep_idx:
            print(f"  episode {cur_ep_idx}: {frames_in_ep} frames")
            dst.save_episode()
            eps_saved += 1
            frames_in_ep = 0

        cur_ep_idx = ep_idx

        frame: dict = {}
        for k, v in item.items():
            if k in META_KEYS_TO_STRIP:
                continue
            frame[k] = v

        top_img = frame[TOP_KEY]
        if not isinstance(top_img, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for '{TOP_KEY}', got {type(top_img).__name__}")
        frame[TOP_KEY] = crop_image_chw(top_img, x=x, y=y, w=w, h=h)

        dst.add_frame(frame)
        frames_in_ep += 1

    if cur_ep_idx is not None and frames_in_ep > 0:
        print(f"  episode {cur_ep_idx}: {frames_in_ep} frames")
        dst.save_episode()
        eps_saved += 1

    print(f"Saved {eps_saved} episodes.")

    if hasattr(dst, "finalize"):
        print("Finalizing...")
        dst.finalize()

    if args.push:
        print(f"Pushing to HF Hub: {args.dst} (private={args.private})")
        dst.push_to_hub(private=args.private)

    print("Done.")


if __name__ == "__main__":
    main()
