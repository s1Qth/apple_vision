"""
Day 14 实操：把检测到的苹果中心换算成 3D 坐标（P3 预览）
=========================================================
依赖: src/localization/pixel_to_3d.py, ultralytics
先用 --depth 给一个模拟深度跑通链路；拿到深度相机后改为读深度图。

用法:
    conda activate apple
    python scripts/day14_3d_preview.py \
        --weights runs/detect/train/weights/best.pt \
        --image data/raw/apple.jpg --depth 0.8
"""
import argparse
import os
import sys

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.localization.pixel_to_3d import pixel_to_camera_coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    parser.add_argument("--image", default="data/raw/apple.jpg")
    parser.add_argument("--depth", type=float, default=0.8, help="苹果深度(米)，模拟用")
    parser.add_argument("--intrinsics", default="configs/camera_intrinsics.yaml")
    args = parser.parse_args()

    # 简单解析相机内参（真实项目可用 yaml 库）
    fx = fy = cx = cy = None
    with open(args.intrinsics,encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("fx:"):
                fx = float(line.split()[1])
            elif line.startswith("fy:"):
                fy = float(line.split()[1])
            elif line.startswith("cx:"):
                cx = float(line.split()[1])
            elif line.startswith("cy:"):
                cy = float(line.split()[1])

    model = YOLO(args.weights)
    results = model.predict(args.image, conf=0.25)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        raise SystemExit("没检测到苹果，换个图或确认权重")

    # 取置信度最高的框
    best = r.boxes[0]
    x1, y1, x2, y2 = best.xyxy[0].tolist()
    u = (x1 + x2) / 2
    v = (y1 + y2) / 2

    point = pixel_to_camera_coords(u, v, args.depth, fx, fy, cx, cy)
    print(f"苹果 2D 中心像素: ({u:.1f}, {v:.1f})")
    print(f"苹果在相机坐标系下的 3D 位置: X={point[0]:.3f}  Y={point[1]:.3f}  Z={point[2]:.3f} m")


if __name__ == "__main__":
    main()
