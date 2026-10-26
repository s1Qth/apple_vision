"""
Day 12 实操：评估 + 推理 YOLOv8 苹果检测器
==========================================
用法:
    conda activate apple
    python scripts/day12_eval.py --weights runs/detect/train/weights/best.pt --image data/raw/apple.jpg
"""
import argparse
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    parser.add_argument("--image", default="data/raw/apple.jpg")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    model = YOLO(args.weights)

    # 评估（在 dataset.yaml 的 val 集上）
    print("=== 评估 ===")
    metrics = model.val()
    print(f"mAP@0.5       = {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95  = {metrics.box.map:.3f}")

    # 单图推理
    print("=== 推理 ===")
    results = model.predict(args.image, conf=args.conf)
    out = "outputs/day12_pred.jpg"
    os.makedirs("outputs", exist_ok=True)
    for r in results:
        r.save(out)
    n = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"检测到 {n} 个目标，结果保存至 {out}")


if __name__ == "__main__":
    main()
