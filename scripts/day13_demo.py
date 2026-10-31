"""
Day 13 实操：一键演示 —— 对一批图跑模型，输出带框结果 + 汇总
===============================================================
用法:
    conda activate apple
    python scripts/day13_demo.py --weights runs/detect/train/weights/best.pt --src data/raw
"""
import argparse
import glob
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    parser.add_argument("--src", default="data/raw")
    parser.add_argument("--out", default="outputs/demo")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    model = YOLO(args.weights)
    os.makedirs(args.out, exist_ok=True)
    imgs = glob.glob(os.path.join(args.src, "*.*"))
    total = 0
    for p in imgs:
        if not p.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        results = model.predict(p, conf=args.conf)
        r = results[0]
        n = len(r.boxes) if r.boxes is not None else 0
        total += n
        name = os.path.basename(p)
        r.save(os.path.join(args.out, name))
        print(f"{name}: {n} 个苹果")
    print(f"\n共处理 {len(imgs)} 个文件，检测到苹果合计 {total} 个，结果在 {args.out}")


if __name__ == "__main__":
    main()
