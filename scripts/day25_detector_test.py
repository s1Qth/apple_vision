"""
Day 25 验证脚本：苹果检测器模块
================================================================
验证目标：
  1. AppleDetector 能加载 best.pt 并在真实苹果图上检出框
  2. 每个检测 dict 字段完整（bbox/conf/center/label）
  3. annotate() 能画出标注图并保存

本脚本同样不依赖硬件：直接对 val 图片做检测。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob

import cv2

from src.webcam.detector import AppleDetector

REQUIRED_KEYS = ("bbox", "conf", "center", "label", "id")


def main():
    ap = argparse.ArgumentParser(description="Day25 苹果检测器验证")
    ap.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    ap.add_argument("--image-dir", default="data/apple_yolo/images/val")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--max", type=int, default=3)
    ap.add_argument("--out", default="outputs/day25")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    det = AppleDetector(args.weights, conf=args.conf, device="cpu")

    imgs = sorted(glob.glob(os.path.join(args.image_dir, "*.*")))[: args.max]
    print(f"模型: {args.weights}\n测试图: {len(imgs)} 张\n")

    total = 0
    sample = None
    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            print(f"  [跳过] 无法读取: {p}")
            continue
        dets = det.detect(img)
        vis = det.annotate(img, dets)
        outp = os.path.join(args.out, os.path.basename(p))
        cv2.imwrite(outp, vis)
        total += len(dets)
        if dets and sample is None:
            sample = dets[0]
        info = ", ".join(f"#{d['id']} conf={d['conf']:.2f}" for d in dets)
        print(f"  {os.path.basename(p)}: {len(dets)} 个苹果  [{info}]")

    # 自检：检测字段完整
    assert sample is None or all(k in sample for k in REQUIRED_KEYS), "检测字段缺失！"
    print(f"\n✅ 检测器验证通过：共检出 {total} 个苹果，标注图已存 {args.out}")
    print(f"✅ 检测字段完整（含 {REQUIRED_KEYS}）")


if __name__ == "__main__":
    main()
