"""
Day 11 实操：训练 YOLOv8 苹果检测器
====================================
用法:
    conda activate apple
    python scripts/day11_train.py --data configs/dataset.yaml --epochs 50 --imgsz 640
等价于 scripts/train_yolo.py；用 yolov8n.pt 预训练权重起步（迁移学习）。
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="train")
    args = parser.parse_args()

    model = YOLO(args.model)   # 预训练权重起步（迁移学习）
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, name=args.name, exist_ok=True)
    print("训练完成，最佳权重在 runs/detect/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()
