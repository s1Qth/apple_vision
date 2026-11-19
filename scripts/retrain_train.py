"""
retrain_train.py — 重训启动器（带调好的超参 + 续训 + 清缓存）(R5)
用法:
  python scripts/retrain_train.py --data configs/dataset.yaml --epochs 100 --imgsz 640
  python scripts/retrain_train.py --model yolov8s.pt --epochs 120        # 更大模型更抗误检(更慢)
  python scripts/retrain_train.py --resume runs/detect/retrain/weights/last.pt
  python scripts/retrain_train.py --smoke                                # 1 epoch/320 冒烟测试
说明:
  默认基于 yolov8n 预训练迁移学习。针对“样本少 + 要压误检”的推荐超参：
    epochs 100, patience 30(早停), imgsz 640, batch 16, lr0 0.001, cos_lr True,
    mosaic 1.0(在线增强), hsv/flip 默认开。
  注意：本机若没有独显，device 用 'cpu'（慢但能跑）；有显卡改 device='0'。
"""
import argparse, os, glob, yaml
from ultralytics import YOLO


def clear_cache(base):
    for c in glob.glob(os.path.join(base, "labels", "*.cache")) + \
            glob.glob(os.path.join(base, "**", "*.cache"), recursive=True):
        try:
            os.remove(c)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/dataset.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--name", default="retrain")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 1
        args.imgsz = 320
        args.batch = 8
        args.name = "smoke"

    if args.resume:
        print("续训:", args.resume)
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    cfg = yaml.safe_load(open(args.data, encoding="utf-8"))
    base = cfg["path"]
    clear_cache(base)
    print("即将训练:")
    print("  数据: %s" % os.path.abspath(args.data))
    print("  基座: %s  epochs=%d imgsz=%d batch=%d lr0=%s device=%s name=%s" % (
        args.model, args.epochs, args.imgsz, args.batch, args.lr0, args.device, args.name))
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        cos_lr=True,
        patience=30,
        name=args.name,
        pretrained=True,
        exist_ok=True,
        device=args.device,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        flipud=0.0, fliplr=0.5, mosaic=1.0,
    )
    print("✅ 训练完成。新权重: runs/detect/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()
