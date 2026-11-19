"""
retrain_eval.py — 评估：val 上 mAP + 人脸/负样本测试集上的误检计数 (R6)
用法:
  python scripts/retrain_eval.py --weights runs/detect/train/weights/best.pt --data configs/dataset.yaml
  python scripts/retrain_eval.py --weights new.pt --neg-test outputs/face_test --conf 0.5
说明:
  1) 在 val 集跑 model.val() 得 mAP；
  2) 在 --neg-test 文件夹（无苹果、空标签）上统计“误检数”= 检测到的框数
     （这些图本不应有任何框，检测到的就是误检/假苹果）。
     误检数越低越好，用来量化“脸还能不能被认成苹果”。
"""
import argparse, os, glob, json
import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="configs/dataset.yaml")
    ap.add_argument("--neg-test", default="outputs/face_test")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/retrain_eval.json")
    args = ap.parse_args()

    model = YOLO(args.weights)
    report = {"weights": os.path.abspath(args.weights)}

    # 1) val mAP
    try:
        metrics = model.val(data=args.data, verbose=False)
        report["val_map50"] = round(float(metrics.box.map50), 4)
        report["val_map5095"] = round(float(metrics.box.map), 4)
    except Exception as e:
        report["val_error"] = str(e)

    # 2) 负样本误检
    negs = sorted(glob.glob(os.path.join(args.neg_test, "*.*")))
    negs = [p for p in negs if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    fp = 0
    per = []
    for p in negs:
        im = cv2.imread(p)
        if im is None:
            continue
        res = model.predict(im, conf=args.conf, verbose=False)
        n = len(res[0].boxes) if res and len(res) else 0
        fp += n
        per.append((os.path.basename(p), n))
    report["neg_test_images"] = len(negs)
    report["neg_test_false_positives"] = fp
    report["neg_test_per_image"] = per

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("=" * 60)
    print("模型评估:", args.weights)
    if "val_map50" in report:
        print("  val mAP@0.5 = %s   mAP@0.5:0.95 = %s" % (
            report["val_map50"], report["val_map5095"]))
    print("  负样本测试集: %d 张, 误检(假苹果)总数 = %d" % (len(negs), fp))
    for name, n in per:
        if n > 0:
            print("    ⚠️ %s: %d 个误检" % (name, n))
    print("=" * 60)
    print("详细:", args.out)


if __name__ == "__main__":
    main()
