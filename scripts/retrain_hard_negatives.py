"""
retrain_hard_negatives.py — 困难负样本挖掘（根治“脸/手被认成苹果”的关键）(R4)
用法:
  python scripts/retrain_hard_negatives.py --neg negative_photos/ --data configs/dataset.yaml --test-ratio 0.3
  python scripts/retrain_hard_negatives.py --neg negative_photos/ --data configs/dataset.yaml --copy   # 真正执行
说明:
  negative_photos/ 里是“没有苹果但容易误认”的图：人脸特写、手、红/黄球、其他水果、
  纯背景、树叶等。
  脚本把 (1-test-ratio) 比例注册进训练集（生成空标签 -> 模型学到“这是背景不是苹果”）；
  剩余 test-ratio 单独存到 outputs/face_test/ 作为评估集（不参与训练，用于 R6 量化误检）。
默认只“预览”，加 --copy 才真正执行。
"""
import argparse, os, glob, shutil, random
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg", required=True, help="负样本图文件夹（无苹果但易误认）")
    ap.add_argument("--data", default="configs/dataset.yaml")
    ap.add_argument("--test-ratio", type=float, default=0.3)
    ap.add_argument("--copy", action="store_true", help="默认只预览，加此参数才真正执行")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.data, encoding="utf-8"))
    base = cfg["path"]

    negs = sorted(glob.glob(os.path.join(args.neg, "*.*")))
    negs = [p for p in negs if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    random.seed(0)
    random.shuffle(negs)
    n_test = int(len(negs) * args.test_ratio)
    test_list = negs[:n_test]
    train_list = negs[n_test:]
    print("负样本: %d 张 -> 训练用 %d, 评估用 %d" % (len(negs), len(train_list), len(test_list)))

    if not args.copy:
        print("（预览模式，加 --copy 执行）")
        return

    for p in train_list:
        name = os.path.basename(p)
        shutil.copy(p, os.path.join(base, "images", "train", name))
        open(os.path.join(base, "labels", "train", os.path.splitext(name)[0] + ".txt"), "w").close()

    test_dir = "outputs/face_test"
    os.makedirs(test_dir, exist_ok=True)
    for p in test_list:
        name = os.path.basename(p)
        shutil.copy(p, os.path.join(test_dir, name))
        open(os.path.join(test_dir, os.path.splitext(name)[0] + ".txt"), "w").close()

    for c in glob.glob(os.path.join(base, "labels", "*.cache")):
        try:
            os.remove(c)
        except Exception:
            pass
    print("✅ 训练负样本已注册(空标签)。评估集: %s" % test_dir)


if __name__ == "__main__":
    main()
