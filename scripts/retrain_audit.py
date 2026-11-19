"""
retrain_audit.py — 数据体检：分析当前苹果数据集的短板（R1）
用法:
  python scripts/retrain_audit.py --data configs/dataset.yaml
  python scripts/retrain_audit.py --data configs/dataset.yaml --out outputs/retrain_audit.json
作用:
  - 统计 train/val 的图像数、标注文件数、框总数、类别分布
  - 统计每个框的归一化面积 / 像素宽高，看尺寸多样性
  - 找出「有图无标注」「有标注无图」的错位
  - 输出 JSON + 人类可读报告，并指出短板（数据太少/太单一/漏标）
说明:
  这是重训前的第一步，先搞清楚“差在哪”，才知道要补什么、补多少。
"""
import argparse, json, os, glob
import cv2
import yaml


def load_dataset(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg.get("path", "")
    train_img = cfg["train"]
    val_img = cfg["val"]
    train_dir = train_img if os.path.isabs(train_img) else os.path.join(base, train_img)
    val_dir = val_img if os.path.isabs(val_img) else os.path.join(base, val_img)
    names = cfg.get("names", {0: "apple"})
    nc = cfg.get("nc", len(names))
    return base, train_dir, val_dir, names, nc


def label_dir_of(img_dir):
    # .../images/train -> .../labels/train
    return img_dir.replace("images", "labels")


def analyze_split(img_dir, names):
    label_dir = label_dir_of(img_dir)
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    imgs = [p for p in imgs if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    labels = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    label_by_stem = {os.path.splitext(os.path.basename(l))[0]: l for l in labels}
    img_stems = [os.path.splitext(os.path.basename(p))[0] for p in imgs]
    no_label = [s for s in img_stems if s not in label_by_stem]
    no_img = [s for s in label_by_stem if s not in set(img_stems)]

    class_counts = {}
    box_areas = []
    box_wh = []
    img_sizes = []
    total_boxes = 0
    for l in labels:
        try:
            lines = open(l, "r", encoding="utf-8").read().strip().splitlines()
        except Exception:
            lines = []
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
                x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except Exception:
                continue
            total_boxes += 1
            class_counts[cid] = class_counts.get(cid, 0) + 1
            box_areas.append(w * h)

    for p in imgs:
        im = cv2.imread(p)
        if im is None:
            continue
        img_sizes.append((im.shape[1], im.shape[0]))

    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        lb = label_by_stem.get(stem)
        if not lb:
            continue
        im = cv2.imread(p)
        if im is None:
            continue
        h0, w0 = im.shape[:2]
        for ln in open(lb, encoding="utf-8").read().strip().splitlines():
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                w = float(parts[3])
                hh = float(parts[4])
            except Exception:
                continue
            box_wh.append((w * w0, hh * h0))

    return {
        "n_images": len(imgs),
        "n_labels": len(labels),
        "n_boxes": total_boxes,
        "class_counts": class_counts,
        "no_label_imgs": len(no_label),
        "no_img_labels": len(no_img),
        "img_sizes_unique": sorted(set(img_sizes)),
        "box_area_norm_mean": round(sum(box_areas) / len(box_areas), 4) if box_areas else 0,
        "box_area_norm_min": round(min(box_areas), 4) if box_areas else 0,
        "box_area_norm_max": round(max(box_areas), 4) if box_areas else 0,
        "box_px_w_mean": round(sum(w for w, _ in box_wh) / len(box_wh), 1) if box_wh else 0,
        "box_px_h_mean": round(sum(h for _, h in box_wh) / len(box_wh), 1) if box_wh else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/dataset.yaml")
    ap.add_argument("--out", default="outputs/retrain_audit.json")
    args = ap.parse_args()
    base, train_dir, val_dir, names, nc = load_dataset(args.data)
    train_stats = analyze_split(train_dir, names)
    val_stats = analyze_split(val_dir, names)
    report = {
        "dataset_yaml": os.path.abspath(args.data),
        "base": base,
        "names": names,
        "nc": nc,
        "train": train_stats,
        "val": val_stats,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("苹果数据集体检报告 (R1)")
    print("=" * 60)
    print("类别: %s  (nc=%s)" % (names, nc))
    for split, sname in [(train_stats, "TRAIN"), (val_stats, "VAL")]:
        print("\n--- %s ---" % sname)
        print("  图像数: %d" % split["n_images"])
        print("  标注文件: %d   框总数: %d" % (split["n_labels"], split["n_boxes"]))
        print("  类别分布: %s" % split["class_counts"])
        print("  有图无标注(漏标风险): %d" % split["no_label_imgs"])
        print("  有标注无图(孤儿标签): %d" % split["no_img_labels"])
        print("  框归一化面积 均值/最小/最大: %s/%s/%s" % (
            split["box_area_norm_mean"], split["box_area_norm_min"], split["box_area_norm_max"]))
        print("  框像素尺寸 宽均值/高均值: %s/%s" % (split["box_px_w_mean"], split["box_px_h_mean"]))
        print("  图像尺寸种类(前5): %s" % split["img_sizes_unique"][:5])
    print("\n" + "=" * 60)
    total_imgs = train_stats["n_images"] + val_stats["n_images"]
    print("总图像数: %d  (建议补到 150~300 张再重训)" % total_imgs)
    if total_imgs < 100:
        print("⚠️ 数据偏少 -> 正样本不足是误检/漏检主因之一")
    if train_stats["no_label_imgs"] > 0:
        print("⚠️ 存在有图无标注 -> 先补标或删除，否则训练会把苹果当背景学偏")
    print("=" * 60)
    print("详细 JSON: %s" % args.out)


if __name__ == "__main__":
    main()
