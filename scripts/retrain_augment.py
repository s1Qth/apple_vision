"""
retrain_augment.py — 离线数据增强（扩充正样本，缓解过拟合/误检）(R3)
用法:
  python scripts/retrain_augment.py --data configs/dataset.yaml --split train --per-image 3
说明:
  对 train 图像做：水平翻转 / 亮度抖动 / 高斯模糊 / 加噪 / 小幅旋转(±12°,框同步变换)。
  生成的新图与标签写入 images/train 与 labels/train（文件名加 _augN 后缀）。
注意:
  YOLO 训练时本身也会做在线增强(mosaic/hsv/flip)。本脚本用于“增加原始样本量”，
  特别适合像你现在这样样本极少(56张)的情况。不要过度增强，per-image 2~4 即可。
"""
import argparse, os, glob, cv2, numpy as np, yaml, random


def load_base(data_yaml):
    cfg = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    return cfg["path"]


def read_boxes(txt):
    boxes = []
    if not os.path.exists(txt):
        return boxes
    for ln in open(txt, encoding="utf-8").read().strip().splitlines():
        if not ln.strip():
            continue
        p = ln.split()
        if len(p) < 5:
            continue
        try:
            boxes.append([int(p[0])] + [float(x) for x in p[1:5]])
        except Exception:
            continue
    return boxes


def write_boxes(txt, boxes):
    with open(txt, "w", encoding="utf-8") as f:
        for b in boxes:
            f.write("%d %.6f %.6f %.6f %.6f\n" % (int(b[0]), b[1], b[2], b[3], b[4]))


def flip_h(img, boxes):
    img2 = cv2.flip(img, 1)
    nb = []
    for b in boxes:
        cid, x, y, bw, bh = b
        nb.append([cid, 1 - x, y, bw, bh])
    return img2, nb


def brightness(img, factor):
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)


def blur(img, k=3):
    return cv2.GaussianBlur(img, (k, k), 0)


def noise(img, sigma=8):
    g = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + g, 0, 255).astype(np.uint8)


def rotate(img, boxes, angle):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    img2 = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))
    nb = []
    for b in boxes:
        cid, x, y, bw, bh = b
        xa = (x - bw / 2) * w
        xb = (x + bw / 2) * w
        ya = (y - bh / 2) * h
        yb = (y + bh / 2) * h
        pts = np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]], dtype=np.float32)
        pts2 = cv2.transform(pts.reshape(1, -1, 2), M).reshape(-1, 2)
        nx1 = pts2[:, 0].min() / w
        nx2 = pts2[:, 0].max() / w
        ny1 = pts2[:, 1].min() / h
        ny2 = pts2[:, 1].max() / h
        nw = nx2 - nx1
        nh = ny2 - ny1
        if nw <= 0.01 or nh <= 0.01:
            continue
        # 旋转后可能超出边界，裁剪到 [0,1]
        nx1 = max(0.0, nx1)
        ny1 = max(0.0, ny1)
        nx2 = min(1.0, nx2)
        ny2 = min(1.0, ny2)
        nw = nx2 - nx1
        nh = ny2 - ny1
        if nw <= 0.01 or nh <= 0.01:
            continue
        nb.append([cid, (nx1 + nx2) / 2.0, (ny1 + ny2) / 2.0, nw, nh])
    return img2, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/dataset.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-image", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="只统计不写文件（测试用）")
    args = ap.parse_args()
    base = load_base(args.data)
    img_dir = os.path.join(base, "images", args.split)
    lab_dir = os.path.join(base, "labels", args.split)
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    imgs = [p for p in imgs if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            and "_aug" not in os.path.basename(p)]
    random.seed(args.seed)
    np.random.seed(args.seed)
    made = 0
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        txt = os.path.join(lab_dir, stem + ".txt")
        boxes = read_boxes(txt)
        img = cv2.imread(p)
        if img is None:
            continue
        for k in range(args.per_image):
            im2 = img.copy()
            bx = boxes[:]
            op = random.choice(["flip", "bright", "blur", "noise", "rot"])
            if op == "flip":
                im2, bx = flip_h(im2, bx)
            elif op == "bright":
                im2 = brightness(im2, random.uniform(0.7, 1.3))
            elif op == "blur":
                im2 = blur(im2, random.choice([3, 5]))
            elif op == "noise":
                im2 = noise(im2, random.uniform(4, 12))
            elif op == "rot":
                im2, bx = rotate(im2, bx, random.uniform(-12, 12))
            if args.dry:
                made += 1
                continue
            out_stem = "%s_aug%d" % (stem, k)
            cv2.imwrite(os.path.join(img_dir, out_stem + ".jpg"), im2)
            write_boxes(os.path.join(lab_dir, out_stem + ".txt"), bx)
            made += 1
    print("✅ 增强完成: 原 %d 张 -> 新增 %d 张增强样本 (存于 %s)" % (len(imgs), made, img_dir))


if __name__ == "__main__":
    main()
