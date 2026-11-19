"""
retrain_prepare.py — 把新拍/新标的数据并入数据集（自动 8:2 切分 + 标签校验）(R2 工具)
用法:
  python scripts/retrain_prepare.py --src new_photos/ --data configs/dataset.yaml --val-ratio 0.2
  python scripts/retrain_prepare.py --src new_photos/ --data configs/dataset.yaml --copy   # 真正复制
说明:
  new_photos/ 里每张图需有同名 .txt 标注（YOLO 格式，类 0=apple）。
  脚本按比例随机切分，复制到 data/apple_yolo/images/{train,val} 与 labels/{train,val}，
  并校验每行格式（class=int, 4 个 0~1 浮点；class 必须为 0）。
  默认只“预览”，加 --copy 才真正复制，避免误操作污染数据集。
"""
import argparse, os, glob, shutil, random
import yaml


def validate_label(txt_path):
    bad = []
    try:
        for i, ln in enumerate(open(txt_path, encoding="utf-8").read().strip().splitlines()):
            if not ln.strip():
                continue
            p = ln.split()
            if len(p) < 5:
                bad.append((i + 1, "列数<5"))
                continue
            try:
                cid = int(p[0])
                vals = [float(x) for x in p[1:5]]
            except Exception:
                bad.append((i + 1, "无法解析"))
                continue
            if cid != 0:
                bad.append((i + 1, "class=%d 应为0" % cid))
            if any(v < 0 or v > 1 for v in vals):
                bad.append((i + 1, "坐标超[0,1]"))
    except Exception as e:
        bad.append((-1, str(e)))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="含 图+同名txt 的文件夹")
    ap.add_argument("--data", default="configs/dataset.yaml")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="默认只预览，加此参数才真复制")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.data, encoding="utf-8"))
    base = cfg["path"]

    imgs = sorted(glob.glob(os.path.join(args.src, "*.*")))
    imgs = [p for p in imgs if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

    ready = []
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        txt = os.path.join(args.src, stem + ".txt")
        if not os.path.exists(txt):
            print("⚠️ 跳过(缺标注): %s" % os.path.basename(p))
            continue
        bad = validate_label(txt)
        if bad:
            print("⚠️ 跳过(标注错误 %s): %s" % (bad, os.path.basename(p)))
            continue
        ready.append(p)

    random.seed(args.seed)
    random.shuffle(ready)
    n_val = int(len(ready) * args.val_ratio)
    val_list = ready[:n_val]
    train_list = ready[n_val:]
    print("待并入: %d 张 (train=%d, val=%d)" % (len(ready), len(train_list), len(val_list)))

    if not args.copy:
        print("（预览模式，未复制。确认无误后加 --copy 执行）")
        return

    for p in train_list:
        stem = os.path.splitext(os.path.basename(p))[0]
        shutil.copy(p, os.path.join(base, "images", "train", os.path.basename(p)))
        shutil.copy(os.path.join(args.src, stem + ".txt"),
                    os.path.join(base, "labels", "train", stem + ".txt"))
    for p in val_list:
        stem = os.path.splitext(os.path.basename(p))[0]
        shutil.copy(p, os.path.join(base, "images", "val", os.path.basename(p)))
        shutil.copy(os.path.join(args.src, stem + ".txt"),
                    os.path.join(base, "labels", "val", stem + ".txt"))

    # 清掉旧缓存，避免训练时读到过期索引
    for c in glob.glob(os.path.join(base, "labels", "*.cache")):
        try:
            os.remove(c)
        except Exception:
            pass
    print("✅ 已复制进数据集。随后跑 retrain_audit.py 复查，再跑 retrain_train.py。")


if __name__ == "__main__":
    main()
