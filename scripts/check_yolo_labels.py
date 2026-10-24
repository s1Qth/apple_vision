"""
Day 10 辅助：校验 YOLO 标签格式是否正确。

遍历 labels 目录下所有 .txt，检查：
  - 每行是否 5 个数（class_id x y w h）
  - 坐标是否归一化在 0~1
  - class_id 是否在 [0, nc-1]
并统计：标签文件数、框总数、每个类别的框数。

用法:
    python scripts/check_yolo_labels.py --labels data/apples_yolo/labels --nc 1
"""
import argparse
import glob
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="labels 目录（可含 train/val 子目录）")
    parser.add_argument("--nc", type=int, default=1, help="类别数，class_id 必须在 [0, nc-1]")
    args = parser.parse_args()

    txt_files = sorted(glob.glob(os.path.join(args.labels, "**", "*.txt"), recursive=True))
    if not txt_files:
        raise SystemExit(f"在 {args.labels} 下没找到任何 .txt 标签")

    total_boxes = 0
    class_counts = {}
    bad_files = []

    for f in txt_files:
        with open(f) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    bad_files.append(f"{f}:{lineno} 应有5个数，实际 {len(parts)}")
                    continue
                try:
                    cid, x, y, w, h = map(float, parts)
                except ValueError:
                    bad_files.append(f"{f}:{lineno} 不是数字: {line}")
                    continue
                if not (0 <= cid < args.nc):
                    bad_files.append(f"{f}:{lineno} class_id {cid} 超出 [0,{args.nc-1}]")
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    bad_files.append(f"{f}:{lineno} 坐标不在 0~1: {line}")
                total_boxes += 1
                class_counts[int(cid)] = class_counts.get(int(cid), 0) + 1

    print(f"标签文件数: {len(txt_files)}")
    print(f"框总数: {total_boxes}")
    for cid, n in sorted(class_counts.items()):
        print(f"  类别 {cid}: {n} 个框")

    if bad_files:
        print(f"\n发现 {len(bad_files)} 处问题：")
        for b in bad_files[:20]:
            print("  -", b)
        if len(bad_files) > 20:
            print(f"  ... 其余 {len(bad_files) - 20} 处省略")
        raise SystemExit("标签校验未通过，请修正后再训练")

    print("\n✅ 标签格式全部正确，可以进入 Day 11 训练。")


if __name__ == "__main__":
    main()
