"""
Day16 闭环演示：看到苹果 -> 框出(YOLO) -> 深度 -> 相机坐标 -> 手眼标定 -> 基座坐标
==================================================================================
把 Day9~15 所有零件拼成一条完整链路，最终输出"机械臂基座坐标系下的苹果位置"，
也就是可以直接发给机器人去抓的目标点。

两种运行方式：
  A) 教学/测试（不用模型、不用相机，手动给一个点）：
     python scripts/day16_closed_loop.py \
         --manual-u 340 --manual-v 220 --manual-z 0.8 --depth 0.8

  B) 真实（用你 Day11 训练的 best.pt + 一张图 + 深度）：
     python scripts/day16_closed_loop.py \
         --weights runs/detect/train/weights/best.pt \
         --image data/raw/apple.jpg --depth 0.8

依赖：opencv-python, numpy, pyyaml, ultralytics(仅方式B需要)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

# 让脚本能 import src 包（运行目录是 apple_vision/）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.localization.pixel_to_3d import (
    pixel_to_camera_coords,
    load_handeye_x,
    camera_to_base,
)


def parse_args():
    p = argparse.ArgumentParser(description="Day16 采摘闭环：相机坐标 -> 基座坐标")
    p.add_argument("--weights", default="yolov8n.pt", help="YOLO 权重(方式B)")
    p.add_argument("--image", help="输入图片(方式B；或在方式A里当作背景图)")
    p.add_argument("--conf", type=float, default=0.5)

    # 深度来源（二选一）：标量深度 或 深度图
    p.add_argument("--depth", type=float, default=0.8,
                   help="标量深度(米)，所有苹果共用；演示用")
    p.add_argument("--depth-image", help="深度图(png/16bit, 单位毫米)；真实 RGB-D 用")

    # 手测点（方式A，不用模型）
    p.add_argument("--manual-u", type=float, help="手动指定像素 u（列）")
    p.add_argument("--manual-v", type=float, help="手动指定像素 v（行）")
    p.add_argument("--manual-z", type=float, help="手动指定深度 Z(米)")

    # 配置路径
    p.add_argument("--camera", default="configs/camera_intrinsics.yaml")
    p.add_argument("--handeye", default="configs/handeye.yaml")
    p.add_argument("--out", default="outputs/day16_targets.json")
    return p.parse_args()


def load_camera_intrinsics(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    c = cfg["camera"]
    return (c["fx"], c["fy"], c["cx"], c["cy"])

def intrinsics_from_image(w, h, fx_ref=525.0, ref_w=640.0):
    """不标定：光轴取画面正中心，焦距按宽度等比缩放。
    保证 x=0 / y=0 落在图片中线上；绝对米数仍是近似值。"""
    s = w / ref_w
    return fx_ref * s, fx_ref * s, w / 2.0, h / 2.0


def load_method(handeye_path):
    import yaml
    with open(handeye_path, encoding="utf-8") as f:
        return yaml.safe_load(f)["calibration"]["method"]


def get_depth_at(depth_image, u, v):
    """从深度图取 (u,v) 处深度，返回米。无效值(0)取 5x5 邻域中位数。"""
    h, w = depth_image.shape[:2]
    u = int(np.clip(round(u), 0, w - 1))
    v = int(np.clip(round(v), 0, h - 1))
    z_mm = depth_image[v, u]
    if z_mm == 0:  # 无效深度，取邻域中位数
        patch = depth_image[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
        vals = patch[patch > 0]
        if len(vals) == 0:
            return None
        z_mm = np.median(vals)
    return float(z_mm) / 1000.0  # 毫米 -> 米


def detect_apples_yolo(weights, image, conf):
    """返回 (dets, annotated_img)。dets 每项含 u/v/conf/box。"""
    from ultralytics import YOLO
    model = YOLO(weights)
    results = model.predict(source=image, conf=conf, verbose=False)
    dets = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append({
                "u": (x1 + x2) / 2.0,
                "v": (y1 + y2) / 2.0,
                "conf": float(b.conf[0]),
                "box": (int(x1), int(y1), int(x2), int(y2)),
            })
        annotated = r.plot()  # BGR 图，已画好框
    return dets, (annotated if results else None)


def main():
    args = parse_args()

    # 👇👇👇 新增：每次运行前清空旧的 JSON 文件 👇👇👇
    if os.path.exists(args.out):
        os.remove(args.out)
        print(f"🧹 已清理旧的 JSON 文件: {args.out}")
    # 👆👆👆 新增结束 👆👆👆

        # ----- 内参：有图就按这张图的分辨率推中线；没图（方式A）才回落到 yaml -----
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"读不到图片: {args.image}")
        h, w = img.shape[:2]
        fx, fy, cx, cy = intrinsics_from_image(w, h)
        print(f"[内参] 由图片 {w}x{h} 推出: fx={fx:.1f} cx={cx:.1f}(中线) cy={cy:.1f}(中线)（假设值，未标定）")
    else:
        fx, fy, cx, cy = load_camera_intrinsics(args.camera)
        print(f"[内参] 来自 {args.camera}")
    X = load_handeye_x(args.handeye)
    method = load_method(args.handeye)
    note = "（模拟值，仅供演示；真机请用 day15_handeye_real.py 标定后替换）" \
        if method == "simulated" else ""
    print(f"[配置] 手眼标定方法 = {method} {note}")

    # 深度图（可选）
    depth_image = None
    if args.depth_image:
        depth_image = cv2.imread(args.depth_image, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise SystemExit(f"读不到深度图: {args.depth_image}")

    # ----- 收集待处理的目标点 -----
    targets = []
    drawn = None  # 用于画标注的底图

    if args.manual_u is not None and args.manual_v is not None and args.manual_z is not None:
        # 方式A：手动给一个点（教学/测试用，不需要模型）
        targets.append({
            "u": args.manual_u, "v": args.manual_v,
            "z": args.manual_z, "conf": 1.0, "box": None,
        })
        if args.image:  # 如果有背景图就画上去
            drawn = cv2.imread(args.image)
    else:
        # 方式B：用 YOLO 检测
        if not args.image:
            raise SystemExit("方式B 需要 --image；或方式A 提供 --manual-u/v/z")
        dets, annotated = detect_apples_yolo(args.weights, args.image, args.conf)
        drawn = annotated
        for d in dets:
            z = get_depth_at(depth_image, d["u"], d["v"]) if depth_image is not None else args.depth
            if z is None or z <= 0:
                z = args.depth
            d["z"] = z
            targets.append(d)

    if not targets:
        raise SystemExit("没有检测到任何目标点，请检查输入。")

    # ----- 核心：相机坐标 -> 基座坐标 -----
    results_out = []
    for i, t in enumerate(targets):
        # 1) 像素 + 深度 -> 相机坐标（Day14 的针孔模型）
        p_cam = pixel_to_camera_coords(t["u"], t["v"], t["z"], fx, fy, cx, cy)
        # 2) 手眼标定：相机坐标 -> 基座坐标（Day16 新增的关键一步）
        p_base = camera_to_base(p_cam, X)

        results_out.append({
            "id": i,
            "pixel": [round(t["u"], 1), round(t["v"], 1)],
            "depth_m": round(t["z"], 3),
            "camera_xyz_m": [round(float(x), 4) for x in p_cam],
            "base_xyz_m": [round(float(x), 4) for x in p_base],
            "conf": round(t.get("conf", 1.0), 3),
        })

                # 在图上标注基座坐标
        if drawn is not None and t.get("box") is not None:
            x1, y1, x2, y2 = t["box"]
            
            # 格式化要显示的文字 (保留两位小数)
            label = f"#{i} base=({p_base[0]:.2f},{p_base[1]:.2f},{p_base[2]:.2f})"
            
            # 1. 画绿色的检测框
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 2. 在框的正下方写坐标 (y2 + 20 就是往下挪 20 个像素)
            # 为了防止字超出图片底部，用 min(y2 + 20, 图片高度 - 10) 兜底
            img_h = drawn.shape[0]
            text_y = min(y2 + 20, img_h - 10)
            
            # 用红色 (0, 0, 255) 显示，字号 0.6，加粗 2，这样最显眼！
            cv2.putText(drawn, label, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ----- 输出 -----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results_out, f, ensure_ascii=False, indent=2)

    print("\n=== 采摘闭环结果（发给机械臂的目标点）===")
    for r in results_out:
        print(f"苹果#{r['id']}  像素({r['pixel'][0]},{r['pixel'][1]})  深度{r['depth_m']}m")
        print(f"   相机坐标: ({r['camera_xyz_m'][0]}, {r['camera_xyz_m'][1]}, {r['camera_xyz_m'][2]}) m")
        print(f"   基座坐标: ({r['base_xyz_m'][0]}, {r['base_xyz_m'][1]}, {r['base_xyz_m'][2]}) m  <-- 发给机械臂")
    print(f"\n已保存 JSON: {args.out}")
    if drawn is not None:
        out_img = "outputs/day16_result.jpg"
        cv2.imwrite(out_img, drawn)
        print(f"已保存可视化: {out_img}")


if __name__ == "__main__":
    main()