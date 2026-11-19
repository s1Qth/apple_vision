"""
Day 26 主程序：实时苹果识别（串联相机 + 检测器 + 主循环）
================================================================
用法:
  # 有摄像头时（你本机）
  python scripts/day26_realtime_detect.py --source webcam --cam-id 0

  # 无硬件时回退验证（沙箱/没接摄像头）
  python scripts/day26_realtime_detect.py --source images \
      --image-dir data/apple_yolo/images/val --no-display --max-frames 3

  # 视频文件回放
  python scripts/day26_realtime_detect.py --source video --video-path demo.mp4

  # 开启“干扰物否决”：用本地 yolov8n 的 person 类删掉落在人/脸上的苹果框（零下载）
  python scripts/day26_realtime_detect.py --source webcam --cam-id 1 --suppress

  # 【推荐】把 COCO 常见物品都当干扰（自动排除 apple 类，避免误删真苹果）
  python scripts/day26_realtime_detect.py --source webcam --cam-id 1 \
      --suppress --block-all

  # 按键保存：不自动存盘，弹窗里按空格才存一张标注帧，q 退出
  python scripts/day26_realtime_detect.py --source webcam --cam-id 1 \
      --weights models/apple_pretrained.pt --suppress --block-all --save-on-space

  # 换专门的脸模型（下载 yolov8n-face.pt 放到 weights/ 后）
  python scripts/day26_realtime_detect.py --source webcam --cam-id 1 \
      --suppress --distractor weights/yolov8n-face.pt --block 0
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import cv2

from src.webcam.camera import create_camera, list_webcams_info
from src.webcam.detector import AppleDetector
from src.webcam.realtime_app import RealtimeDetector
from src.webcam.distractor_suppressor import DistractorSuppressor


def main():
    ap = argparse.ArgumentParser(description="Day26 实时苹果识别")
    ap.add_argument("--source", default="images", choices=["webcam", "images", "video"])
    ap.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    ap.add_argument("--image-dir", default="data/apple_yolo/images/val")
    ap.add_argument("--video-path", default="")
    ap.add_argument("--cam-id", type=int, default=0, help="摄像头设备号：内置一般 0，外接 USB 一般 1")
    ap.add_argument("--conf", type=float, default=0.5, help="置信度阈值，越高越严格、误检越少（脸误检建议 0.6~0.7）")
    ap.add_argument("--min-aspect", type=float, default=0.45, help="框宽/高比下限，过窄的框（如侧脸）丢弃")
    ap.add_argument("--max-aspect", type=float, default=2.2, help="框宽/高比上限，过长的框（如竖脸）丢弃")
    ap.add_argument("--list-cameras", action="store_true", help="列出所有可用摄像头设备号+分辨率后退出")
    ap.add_argument("--max-frames", type=int, default=0, help="0=无限（直到退出/读完）")
    ap.add_argument("--no-display", action="store_true", help="headless：不弹窗口，仅保存标注帧")
    ap.add_argument("--save-on-space", action="store_true",
                    help="按键保存模式：不自动存盘，弹窗里按空格键才存当前帧，q 退出；"
                         "需弹窗(不能和 --no-display 同用)")
    ap.add_argument("--out", default="outputs/day26")
    ap.add_argument("--report", default="outputs/day26_report.json")
    # 干扰物否决（治“脸被识别成苹果”）
    ap.add_argument("--suppress", action="store_true",
                    help="开启干扰物否决：删掉落在干扰物上的苹果框")
    ap.add_argument("--distractor", default="yolov8n.pt",
                    help="干扰模型路径，默认本地 yolov8n.pt(COCO 80 类)")
    ap.add_argument("--block", type=int, nargs="+", default=[0],
                    help="要否决的干扰类别 id，yolov8n 里 person=0；脸模型里 face 通常是 0")
    ap.add_argument("--block-all", action="store_true",
                    help="把干扰模型识别的所有类都当干扰（自动排除 apple 类），突出苹果；"
                         "开启后忽略 --block")
    ap.add_argument("--distractor-conf", type=float, default=0.3,
                    help="干扰模型置信度阈值")
    ap.add_argument("--iou", type=float, default=0.35,
                    help="苹果框与干扰框重叠超过该 IoU 就删（0.3~0.5）")
    args = ap.parse_args()

    # 0) 列出摄像头后退出（帮你确认该填哪个 cam-id）
    if args.list_cameras:
        print("===== 可用摄像头（id : 实际分辨率）=====")
        for c in list_webcams_info():
            res = f"{c['w']}x{c['h']}" if c["w"] else "无法读帧"
            print(f"  cam_id={c['id']} -> {res}")
        print("提示：内置摄像头通常是 0；你接的 USB 摄像头通常是 1。")
        return

    # 1) 相机层（Day24）
    cam_kwargs = {}
    if args.source == "images":
        cam_kwargs["image_dir"] = args.image_dir
    elif args.source == "video":
        cam_kwargs["video_path"] = args.video_path
    else:  # webcam
        cam_kwargs["cam_id"] = args.cam_id
        cam_kwargs["width"] = 1280
        cam_kwargs["height"] = 720
        cam_kwargs["fps"] = 30
    camera = create_camera(args.source, **cam_kwargs)

    # 2) 检测器（Day25）—— 启动时明确打印用的是哪个模型，避免"不知道接的什么模型"
    detector = AppleDetector(
        args.weights, conf=args.conf, device="cpu",
        min_aspect=args.min_aspect, max_aspect=args.max_aspect,
    )
    print("========== 模型信息 ==========")
    print(detector.summary())
    print("==============================")

    # 3) 实时编排（Day26）—— 可选接入干扰物否决器
    suppressor = None
    if args.suppress:
        suppressor = DistractorSuppressor(
            weights=args.distractor, block_classes=args.block,
            conf=args.distractor_conf, iou_thresh=args.iou, device="cpu",
            block_all=args.block_all,
        )
        print("========== 干扰模型信息 ==========")
        suppressor.summary()
        print("==================================")
    app = RealtimeDetector(camera, detector, window_name="Apple Detection (Day26)",
                           suppressor=suppressor)
    display = not args.no_display
    try:
        report = app.run(display=display, save_dir=args.out,
                         max_frames=args.max_frames, quit_key="q",
                         save_on_space=args.save_on_space)
    finally:
        cv2.destroyAllWindows()

    # 4) 汇总 + 导出计数报告
    total_apples = sum(n for _, n in report)
    print(f"\n✅ 实时识别链路验证通过：共处理 {len(report)} 帧，累计检出苹果 {total_apples} 个")
    print(f"   标注帧已存：{args.out}")
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(
            {
                "frames": len(report),
                "total_apples": total_apples,
                "per_frame": [{"frame": i, "apples": n} for i, n in report],
                "source": args.source,
                "weights": args.weights,
                "conf": args.conf,
                "suppress": bool(args.suppress),
                "block_all": bool(args.block_all) if args.suppress else False,
                "distractor": args.distractor if args.suppress else None,
                "save_on_space": bool(args.save_on_space),
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"   计数报告：{args.report}")


if __name__ == "__main__":
    main()
