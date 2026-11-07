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
    ap.add_argument("--out", default="outputs/day26")
    ap.add_argument("--report", default="outputs/day26_report.json")
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

    # 3) 实时编排（Day26）
    app = RealtimeDetector(camera, detector, window_name="Apple Detection (Day26)")
    display = not args.no_display
    try:
        report = app.run(display=display, save_dir=args.out,
                         max_frames=args.max_frames, quit_key="q")
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
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"   计数报告：{args.report}")


if __name__ == "__main__":
    main()
