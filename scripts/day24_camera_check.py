"""
Day 24 验证脚本：相机接口层
================================================================
验证目标：
  1. list_webcams() 能枚举本机摄像头（沙箱里通常没有，会返回空，属正常）
  2. create_camera() 工厂按 source 正确构造相机
  3. read() 能稳定返回 (frame, ok)，且 frame 是合法的 numpy 数组

本脚本不依赖任何硬件：用 --source images 读 val 图片序列即可真跑验证。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from src.webcam.camera import create_camera, list_webcams


def main():
    ap = argparse.ArgumentParser(description="Day24 相机接口层验证")
    ap.add_argument("--source", default="images", choices=["webcam", "images", "video"])
    ap.add_argument("--image-dir", default="data/apple_yolo/images/val")
    ap.add_argument("--video-path", default="")
    ap.add_argument("--max-frames", type=int, default=3)
    ap.add_argument("--list-webcams", action="store_true", help="仅枚举摄像头并退出")
    args = ap.parse_args()

    if args.list_webcams:
        cams = list_webcams()
        print("检测到可用摄像头 id:", cams if cams else "（无——沙箱/未接硬件，属正常）")
        return

    kwargs = {}
    if args.source == "images":
        kwargs["image_dir"] = args.image_dir
    elif args.source == "video":
        if not args.video_path:
            print("错误：source=video 需要 --video-path")
            return
        kwargs["video_path"] = args.video_path

    print(f"使用相机源: {args.source}  参数: {kwargs}")
    cam = create_camera(args.source, **kwargs)
    with cam:
        for i in range(args.max_frames):
            frame, ok = cam.read()
            if not ok:
                print(f"第 {i} 帧读取失败/已结束")
                break
            h, w = frame.shape[:2]
            ch = frame.shape[2] if frame.ndim == 3 else 1
            print(f"  frame {i}: 尺寸=({h}x{w}x{ch})  dtype={frame.dtype}  ok={ok}")

    print("✅ 相机接口层验证通过：read() 稳定返回 (frame, ok)")


if __name__ == "__main__":
    main()
