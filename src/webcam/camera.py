"""
Day 24: 模块化相机接口层（换摄像头只改这里）
================================================================
为什么需要这一层？
  - 你现在的相机是 USB 免驱摄像头（UVC），后期可能换成 RealSense / 工业相机 /
    网络摄像头(RTSP) / 视频文件回放。
  - 如果检测代码里到处写 cv2.VideoCapture(0)，换相机就要改一堆地方，容易出错。
  - 所以把"相机"抽象成一个统一接口：read() -> (frame_bgr, ok)。
    上层（检测器、实时主循环）永远只调用 read()，不关心画面从哪来。

本文件提供 3 种相机实现，全部遵循同一个 BaseCamera 接口：
  1. WebcamCamera      —— 你的 USB 摄像头（cv2.VideoCapture）
  2. ImageSequenceCamera —— 无硬件时回退：读文件夹里的图片序列（验证/演示用）
  3. VideoFileCamera   —— 读一个视频文件回放（验证用）

★ 换摄像头的唯一改动点：create_camera() 工厂函数。
  你只要改 source 字符串和对应参数，上层代码一行都不用动。
"""
from __future__ import annotations

import glob
import os
from abc import ABC, abstractmethod

import cv2
import numpy as np


class BaseCamera(ABC):
    """所有相机的统一接口。子类实现 start / read / stop。

    read() 约定返回 (frame_bgr, ok)：
      - frame_bgr: 一张 BGR 彩色图(numpy 数组)，或 None
      - ok:        bool，是否成功取到一帧
    这样上层无论接什么相机，拿到的数据格式都一致。
    """

    @abstractmethod
    def start(self):
        """打开/初始化相机（在 with 语句进入时自动调用）。"""
        raise NotImplementedError

    @abstractmethod
    def read(self):
        """返回 (frame_bgr, ok)。取不到帧时返回 (None, False)。"""
        raise NotImplementedError

    def stop(self):
        """释放资源（在 with 语句退出时自动调用）。默认无操作。"""
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


class WebcamCamera(BaseCamera):
    """你的 USB 免驱摄像头。OpenCV 直接用系统设备号打开，不需要任何 SDK。

    参数:
      cam_id : 设备号，笔记本内置摄像头一般是 0；外接 USB 摄像头可能是 1
      width/height : 请求的分辨率（摄像头不一定全支持，start 后会按实际值来）
      fps    : 请求帧率
    """

    def __init__(self, cam_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.cam_id = cam_id
        self.req_w, self.req_h, self.req_fps = width, height, fps
        self.cap = None
        self.actual_w = self.actual_h = self.actual_fps = None

    def start(self):
        self.cap = cv2.VideoCapture(self.cam_id)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"打不开摄像头 id={self.cam_id}。\n"
                f"  1) 确认摄像头已插好；\n"
                f"  2) 用 scripts/day24_camera_check.py --list-webcams 看可用设备号；\n"
                f"  3) 如果是虚拟/网络摄像头，cam_id 可能需要换成 URL。"
            )
        # 尽量设置分辨率/帧率（部分摄像头会忽略）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_h)
        self.cap.set(cv2.CAP_PROP_FPS, self.req_fps)
        # 读回实际值，避免后面假设错误
        self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, False
        return frame, True

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class ImageSequenceCamera(BaseCamera):
    """无硬件回退：从文件夹逐张读图片，模拟"一帧一帧"的相机。

    参数:
      image_dir : 图片文件夹（.jpg/.png/.jpeg）
      loop      : 读完一轮是否循环（默认 True，适合长时演示）
    验证整套链路、调试检测逻辑时最方便——不需要真摄像头。
    """

    def __init__(self, image_dir: str, loop: bool = True, exts=(".jpg", ".png", ".jpeg")):
        self.image_dir = image_dir
        self.loop = loop
        self.files = sorted(
            p for ext in exts for p in glob.glob(os.path.join(image_dir, f"*{ext}"))
        )
        if not self.files:
            raise ValueError(f"{image_dir} 里没有图片，请检查路径")
        self.i = 0

    def start(self):
        pass  # 图片列表已在 __init__ 准备好

    def read(self):
        if self.i >= len(self.files):
            if self.loop:
                self.i = 0
            else:
                return None, False
        path = self.files[self.i]
        self.i += 1
        img = cv2.imread(path)
        if img is None:
            return None, False
        return img, True

    def stop(self):
        pass


class VideoFileCamera(BaseCamera):
    """读一个视频文件当相机（回放验证用）。loop=True 时播完自动重头播。"""

    def __init__(self, video_path: str, loop: bool = True):
        self.video_path = video_path
        self.loop = loop
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"打不开视频文件: {self.video_path}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if not ok:
                    return None, False
            else:
                return None, False
        return frame, True

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def create_camera(source: str = "webcam", **kwargs) -> BaseCamera:
    """★ 相机工厂：换摄像头的唯一改动点。

    用法:
      cam = create_camera("webcam", cam_id=0, width=1280, height=720, fps=30)
      cam = create_camera("images", image_dir="data/apple_yolo/images/val")
      cam = create_camera("video",  video_path="demo.mp4")

    上层代码只认 BaseCamera 接口，永远不关心具体是哪款相机。
    """
    source = (source or "webcam").lower()
    if source == "webcam":
        return WebcamCamera(
            cam_id=kwargs.get("cam_id", 0),
            width=kwargs.get("width", 1280),
            height=kwargs.get("height", 720),
            fps=kwargs.get("fps", 30),
        )
    if source == "images":
        if "image_dir" not in kwargs:
            raise ValueError("source='images' 时必须给 image_dir 参数")
        return ImageSequenceCamera(image_dir=kwargs["image_dir"], loop=kwargs.get("loop", True))
    if source == "video":
        if "video_path" not in kwargs:
            raise ValueError("source='video' 时必须给 video_path 参数")
        return VideoFileCamera(video_path=kwargs["video_path"], loop=kwargs.get("loop", True))
    raise ValueError(f"未知相机源: {source}（支持 webcam / images / video）")


def list_webcams(max_id: int = 4):
    """枚举系统里可用的摄像头设备号，方便确认 cam_id 该填几。"""
    found = []
    for i in range(max_id):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
            cap.release()
    return found
