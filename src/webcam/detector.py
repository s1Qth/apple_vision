"""
Day 25: 苹果检测器模块（封装 YOLOv8，复用已训 best.pt）
================================================================
前面 Day11-16 你已经训好了一个苹果检测模型 best.pt（mAP@0.5≈0.69）。
今天不重新训练，而是把它封装成一个干净、可复用的类 AppleDetector：

    det = AppleDetector(weights="runs/detect/train/weights/best.pt", conf=0.3)
    dets = det.detect(frame_bgr)     # -> 一串苹果框（结构化 dict 列表）
    vis = det.annotate(frame_bgr, dets)  # -> 画好框的图（给人看）

为什么单独封装一层？
  - 实时主循环(Day26)只关心"给我苹果框"，不关心 YOLO 内部；
  - 以后换模型 / 调阈值 / 换类别，只改这一处；
  - detect() 是纯函数式、好单测、好接别的相机来源。

每个检测结果是一个 dict：
    {
      "id":     0,                 # 该帧内的检测序号
      "bbox":   [x1, y1, x2, y2],  # 像素坐标，左上+右下
      "conf":   0.87,              # 置信度 0~1
      "cls":    0,                 # 类别 id（本数据只有 1 类：apple）
      "label":  "apple",
      "center": (u, v),            # 框中心像素 (列, 行)
    }
"""
from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO


class AppleDetector:
    """对单张 BGR 彩色图做苹果检测，返回结构化结果列表。"""

    def __init__(self, weights: str, conf: float = 0.3, device: str = "cpu"):
        """
        weights : YOLOv8 权重路径，例如 runs/detect/train/weights/best.pt
        conf    : 置信度阈值，低于它的框不要（越高越严格、越少误检）
        device  : 'cpu' 或 '0'(GPU)；本机没显卡用 'cpu'
        """
        self.model = YOLO(weights)
        self.conf = float(conf)
        self.device = device
        names = getattr(self.model, "names", None)
        self.class_names = names if isinstance(names, dict) else {0: "apple"}

    def detect(self, frame_bgr: np.ndarray) -> list:
        """
        输入一张 BGR 彩色图（来自相机层的 read()），返回检测列表。
        没有任何苹果时返回空列表（不会报错）。
        """
        results = self.model.predict(
            frame_bgr, conf=self.conf, device=self.device, verbose=False
        )
        r = results[0]
        dets: list = []
        if r.boxes is None or len(r.boxes) == 0:
            return dets

        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            dets.append(
                {
                    "id": i,
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "cls": cls,
                    "label": self.class_names.get(cls, str(cls)),
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                }
            )
        # 按置信度从高到低排，方便后续"最可信的优先"
        dets.sort(key=lambda d: -d["conf"])
        return dets

    def annotate(self, frame_bgr: np.ndarray, dets: list, color=(0, 200, 0)) -> np.ndarray:
        """
        在图上画检测框 + 置信度，返回一张新图，不改变原图。
        color 用 (B, G, R)，默认绿色。
        """
        vis = frame_bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = [int(round(c)) for c in d["bbox"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis, f"{d['label']} {d['conf']:.2f}", (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        return vis
