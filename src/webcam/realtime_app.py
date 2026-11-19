"""
Day 26: 实时检测编排（串联相机 + 检测器 + 显示/计数）
================================================================
把前两天做好的两层拼起来：
    相机层(Day24).read()  -->  检测器(Day25).detect()  -->  画框 + 计数 + 显示

设计要点（仍是模块化）：
  - RealtimeDetector 只持有"一个相机"和"一个检测器"，两者都通过统一接口交互；
  - 换摄像头：相机层改一个类，这里一行不动；
  - 换/调检测器：改 AppleDetector 的 weights/conf，这里一行不动；
  - 支持 headless（无显示器）模式：不弹窗口、只存标注帧 + 打印计数，
    方便在没有 GUI 的环境里做真跑验证。
"""
from __future__ import annotations

import os
import time

import cv2

from src.webcam.distractor_suppressor import DistractorSuppressor


class RealtimeDetector:
    def __init__(self, camera, detector, window_name: str = "Apple Detection",
                 suppressor: "DistractorSuppressor | None" = None):
        """
        camera  : 任意 BaseCamera 子类（WebcamCamera / ImageSequenceCamera / ...）
        detector: 任意有 detect(frame)/annotate(frame,dets) 接口的对象（AppleDetector）
        suppressor: 可选。干扰物否决器（DistractorSuppressor），用来删掉落在人/脸上的苹果框。
                    不传则行为和之前完全一致。
        """
        self.camera = camera
        self.detector = detector
        self.window_name = window_name
        self.suppressor = suppressor

    def run(
        self,
        display: bool = True,
        save_dir: str | None = None,
        max_frames: int = 0,
        quit_key: str = "q",
        save_on_space: bool = False,
    ):
        """
        display       : 是否用 cv2.imshow 实时弹窗（无 GUI 的环境设为 False）
        save_dir      : 保存目录；自动模式下每帧存，手动模式下按空格才存
        max_frames    : 最多处理多少帧（0 = 无限，直到退出或相机读完）
        quit_key      : 显示模式下按哪个键退出
        save_on_space : True = 手动保存模式，弹窗里按空格键才存当前帧（不按不存）。
                        需要 display=True；headless 下不支持，会自动退回每帧保存。
                        False（默认）= 每帧自动保存，行为与之前完全一致。

        返回 count_report: list[(frame_index, num_apples)]，供后续统计/导出。
        """
        if save_on_space and not display:
            print("[警告] 按空格保存需要弹窗，headless 模式不支持，已退回每帧自动保存")
            save_on_space = False
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        manual_saved = 0

        count_report = []
        frame_i = 0
        fps = 0.0
        prev_t = time.time()

        with self.camera:  # 自动 start / stop
            while True:
                frame, ok = self.camera.read()
                if not ok:
                    print("[结束] 相机取帧失败或已读完")
                    break

                # 检测
                t0 = time.time()
                dets = self.detector.detect(frame)
                dt_infer = time.time() - t0

                # 干扰物否决（可选）：删掉落在人/脸上的苹果框
                removed = 0
                distractors = []
                if self.suppressor is not None:
                    dets, removed, distractors = self.suppressor.suppress(dets, frame)

                # 画框
                vis = self.detector.annotate(frame, dets)
                # 把干扰物框画成红色，方便你直观看到“为什么删”
                for db in distractors:
                    x1, y1, x2, y2 = [int(v) for v in db["bbox"]]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(vis, f"{db['label']} {db['conf']:.2f}",
                                (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1, cv2.LINE_AA)

                # FPS 平滑
                now = time.time()
                inst = 1.0 / (now - prev_t) if (now - prev_t) > 0 else 0.0
                fps = 0.9 * fps + 0.1 * inst
                prev_t = now

                n = len(dets)
                count_report.append((frame_i, n))

                # 叠加统计文字
                cv2.putText(vis, f"apples: {n}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(vis, f"fps: {fps:.1f}", (10, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

                if save_dir and not save_on_space:
                    cv2.imwrite(os.path.join(save_dir, f"frame_{frame_i:04d}.jpg"), vis)

                if display:
                    if save_on_space:
                        # 手动模式：在画面上提示按键方式
                        cv2.putText(vis, "SPACE=save  Q=quit", (10, vis.shape[0] - 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
                    cv2.imshow(self.window_name, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(quit_key):
                        print("[退出] 用户按键退出")
                        break
                    if save_on_space and key == ord(" "):
                        if save_dir:
                            path = os.path.join(save_dir, f"frame_{frame_i:04d}.jpg")
                            cv2.imwrite(path, vis)
                            manual_saved += 1
                            print(f"  [已保存] {path}  (第 {manual_saved} 张)")
                else:
                    print(f"  frame {frame_i}: apples={n}  infer={dt_infer*1000:.0f}ms  fps={fps:.1f}")

                frame_i += 1
                if max_frames and frame_i >= max_frames:
                    print(f"[结束] 已达到 max_frames={max_frames}")
                    break

        if save_on_space:
            print(f"[手动保存模式] 共保存 {manual_saved} 张标注帧到 {save_dir}")
        return count_report
