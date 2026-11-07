# Day 26 · 实时检测主循环（串联相机 + 检测器 + 计数）

> 系列：普通摄像头实时苹果识别 · 第 3 天（收尾）
> 目标：把 Day24 相机层 + Day25 检测器串成一个实时循环，画面里实时显示苹果框 + 数量 + FPS。

---

## 1. 今天做什么

把前两天做好的两层拼起来，形成一个能"一直跑"的实时识别程序：

```
相机层.read()  →  检测器.detect()  →  画框 + 计数 + 显示
 (Day24)           (Day25)            (Day26 主循环)
```

产出文件：

| 文件 | 作用 |
|---|---|
| `src/webcam/realtime_app.py` | `RealtimeDetector`：编排主循环（取帧→检测→画框→计数→显示/保存） |
| `scripts/day26_realtime_detect.py` | 命令行入口，选 `--source webcam/images/video` |

---

## 2. 三层如何接线（模块化的关键）

```python
from src.webcam.camera import create_camera     # Day24 相机层
from src.webcam.detector import AppleDetector    # Day25 检测器
from src.webcam.realtime_app import RealtimeDetector

camera   = create_camera("images", image_dir="data/apple_yolo/images/val")
detector = AppleDetector("runs/detect/train/weights/best.pt", conf=0.3)
app      = RealtimeDetector(camera, detector)
report   = app.run(display=True, save_dir="outputs/day26", max_frames=0)
```

- 主循环只认"一个相机接口"和"一个检测器接口"，**完全不知道画面从哪来**。
- **换摄像头**：把 `create_camera("webcam", cam_id=0)` 一改即可，主循环零改动。
- **换/调模型**：改 `AppleDetector` 的 `weights` / `conf`，主循环零改动。

这就是你要求"后期换摄像头好改"的最终落点。

---

## 3. 主循环核心逻辑（`realtime_app.py`）

```python
with self.camera:                         # 自动 start/stop
    while True:
        frame, ok = self.camera.read()    # 取一帧（来自任意相机）
        if not ok: break
        dets = self.detector.detect(frame)         # 算苹果框
        vis  = self.detector.annotate(frame, dets) # 画框
        # 叠加计数 + FPS
        cv2.putText(vis, f"apples: {len(dets)}", ...)
        cv2.putText(vis, f"fps: {fps:.1f}", ...)
        if save_dir: cv2.imwrite(...)     # headless 模式存图
        if display:
            cv2.imshow(...); if waitKey==q: break  # 有界面时弹窗
```

关键参数：
- `display=True`：弹实时窗口（你本机接摄像头用）；
- `display=False`（或 `--no-display`）：不弹窗，只存标注帧 + 打印计数（沙箱/无 GUI 验证用）；
- `max_frames`：限制帧数（0=无限），演示/验证时方便设小。

---

## 4. 真跑验证（本机命令）

```bash
# 无硬件回退（本系列全程用这个验证，沙箱/没接摄像头都能跑）
python scripts/day26_realtime_detect.py --source images \
    --image-dir data/apple_yolo/images/val --no-display --max-frames 3

# 你本机接上 USB 摄像头（真实实时画面）
python scripts/day26_realtime_detect.py --source webcam --cam-id 0

# 视频文件回放
python scripts/day26_realtime_detect.py --source video --video-path demo.mp4
```

验证输出（节选）：
```
  frame 0: apples=10  infer=4290ms  fps=0.0     # 首帧含模型预热
  frame 1: apples=8   infer=80ms    fps=1.0
  frame 2: apples=8   infer=73ms    fps=2.0
✅ 实时识别链路验证通过：共处理 3 帧，累计检出苹果 26 个
   标注帧已存：outputs/day26
   计数报告：outputs/day26_report.json
```
- 标注帧示例：`outputs/day26/frame_0000.jpg`（绿框 + 置信度 + 左上角 `apples: N` / `fps:`）。
- 计数报告：`outputs/day26_report.json`，含每帧苹果数与总计，方便后续做统计/导出。

### 性能说明（CPU 实测）
- 首帧 ~4.3s 是 YOLO 模型加载/编译预热，之后稳定在 **73–80ms/帧 ≈ 12–13 FPS**（纯 CPU）。
- 如需更高帧率：换 GPU（`--device 0`）、或减小输入尺寸（`detector.detect` 里 `predict(imgsz=320)`）。

---

## 5. 你本机接摄像头的注意点

1. 先确认设备号：`python scripts/day24_camera_check.py --list-webcams`；
2. 若 `cam_id=0` 打不开，改成枚举出来的号；
3. 显示窗口里按 **`q`** 退出；
4. 如果画面里苹果框太多/太少，调 `--conf`（0.3 起，误检多就调高，漏检多就调低）。

---

## 6. 本系列成果 & 下一步

✅ **三天成果**：一个模块化、可换摄像头的实时苹果识别程序。
- `src/webcam/camera.py`：相机接口层（换摄像头只改这里）
- `src/webcam/detector.py`：苹果检测器（封装 best.pt）
- `src/webcam/realtime_app.py` + `scripts/day26_realtime_detect.py`：实时主循环

🔜 **下一步（之前讨论过、尚未做）**：在检测器拿到苹果框之后，加一层**大小分类**——根据"给定的数据/尺寸阈值"把苹果标成「大果/小果」。因为单目摄像头没有深度，真实直径需要靠标定/参照物换算（这正是之前大纲里 Day 3 的尺寸测量模块）。等你把"给定数据"的标准定下来（物理直径阈值 or 像素阈值），我就接着往下推。
