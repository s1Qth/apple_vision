# Day 24 · 模块化相机接口层（换摄像头只改这里）

> 系列：普通摄像头实时苹果识别 · 第 1 天
> 目标：把"相机"抽象成统一接口，让后期换摄像头（USB → RealSense/工业相机/网络摄像头）只改一个文件。

---

## 1. 为什么今天要先做"相机层"

你这次的相机是一台 **USB 免驱摄像头（UVC 标准，CMOS 彩色传感器）**：
- 接口 USB，免驱，OpenCV 的 `cv2.VideoCapture` 直接能用，不需要任何 SDK；
- 分辨率 320×240 / 640×480 / 1280×720 / 1920×1080 / 2560×1440 / 3840×2160，均 **30FPS**；
- 24bit 真彩、手动对焦、最近对焦 5cm。

但**后期你要换摄像头**（比如升级成 RealSense 做深度、或工业相机做高速）。如果检测代码里到处写 `cv2.VideoCapture(0)`，换相机就得改一堆地方，极易出错。

所以今天的核心思想：**把"相机"抽象成一个统一接口 `read() -> (frame, ok)`**。上层（检测器、实时主循环）永远只调用 `read()`，不关心画面从哪来。换相机 = 换一个底层类，上层一行不动。

这就是你要求的"模块化、后期换摄像头好改"的落点。

---

## 2. 今天的产出

| 文件 | 作用 |
|---|---|
| `src/webcam/camera.py` | 相机接口层：基类 + 3 种相机 + 工厂函数 |
| `scripts/day24_camera_check.py` | 验证脚本（枚举 + 取帧） |

`src/webcam/camera.py` 提供 3 种相机，**全遵循同一个 `BaseCamera` 接口**：

| 类 | 用途 | 是否需硬件 |
|---|---|---|
| `WebcamCamera` | 你的 USB 摄像头（`cv2.VideoCapture`） | 需要 |
| `ImageSequenceCamera` | 读文件夹图片序列（验证/演示回退） | 不需要 |
| `VideoFileCamera` | 读视频文件回放（验证回退） | 不需要 |

接口约定（所有相机都遵守）：
```python
frame, ok = camera.read()   # frame: BGR 图(numpy)或None；ok: 是否成功
```

---

## 3. 关键代码讲解

### 3.1 统一接口（抽象基类）
```python
class BaseCamera(ABC):
    @abstractmethod
    def start(self): ...          # 打开相机（with 进入时自动调）
    @abstractmethod
    def read(self): ...           # 返回 (frame_bgr, ok)
    def stop(self): ...           # 释放（with 退出时自动调）
    def __enter__(self): self.start(); return self
    def __exit__(self, *e): self.stop()
```

### 3.2 你的 USB 摄像头
```python
class WebcamCamera(BaseCamera):
    def __init__(self, cam_id=0, width=1280, height=720, fps=30): ...
    def start(self):
        self.cap = cv2.VideoCapture(self.cam_id)
        if not self.cap.isOpened(): raise RuntimeError("打不开摄像头...")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.req_w)   # 尽量设分辨率/帧率
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_h)
        self.cap.set(cv2.CAP_PROP_FPS, self.req_fps)
    def read(self):
        ok, frame = self.cap.read()
        return (frame, True) if ok else (None, False)
```
> `cam_id`：笔记本内置摄像头一般是 `0`；外接 USB 摄像头可能是 `1`。用 `list_webcams()` 确认。

### 3.3 回退相机（无硬件也能调）
```python
class ImageSequenceCamera(BaseCamera):    # 读文件夹图片当"一帧帧"
    def __init__(self, image_dir, loop=True): ...
class VideoFileCamera(BaseCamera):        # 读视频文件
    def __init__(self, video_path, loop=True): ...
```
> 这两个让你**在没有摄像头时也能把整套链路跑通、调检测逻辑**——本系列全程用它们做真跑验证。

### 3.4 ★ 换摄像头的唯一改动点：工厂函数
```python
def create_camera(source="webcam", **kwargs) -> BaseCamera:
    source = source.lower()
    if source == "webcam":  return WebcamCamera(cam_id=kwargs.get("cam_id",0), ...)
    if source == "images":  return ImageSequenceCamera(image_dir=kwargs["image_dir"], ...)
    if source == "video":   return VideoFileCamera(video_path=kwargs["video_path"], ...)
```
以后换 RealSense/网络摄像头，**只需在这里加一个 `if source == "realsense":` 分支、写一个新类**，上层 `create_camera(...)` 的调用方式完全不变。

---

## 4. 真跑验证（本机命令）

```bash
# 1) 枚举本机可用摄像头设备号
python scripts/day24_camera_check.py --list-webcams

# 2) 用图片序列回退验证 read()（无需硬件）
python scripts/day24_camera_check.py --source images \
    --image-dir data/apple_yolo/images/val --max-frames 3
```

预期输出（节选）：
```
检测到可用摄像头 id: [0, 1]          # 沙箱里的虚拟设备，真实机器会显示你接的摄像头
使用相机源: images  参数: {...}
  frame 0: 尺寸=(748x502x3)  dtype=uint8  ok=True
  frame 1: 尺寸=(706x495x3)  dtype=uint8  ok=True
  frame 2: 尺寸=(333x498x3)  dtype=uint8  ok=True
✅ 相机接口层验证通过：read() 稳定返回 (frame, ok)
```

> 注：`list_webcams` 探测时可能打印 `Camera index out of range` 之类的 ERROR，是 OpenCV 探测空设备槽的无害报错，不影响功能。

---

## 5. 你本机接上摄像头后怎么用

```bash
# 直接读你的 USB 摄像头（把画面交给明后两天的检测器）
python scripts/day24_camera_check.py --source webcam --max-frames 50
```
若报错"打不开摄像头"，先跑 `--list-webcams` 看设备号，再 `create_camera("webcam", cam_id=正确的号)`。

---

## 6. 明天预告

Day 25 做**苹果检测器模块**：把之前训好的 `best.pt` 封装成一个 `AppleDetector` 类，`detect(frame)` 直接返回一串苹果框。它只依赖今天的相机层提供的 `frame`，两者解耦——这正是模块化的好处。
