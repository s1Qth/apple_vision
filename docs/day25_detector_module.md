# Day 25 · 苹果检测器模块（封装 YOLOv8，复用 best.pt）

> 系列：普通摄像头实时苹果识别 · 第 2 天
> 目标：把之前训好的 `best.pt` 封装成 `AppleDetector`，让上层只调用 `detect(frame)` 拿苹果框。

---

## 1. 今天做什么

你之前 Day11–16 已经训好一个苹果检测模型 `runs/detect/train/weights/best.pt`（mAP@0.5≈0.69）。今天**不重新训练**，而是把它封装成一个干净、可复用的类：

```python
det = AppleDetector(weights="runs/detect/train/weights/best.pt", conf=0.3)
dets = det.detect(frame_bgr)        # frame 来自相机层的 read()
vis  = det.annotate(frame_bgr, dets)  # 画好框的图
```

为什么要单独封装：
- 实时主循环（Day26）只关心"给我苹果框"，不关心 YOLO 内部；
- 以后换模型 / 调阈值 / 加类别，只改这一处；
- `detect()` 是纯函数式、好单测、好接任何相机来源。

---

## 2. 产出文件

| 文件 | 作用 |
|---|---|
| `src/webcam/detector.py` | `AppleDetector` 类（加载 best.pt + 检测 + 画框） |
| `scripts/day25_detector_test.py` | 单图/多图验证脚本 |

---

## 3. 关键代码讲解

### 3.1 初始化：加载模型
```python
class AppleDetector:
    def __init__(self, weights, conf=0.3, device="cpu"):
        self.model = YOLO(weights)
        self.conf = float(conf)
        self.device = device
        names = getattr(self.model, "names", None)
        self.class_names = names if isinstance(names, dict) else {0: "apple"}
```
- `weights`：YOLOv8 权重路径（用你训好的 `best.pt`）。
- `conf`：置信度阈值。**越高越严格、误检少但可能漏检；越低越灵敏但可能误检**。先从 0.3 起，现场按效果调。
- `device`：没显卡用 `'cpu'`。

### 3.2 检测：返回结构化结果
```python
def detect(self, frame_bgr):
    results = self.model.predict(frame_bgr, conf=self.conf, device=self.device, verbose=False)
    r = results[0]
    dets = []
    if r.boxes is None or len(r.boxes) == 0:
        return dets                      # 没苹果就返回空列表，不报错
    for i, box in enumerate(r.boxes):
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        conf = float(box.conf[0]); cls = int(box.cls[0])
        dets.append({
            "id": i,
            "bbox": [x1, y1, x2, y2],
            "conf": conf, "cls": cls,
            "label": self.class_names.get(cls, str(cls)),
            "center": ((x1+x2)/2.0, (y1+y2)/2.0),
        })
    dets.sort(key=lambda d: -d["conf"])   # 按置信度从高到低，方便"最可信优先"
    return dets
```
每个检测结果是一个 dict，字段：
- `bbox`：左上+右下像素坐标
- `conf`：置信度 0~1
- `center`：框中心像素 `(u, v)`（后续做尺寸测量/定位会用到）
- `label` / `cls` / `id`

### 3.3 画框（给人看的）
```python
def annotate(self, frame_bgr, dets, color=(0, 200, 0)):
    vis = frame_bgr.copy()
    for d in dets:
        x1,y1,x2,y2 = [int(round(c)) for c in d["bbox"]]
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        cv2.putText(vis, f"{d['label']} {d['conf']:.2f}", (x1, max(y1-6,12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis
```

---

## 4. 真跑验证（本机命令）

```bash
python scripts/day25_detector_test.py \
    --weights runs/detect/train/weights/best.pt \
    --image-dir data/apple_yolo/images/val --max 5 --conf 0.3
```

预期输出（节选）：
```
apple_41.jpg: 10 个苹果  [#0 conf=0.93, #1 conf=0.93, ...]
apple_42.jpg:  8 个苹果  [#0 conf=0.97, ...]
apple_43.jpg:  8 个苹果  [#0 conf=0.99, ...]
apple_44.jpg:  4 个苹果  [#0 conf=0.91, ...]
apple_45.jpg:  9 个苹果  [#0 conf=0.95, ...]

✅ 检测器验证通过：共检出 39 个苹果，标注图已存 outputs/day25
✅ 检测字段完整（含 ('bbox', 'conf', 'center', 'label', 'id')）
```
标注图保存在 `outputs/day25/`，可用图片查看器打开核对框得对不对。

---

## 5. 调参提示

- **误检多**（把叶子/背景当苹果）→ 把 `conf` 调到 0.4~0.5。
- **漏检多**（远处/小苹果没框住）→ 把 `conf` 降到 0.2，或换更大的输入尺寸（YOLO 默认 640，可在 `predict` 加 `imgsz=1280`）。
- 不满意总体效果 → 回到 Day11–16 增数据/调训练，再导出新的 `best.pt`，检测器这里只改 `weights` 路径。

---

## 6. 明天预告

Day 26 做**实时检测主循环**：用 Day24 的相机层取帧、Day25 的检测器算框，串成一个 `while` 循环，实时显示画面 + 画框 + 计苹果数 + 显示 FPS。换摄像头只需把 `--source` 从 `images` 改成 `webcam`。
