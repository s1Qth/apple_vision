"""
distractor_suppressor.py —— 干扰物"一票否决"模块

作用：你的苹果模型（best.pt）把脸/人误认成苹果时，用第二个"干扰物模型"
      （如 yolov8n 的 COCO 全类、或专门的 yolov8n-face 脸模型）把落在
      干扰物上的苹果框删掉。这样不用自己拍数据，直接拿现成模型补漏洞。

设计要点：
- 完全独立，不改动你训好的苹果模型；只在检测结果外面"包一层过滤"。
- 默认用本地已有的 yolov8n.pt（只 block person 类 0），零下载，行为向后兼容。
- 开启 block_all=True 后，把干扰模型能识别的"所有类"都当干扰（自动排除名字
  含 apple 的类，避免真苹果被自己的 apple 框否决），用于突出苹果。
- 也可换成专门的脸模型 yolov8n-face.pt，此时 block_all 只会 block 它的 face 类。
"""

from ultralytics import YOLO


def box_iou(a, b):
    """计算两个 [x1,y1,x2,y2] 框的 IoU（交并比）。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class DistractorSuppressor:
    """
    干扰物否决器。

    weights      : 干扰物模型路径，默认 'yolov8n.pt'（本地已有）
    block_classes: 手动指定要"一票否决"苹果框的类别 id，默认 person=0。
                   block_all=False 时生效。
    block_all    : True = 自动把干扰模型能识别的所有类都当干扰（排除 apple）。
                   适合"只要被通用模型认成别的东西就删，突出苹果"。
    exclude_names: block_all 时要排除的类名（不分大小写），默认 ('apple',)。
                   这是防止真苹果被否决的关键，不要清空。
    conf         : 干扰模型的置信度阈值（越高越严）
    iou_thresh   : 苹果框与干扰框重叠超过这个 IoU 就删苹果框（0.3~0.5 常用）
    device       : 'cpu' 或 '0'(GPU)
    """

    def __init__(self, weights="yolov8n.pt", block_classes=(0,), conf=0.3,
                 iou_thresh=0.35, device="cpu", block_all=False,
                 exclude_names=("apple",)):
        self.weights = weights
        self.block_all = bool(block_all)
        self.exclude_names = tuple(str(n).lower() for n in exclude_names)
        self.block_classes = set(int(c) for c in block_classes)
        self.conf = float(conf)
        self.iou_thresh = float(iou_thresh)
        self.device = device
        self.model = None
        self._names = None

    def _resolve_block_classes(self):
        """根据当前模式解析最终要否决的类别 id 集合（不跑模型，便于单测）。"""
        if self.block_all and self._names is not None:
            return {
                int(c) for c, name in self._names.items()
                if str(name).lower() not in self.exclude_names
            }
        return set(int(c) for c in self.block_classes)

    def load(self):
        """延迟加载干扰模型（第一次用时才加载，省内存）。"""
        if self.model is None:
            self.model = YOLO(self.weights)
            self._names = self.model.names
        self.block_classes = self._resolve_block_classes()
        return self

    def summary(self):
        """打印干扰模型信息，方便确认类别 id 和模式对不对。"""
        self.load()
        names = {c: self._names[c] for c in sorted(self.block_classes) if c in self._names}
        mode = "全部物品(已排除 apple)" if self.block_all else "手动指定类别"
        print(f"[干扰模型] 权重={self.weights}")
        print(f"          模式={mode}")
        print(f"          模型总类别数={len(self._names)}，被否决类别数={len(names)}")
        if len(names) <= 12:
            print(f"          被否决类别 id->名称: {names}")
        else:
            sample = dict(list(names.items())[:8])
            print(f"          被否决类别(前8个): {sample} ... 共 {len(names)} 类")
        return names

    def detect_distractors(self, frame):
        """在画面上跑干扰模型，只返回属于被否决类别的框。"""
        self.load()
        res = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)
        boxes = []
        for r in res:
            for b in r.boxes:
                cls = int(b.cls[0])
                if cls in self.block_classes:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                    boxes.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(b.conf[0]),
                        "cls": cls,
                        "label": self._names.get(cls, str(cls)),
                    })
        return boxes

    def filter_apples(self, apple_dets, distractors):
        """
        纯决策：给定苹果框 + 干扰框，返回 (kept, removed)。
        不跑模型，便于单元测试与复用。
        """
        kept, removed = [], []
        for d in apple_dets:
            x1, y1, x2, y2 = d["bbox"]
            hit = any(box_iou((x1, y1, x2, y2), tuple(db["bbox"])) >= self.iou_thresh
                      for db in distractors)
            (removed if hit else kept).append(d)
        return kept, removed

    def suppress(self, apple_dets, frame):
        """
        对苹果检测结果做否决（先跑干扰模型，再做决策）。

        返回 (kept, removed, distractors)：
          kept       : 保留的苹果框（列表）
          removed    : 被否决的苹果框（列表，便于统计/调试）
          distractors: 检测到的干扰框（可用于画红框提示）
        """
        distractors = self.detect_distractors(frame)
        kept, removed = self.filter_apples(apple_dets, distractors)
        return kept, removed, distractors
