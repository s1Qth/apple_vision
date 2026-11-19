# 苹果检测模型「补数据 + 重训」完整流程（R1–R7）

> 目标：根治你遇到的「脸/手被识别成苹果」误检问题。
> 配套脚本：`scripts/retrain_*.py`（已写好，本环境已真跑验证 5/6，训练脚本冒烟通过）。
> 适用对象：零基础、慢工出细活、求不出错。

---

## 0. 为什么要重训（先搞懂根因，才不白忙）

你现在的模型（`runs/detect/train/weights/best.pt`）是怎么来的：

- 只用了 **56 张图**（40 训练 + 16 验证）训练；
- **单类 apple**，**完全没有负样本**（没有人、手、红物、其他水果、纯背景当"背景"学过）；
- mAP@0.5 ≈ 0.69，在这么少的数据上已经严重**过拟合**。

结果就是：模型只"见过"苹果的长相，没见过"不是苹果但有点像"的东西。于是看到**圆、偏肤色**的（脸、眼镜反光）就乱判成苹果——这就是你截图里 `apple 0.55 / apple 0.32` 的来源。

**关键点**：之前把 `--conf` 从 0.3 提到 0.5 + 长宽比过滤，能"压住"脸误检，但那是治标。要**治本**，必须让模型学会"什么不是苹果"，方法是：

1. 补**更多、更多样**的苹果正样本（不同光照/距离/遮挡/品种）；
2. 加**困难负样本**（人脸、手、红物、其他水果、纯背景）——让模型明确学到"这些不是苹果"；
3. 重训，用更合理的超参 + 在线增强。

> 调阈值 = 把错误答案藏起来；补数据+重训 = 让模型本来就不出错。

---

## 1. 整体路线图

| 阶段 | 名称 | 谁来做 | 核心产出 |
|---|---|---|---|
| **R1** | 数据体检 | 脚本自动 | 弄清差在哪（少/单一/漏标） |
| **R2** | 补正样本 + 标注 | **你拍+标**（脚本切分） | +几十~上百张苹果图及标签 |
| **R3** | 数据增强 | 脚本自动 | 把样本量再翻 2~4 倍 |
| **R4** | 困难负样本挖掘 | **你收集** + 脚本注册 | 人脸/手等负样本进训练集 |
| **R5** | 重训模型 | 脚本跑 | 新的 `best.pt` |
| **R6** | 评估对比 | 脚本自动 | 旧 vs 新 mAP + **人脸误检数** |
| **R7** | 上线替换 | 脚本+你确认 | 接回 Day26 实时识别 |

建议节奏：**R1 今天就能跑**；R2/R4 是你用 USB 摄像头拍素材的时间（几天）；R3/R5/R6/R7 在你素材齐了之后连续做。

---

## 2. 各阶段详细做法

### R1 · 数据体检（`retrain_audit.py`）
```bash
python scripts/retrain_audit.py --data configs/dataset.yaml
```
- 看报告里的「总图像数」「框尺寸分布」「有图无标注」。
- **验收**：报告出炉，明确短板（目前会是"⚠️ 数据偏少"）。
- 现在跑出来的结论：**56 张太少 + 框尺寸跨度大（最小几乎看不见、最大占满画面）**，需要补数据。

### R2 · 补正样本 + 标注（你来做，脚本帮你切分）
**拍摄规范**（越多越多样越好，目标 +60~150 张）：
- 不同**光照**：阴天 / 太阳直照 / 逆光；
- 不同**距离**：近（苹果占大半屏）/ 中 / 远（小苹果）；
- **遮挡**：叶子遮住 1/3 的苹果；
- 不同**品种颜色**：红富士 / 青苹果 / 黄苹果；
- **多果同框** + **单果特写**都要有；
- 背景里**故意放非苹果的圆物**（红气球、橘子），让模型学区分；
- 不同**角度**（平视 / 俯视）。

**标注工具**：`pip install labelImg`，打开后：
- 选 YOLO 格式，类名为 `apple`（类号 0）；
- 每张图里**所有苹果**都框上，框要贴边别太松。

**并入数据集**（脚本自动 8:2 切分 + 格式校验）：
```bash
# 先预览（不复制），确认无误
python scripts/retrain_prepare.py --src 你拍的文件夹/
# 确认无误后真正复制进数据集
python scripts/retrain_prepare.py --src 你拍的文件夹/ --copy
```
- **验收**：再跑 R1 体检，train 图像数变多、无"有图无标注"。

### R3 · 数据增强（`retrain_augment.py`，脚本自动）
```bash
python scripts/retrain_augment.py --data configs/dataset.yaml --split train --per-image 3
```
- 对每张训练图生成 3 张增强（水平翻转 / 亮度抖动 / 模糊 / 加噪 / ±12° 旋转，框同步变换）。
- 注：YOLO 训练时本身也会在线增强（mosaic/hsv/flip），本步是**额外增加原始样本量**，对 56 张这种极小数据尤其有用。
- **别过度**：`--per-image` 2~4 即可，太多假数据反而乱。
- **验收**：体检里 train 图像数成倍增加；抽查几张增强图+标签对齐。

### R4 · 困难负样本挖掘（`retrain_hard_negatives.py`，根治误检的关键）
**你收集**"没有苹果但容易误认"的图，放进一个文件夹，例如：
- **人脸特写**（你/同学的正脸、侧脸、戴眼镜的脸）；
- **手**；
- 红/黄/绿球、橘子、西红柿等其他圆物；
- 纯背景、树叶、天空。

**注册进训练集**（脚本自动，70% 进训练、30% 留作评估）：
```bash
python scripts/retrain_hard_negatives.py --neg 你收集的负样本文件夹/   # 预览
python scripts/retrain_hard_negatives.py --neg 你收集的负样本文件夹/ --copy
```
- 这些图会带**空标签**进训练集 → 模型学到"这是背景，不是苹果"。
- 留出的 30% 会存到 `outputs/face_test/`，**不参与训练**，专门给 R6 量化"脸还能不能被认成苹果"。
- **验收**：训练集图像数再增加；`outputs/face_test/` 里有若干人脸图。

### R5 · 重训（`retrain_train.py`，脚本跑）
```bash
# 普通重训（本机无显卡用 cpu，有显卡把 --device 0）
python scripts/retrain_train.py --data configs/dataset.yaml --epochs 100 --imgsz 640 --device cpu

# 想更抗误检、不怕慢：换更大模型
python scripts/retrain_train.py --model yolov8s.pt --epochs 120 --device cpu

# 中断后续训
python scripts/retrain_train.py --resume runs/detect/retrain/weights/last.pt
```
- 推荐超参（已写进脚本默认）：`epochs 100, patience 30(早停), imgsz 640, batch 16, lr0 0.001, cos_lr, mosaic 1.0`。
- 训练完新权重在 `runs/detect/retrain/weights/best.pt`。
- **验收**：loss 下降、mAP 上升；生成新 `best.pt`；终端打印 mAP 数值。

### R6 · 评估对比（`retrain_eval.py`，脚本自动，重点看人脸误检）
```bash
# 旧模型 vs 新模型，都跑一遍对比
python scripts/retrain_eval.py --weights runs/detect/train/weights/best.pt
python scripts/retrain_eval.py --weights runs/detect/retrain/weights/best.pt
```
- 报告两件事：① val 上的 `mAP@0.5`；② **负样本测试集（人脸等）上的误检(假苹果)总数**。
- **验收（核心指标）**：新模型的「负样本误检数」明显低于旧模型，目标 **≤ 个位数甚至 0**。

### R7 · 上线替换（接回 Day26 实时识别）
```bash
# 把新权重指给检测器：改 scripts/day26_realtime_detect.py 的默认 --weights
# 或直接覆盖（二选一）
copy runs/detect/retrain/weights/best.pt runs/detect/train/weights/best.pt
```
- 然后回归 Day26 实时识别（接 USB 摄像头），看脸还被误认不：
```bash
python scripts/day26_realtime_detect.py --source webcam --cam-id 1
```
- **验收**：实时画面里脸/手不再被框成苹果，真苹果仍稳定检出。

---

## 3. 误检/漏检还严重时怎么办（决策表）

| 现象 | 原因 | 对策 |
|---|---|---|
| 人脸仍被认成苹果 | 负样本不够 / 阈值还低 | R4 加更多人脸负样本；`--conf` 提到 0.6~0.7；保留长宽比过滤 |
| 真苹果漏检（该框没框） | 正样本不足 / 太严 | R2 补更多苹果图；`--conf` 降到 0.4；R3 增强 |
| val mAP 低 | 数据量/多样性不够 | R2 继续补；延长 epoch；换 `yolov8s` |
| 训练 loss 不降/NaN | 标签错 / 显存炸 | 跑 `retrain_audit.py` 查漏标；`imgsz` 降到 416、`batch` 降到 8 |
| 过拟合（train 好 val 差） | 数据太少 | R2+R3 增数据；开 `mosaic`；适当早停 |

---

## 4. 工作量与顺序建议

- **今天**：跑 R1（已可跑），开始 R2 拍苹果 + R4 拍人脸负样本（并行收集）。
- **素材齐了**（建议苹果 ≥100 张 + 人脸/干扰 ≥30 张）：R3 增强 → R5 重训 → R6 评估 → R7 上线。
- 整个自动部分（R1/R3/R5/R6）几行命令；**真正的瓶颈是你拍素材的时间**——这也是为什么之前说"时间花久点无所谓"。

---

## 5. 与现有 Day24–26 的关系

- Day24–26 是**实时识别管线**（相机层→检测器→主循环），不依赖本流程；
- 本流程产出新的 `best.pt` 后，只要替换权重，Day26 立刻用上更好的模型；
- 之前加的「长宽比过滤 + 高 conf 默认」**建议保留**作为兜底，和重训互补，双保险。

---

## 脚本清单（均在 `scripts/`）

| 脚本 | 阶段 | 作用 |
|---|---|---|
| `retrain_audit.py` | R1 | 数据体检 |
| `retrain_prepare.py` | R2 | 并入新数据 + 8:2 切分 + 标签校验 |
| `retrain_augment.py` | R3 | 离线增强（框同步变换） |
| `retrain_hard_negatives.py` | R4 | 负样本注册进训练集 + 生成人脸评估集 |
| `retrain_train.py` | R5 | 重训启动器（超参/续训/清缓存） |
| `retrain_eval.py` | R6 | val mAP + 人脸误检计数对比 |
