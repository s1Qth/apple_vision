# 数据集来源说明 — apple_yolo_1k

## 来源
- 数据集：**SyneticAI/ApplesM5-Dataset**（`real-original` 子集，真实果园苹果图）
- 许可：**CC BY 4.0**
- 下载镜像：hf-mirror.com
- 图片与标注：1000 张图片 + 1000 个 YOLO 格式标注，一一对应

## 构成
- 原始图片：`data/raw/apple_57.jpg` ~ `apple_1056.jpg`（1000 张，640×640）
- 原始标注：`data/raw/labels/apple_57.txt` ~ `apple_1056.txt`（1000 个）
- 训练数据集（本目录）：9:1 划分，固定随机种子 42，可复现
  - `images/train` + `labels/train`：900 张
  - `images/val` + `labels/val`：100 张
- 标注格式：YOLO，单类 `0 = apple`，共 8748 个框，全部校验通过

## 训练配置
- 配置文件：`configs/apple_yolo_1k.yaml`
- 训练命令：
  ```bash
  cd C:\Users\asus\apple_vision
  conda activate apple
  python scripts/retrain_train.py --data configs/apple_yolo_1k.yaml --epochs 100 --device cpu
  ```
- 冒烟测试已通过：900 train / 100 val，0 corrupt，1 epoch mAP50=0.71

## 引用（论文/报告署名）
若用于论文或报告，请按数据集原作者要求标注来源（ApplesM5-Dataset, SyneticAI, CC BY 4.0）。
