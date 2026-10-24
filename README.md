# 🍎 Apple Vision - 苹果视觉检测项目 (P1 阶段)

> **作者**：GitHub: @s1Qth
> **当前进度**：第一周 (Week 1) - 传统计算机视觉 (OpenCV) 基线打通
> **项目目标**：从零搭建一个基于视觉的苹果采摘机器人识别模块，最终实现从“图像输入”到“3D 空间坐标输出”的全流程。

---

## 📖 1. 项目简介
本项目是苹果采摘机器人视觉系统的第一阶段（P1）实践。在第一周中，我们暂时不引入深度学习，而是使用**传统计算机视觉技术（OpenCV）**，通过 HSV 色彩空间阈值分割和轮廓检测，实现基础的红苹果识别与计数。

### 📌 本周核心成果：
- [x] 搭建标准化的 Python 项目目录结构（配置与代码分离、数据/权重不入库）。
- [x] 配置 Conda 虚拟环境与 Git/GitHub 版本控制（成功解决 SSH 22 端口被封锁的痛点，改用 HTTPS 通道）。
- [x] 掌握 OpenCV 基础操作（读图、BGR/RGB 转换、灰度图）。
- [x] 实现基于 HSV 色彩空间的红色区域提取（掩膜生成）。
- [x] 实现基于形态学（开/闭运算）的图像降噪。
- [x] 实现基于轮廓检测（`findContours`）与最小外接圆（`minEnclosingCircle`）的苹果画圈与计数。

---

## 📂 2. 目录结构
本项目采用工程化结构，严格区分源码、配置、数据与输出：

```text
apple_vision/
├── README.md                  # 项目说明文档
├── environment.yml            # conda 环境定义
├── .gitignore                 # Git 忽略文件（忽略数据、模型、环境等）
├── configs/                   # 配置文件（相机内参、数据集 yaml）
├── src/                       # 核心源码模块
│   ├── capture/               # 图像采集
│   ├── detection/             # 检测算法（传统CV/YOLO）
│   ├── localization/          # 3D定位
│   └── utils/                 # 公共工具函数
├── scripts/                   # 可运行入口脚本
│   ├── day5_hsv_threshold.py  # Day5: 单图/批量 HSV 阈值分割
│   └── day6_contours.py       # Day6: 批量轮廓检测与画圆计数
├── data/                      # 数据集目录（被 .gitignore 忽略）
│   └── raw/                   # 存放原始图片
├── models/                    # 训练权重 .pt（被忽略）
├── outputs/                   # 运行结果图/掩膜图存放地
└── notebooks/                 # 探索性 Notebook