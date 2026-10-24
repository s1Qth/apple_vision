"""
Day 8 实操：PyTorch 初体验（张量 + 自动求导 + 训练四步）
========================================================
目标：用最简单的"学一条直线 y = w*x + b"玩具例子，把深度学习训练的四步
      （前向 → 算损失 → 反向 → 更新）跑通并看懂。
真正训练 YOLO 时，ultralytics 内部就是这套循环，只是模型更复杂。

运行:
    conda activate apple
    python scripts/day8_pytorch_basics.py
"""
import torch
import torch.optim as optim


def main():
    # ---------- 0. 环境检查 ----------
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 可用:", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用设备:", device)

    # ---------- 1. 张量 Tensor：像 numpy，但能上 GPU ----------
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    print("张量相加:", a + b)                 # 和 numpy 一样直观
    a_gpu = a.to(device)                      # 搬到 GPU（若没有则还是 CPU）
    print("张量所在设备:", a_gpu.device)

    # ---------- 2. 自动求导 autograd：梯度自动算 ----------
    x = torch.tensor(2.0, requires_grad=True)   # 告诉 PyTorch：要算 x 的梯度
    y = x ** 2 + 3 * x                          # 构造一个小计算图
    y.backward()                                # 反向传播，自动算出 dy/dx
    print("dy/dx 在 x=2 处 =", x.grad.item())   # 理论值 2x+3 = 7

    # ---------- 3. 玩具训练：学出 y = 3x + 1 ----------
    # 造"真实"数据：y = 3x + 1 + 噪声
    torch.manual_seed(0)
    X = torch.linspace(0, 1, 100).reshape(-1, 1)        # 100 个样本，形状 (100,1)
    true_w, true_b = 3.0, 1.0
    Y = true_w * X + true_b + 0.1 * torch.randn(100, 1)  # 加点噪声

    # 模型就是 y = w*x + b，w,b 是我们要学的参数
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    optimizer = optim.SGD([w, b], lr=0.1)     # 优化器：负责"更新参数"

    print("\n开始训练（看 loss 逐步下降）...")
    for epoch in range(200):
        # (1) 前向：用当前 w,b 算预测值
        Y_pred = w * X + b
        # (2) 算损失：预测和真值的均方误差 MSE
        loss = ((Y_pred - Y) ** 2).mean()
        # (3) 反向 + 更新
        optimizer.zero_grad()      # 清空上一步累积的梯度（重要！）
        loss.backward()            # 自动算 loss 对 w,b 的梯度
        optimizer.step()           # 沿梯度反方向更新 w,b

        if epoch % 40 == 0 or epoch == 199:
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  "
                  f"w={w.item():.3f}  b={b.item():.3f}")

    print(f"\n学出的参数: w≈{w.item():.3f} (真值3.0), b≈{b.item():.3f} (真值1.0)")
    print("如果 w 接近 3、b 接近 1，说明训练四步跑通了！")


if __name__ == "__main__":
    main()