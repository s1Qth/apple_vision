"""
Day15 手眼标定（眼在手外 / eye-to-hand）模拟演练
================================================
目的：在没有真实机械臂和相机的情况下，用"造出来"的数据把
      手眼标定的数学链路跑通，确认它是对的。

真实场景里你要收集两类数据（每个机器人姿态 i 各一份）：
  - g_i = 机械臂给你的"夹爪在基座下的位姿"  T_base->gripper
          （来自正运动学 / 示教器读关节角算出来）
  - c_i = 相机拍到标定板后，solvePnP 算出的"标定板在相机下的位姿"
          T_cam->target
  我们要解的就是固定不变的 X = 基座 -> 相机  T_base->cam
  有了 X，相机看到的任意点都能换算成机械臂听得懂的基座坐标。

这里我们用一个"真实 X"去造数据，再让算法反推，看能不能还原出来。

说明：本脚本优先用 OpenCV 的 cv2.calibrateHandEye；若你的 OpenCV 版本
     没有该函数（极少数 5.x 精简构建），会自动改用内置的纯 numpy 求解器，
     保证一定能跑通。
"""

import numpy as np


# ---------- 4x4 变换矩阵小工具 ----------
def translate(dx, dy, dz):
    T = np.eye(4)
    T[:3, 3] = [dx, dy, dz]
    return T


def rotate_axis(axis, angle):
    """绕任意轴旋转 angle(弧度)，手写 Rodrigues，避免引入额外依赖。"""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    return T


def inv(T):
    """4x4 刚体变换求逆：R^T 和 -R^T * t。"""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def to_rt(T):
    """从 4x4 拆出 (R, t)，符合 OpenCV 的输入习惯。"""
    return T[:3, :3].copy(), T[:3, 3].copy()


# ---------- 内置纯 numpy 手眼求解器（AX=XB）----------
def _quat_from_R(R):
    """旋转矩阵 -> 单位四元数 [w, x, y, z]。"""
    w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2 * (1 if R[2, 1] > R[1, 2] else -1)
    y = np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2 * (1 if R[0, 2] > R[2, 0] else -1)
    z = np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2 * (1 if R[1, 0] > R[0, 1] else -1)
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _R_from_quat(q):
    """单位四元数 [w,x,y,z] -> 旋转矩阵。"""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),   1 - 2*(x*x + y*y)]])


def _L(q):
    """左乘矩阵：L(q) p = q ⊗ p。"""
    w, x, y, z = q
    return np.array([[w, -x, -y, -z],
                     [x,  w, -z,  y],
                     [y,  z,  w, -x],
                     [z, -y,  x,  w]])


def _R(q):
    """右乘矩阵：R(q) p = p ⊗ q。"""
    w, x, y, z = q
    return np.array([[w, -x, -y, -z],
                     [x,  w,  z, -y],
                     [y, -z,  w,  x],
                     [z,  y, -x,  w]])


def solve_handeye_numpy(R_gb, t_gb, R_tc, t_tc):
    """
    输入：N 组绝对位姿
      R_gb, t_gb : 夹爪在基座下的位姿  T_base->gripper
      R_tc, t_tc : 标定板在相机下的位姿  T_cam->target
    返回：R_cam2base, t_cam2base = X = T_base->cam
    原理：构造相对运动 A_ij=g_j*g_i^-1, B_ij=c_j*c_i^-1，解 A X = X B。
          旋转用四元数法（避免矩阵向量化的符号坑），平移用最小二乘。
    """
    n = len(R_gb)
    A_list, B_list = [], []
    M = np.zeros((4, 4))            # 四元数法旋转系统
    for i in range(n):
        for j in range(i + 1, n):
            R_A = R_gb[j] @ R_gb[i].T
            t_A = t_gb[j] - R_A @ t_gb[i]
            R_B = R_tc[j] @ R_tc[i].T
            t_B = t_tc[j] - R_B @ t_tc[i]
            qA = _quat_from_R(R_A)
            qB = _quat_from_R(R_B)
            C = _L(qA) - _R(qB)     # (L(qA) - R(qB)) q_x = 0
            M += C.T @ C
            A_list.append((R_A, t_A))
            B_list.append((R_B, t_B))
    # 旋转：M 最小特征向量 -> 四元数
    w, v = np.linalg.eigh(M)
    qx = v[:, 0]
    qx = qx / np.linalg.norm(qx)
    Rx = _R_from_quat(qx)
    # 平移：(R_A - I) t_x = R_x t_B - t_A
    A_mat, b_vec = [], []
    for (R_A, t_A), (R_B, t_B) in zip(A_list, B_list):
        A_mat.append(R_A - np.eye(3))
        b_vec.append(Rx @ t_B - t_A)
    tx, *_ = np.linalg.lstsq(np.vstack(A_mat), np.concatenate(b_vec), rcond=None)
    return Rx, tx


# ---------- 1. 设定"真实"的基座->相机变换 X（这是我们要恢复的目标）----------
# 含义：相机在基座正前方 1.2m，右偏 0.30m，上偏 0.10m，并绕 Y 轴转了 15°
X_true = (translate(0.0, 0.0, 1.2)
          @ rotate_axis([0, 1, 0], np.deg2rad(15))
          @ translate(0.30, 0.10, 0.0))

# 标定板相对夹爪的固定位姿 m：板子装在夹爪正前方 10cm 处
m = translate(0.0, 0.0, 0.10)

# ---------- 2. 造 N 组机器人位姿，并据此算出相机看到的板子位姿 ----------
np.random.seed(42)
N = 15
R_gripper2base, t_gripper2base = [], []   # 输入1：夹爪在基座下的位姿
R_target2cam, t_target2cam = [], []       # 输入2：板子在相机下的位姿

for i in range(N):
    g = (translate(np.random.uniform(-0.3, 0.3),
                   np.random.uniform(-0.3, 0.3),
                   np.random.uniform(0.4, 0.8))
         @ rotate_axis([0, 0, 1], np.random.uniform(-0.5, 0.5))
         @ rotate_axis([1, 0, 0], np.random.uniform(-0.3, 0.3)))
    Rg, tg = to_rt(g)
    R_gripper2base.append(Rg)
    t_gripper2base.append(tg)

    T_base_target = g @ m                       # 板子随夹爪动
    c = inv(X_true) @ T_base_target             # 相机看到板子：T_cam->target
    Rc, tc = to_rt(c)
    R_target2cam.append(Rc)
    t_target2cam.append(tc)

# ---------- 3. 求解 X（优先 OpenCV，否则用内置 numpy 求解器）----------
backend = "OpenCV"
try:
    import cv2
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI)
except (AttributeError, ImportError):
    backend = "纯 numpy 求解器（OpenCV 无 calibrateHandEye，已自动兜底）"
    R_cam2base, t_cam2base = solve_handeye_numpy(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam)

X_est = np.eye(4)
X_est[:3, :3] = R_cam2base
X_est[:3, 3] = t_cam2base

print("使用后端:", backend)
print("\n真实 X (base->cam):\n", np.round(X_true, 4))
print("估算 X (base->cam):\n", np.round(X_est, 4))

# ---------- 4. 验证：拿一个苹果点，对比真实/估算换算到基座的结果 ----------
p_cam = np.array([0.20, -0.10, 0.80, 1.0])     # 相机测得苹果在相机坐标下
p_base_true = X_true @ p_cam
p_base_est = X_est @ p_cam
print("\n苹果在基座坐标下 (真实):", np.round(p_base_true[:3], 4))
print("苹果在基座坐标下 (估算):", np.round(p_base_est[:3], 4))
print("换算误差(米):", round(float(np.linalg.norm(p_base_true[:3] - p_base_est[:3])), 8))
