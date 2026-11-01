import numpy as np
import yaml

def pixel_to_camera_coords(u, v, depth, fx, fy, cx, cy):
    """将 2D 像素坐标 (u, v) 加上深度 (depth) 转换为 3D 坐标 (X, Y, Z)"""
    if depth <= 0:
        return None
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z

def load_handeye_x(yaml_path):
    """
    读取 handeye.yaml 中的旋转矩阵 R 和平移向量 t，拼接成 4x4 齐次矩阵 X
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    calib = config['calibration']
    R = np.array(calib['rotation_matrix'])
    t = np.array(calib['translation'])
    
    # 构造 4x4 矩阵
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = t
    return X

def camera_to_base(p_cam, X):
    """
    将相机坐标 (Xc, Yc, Zc) 转换到机械臂基座坐标 (Xb, Yb, Zb)
    """
    if p_cam is None:
        return None
    # 将 3D 坐标补 1 变成齐次坐标 [Xc, Yc, Zc, 1]
    p = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
    # 矩阵乘法
    p_base = X @ p
    # 返回前 3 个元素
    return p_base[:3]