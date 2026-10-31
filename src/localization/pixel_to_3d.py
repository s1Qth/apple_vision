def pixel_to_camera_coords(u, v, depth, fx, fy, cx, cy):
    """
    将 2D 像素坐标 (u, v) 加上深度 (depth) 转换为 3D 坐标 (X, Y, Z)
    """
    # 防止深度为 0 导致除零报错
    if depth <= 0:
        return None
        
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    return X, Y, Z