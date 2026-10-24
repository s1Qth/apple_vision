import cv2
import numpy as np
from pathlib import Path

def detect_apples(img, min_area=2000):
    """
    输入一张 BGR 原图，返回画了圆圈的图和检测到的苹果数量
    """
    # 1. 转 HSV (Day 5 逻辑)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. 设定阈值 (建议保留你昨天调好的参数)
    # 注意：为了适应更多图，这里我把 S 的下限提高了，并加大了 kernel
    lower_red1 = np.array([0, 120, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([10, 120, 50])
    upper_orange = np.array([30, 255, 255])

    # 3. 获取掩膜并合并
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_o = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = mask1 | mask2 | mask_o

    # 4. 形态学去噪 (加大了 kernel 到 11x11 滤除更多噪点)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 5. 找轮廓 (Day 6 核心)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()
    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 过滤小噪点
        if area < min_area:
            continue
        
        # 计算周长
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
            
        # 计算圆度 (4 * pi * 面积 / 周长^2)。完美的圆是 1.0，形状越奇怪值越小。
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 🌟 只有圆度大于 0.6 的，才认为是苹果
        if circularity < 0.6:
            continue
        
        # 找到最小外接圆
        (x, y), r = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(r)

        # 过滤掉半径太小的假圆 (防止把一小撮白叶子当苹果)
        if radius < 10:
            continue
        
        # 🌟 新增：过滤掉那种包住半个屏幕的幽灵大圆
        if radius > 300: # 这个数字根据你的图片大小调，差不多是真实苹果最大半径
            continue
        
        # 画绿色的圆
        cv2.circle(result, center, radius, (0, 255, 0), 2)
        
        # 在圆的上方写上编号
        cv2.putText(result, f"#{count}", (center[0] - radius, center[1] - radius - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        count += 1

    return result, count

def main():
    input_dir = Path("data/raw")
    output_dir = Path("outputs/day6_contours")
    output_dir.mkdir(exist_ok=True)

    # 过滤出所有图片文件
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in valid_extensions]

    if not image_files:
        print(f"警告：在 {input_dir} 文件夹里没有找到任何图片！请放入图片再试。")
        return

    print(f"找到 {len(image_files)} 张图片，开始检测...\n" + "-"*40)

    for img_path in image_files:
        print(f"正在处理: {img_path.name} ...", end=" ")
        
        img = cv2.imread(str(img_path))
        if img is None:
            print("读取失败！跳过。")
            continue

        # 调用检测函数 (你可以在这里修改 min_area 的值来调整灵敏度)
        result_img, apple_count = detect_apples(img, min_area=2000)

        # 保存结果图
        save_name = f"{img_path.stem}_detect.jpg"
        save_path = output_dir / save_name
        cv2.imwrite(str(save_path), result_img)
        
        print(f"完成! 检测到 {apple_count} 个苹果，已保存至 {save_path}")

    print("-"*40 + "\n全部处理完成！去 outputs 文件夹看结果吧。")

if __name__ == "__main__":
    main()