import cv2
import numpy as np
from pathlib import Path

def process_image(img_path, output_dir):
    """处理单张图片的函数"""
    print(f"正在处理: {img_path.name} ...", end=" ")
    
    # 1. 读取图片
    img = cv2.imread(str(img_path))
    if img is None:
        print("读取失败！跳过。")
        return

    # 2. 转换色彩空间 BGR -> HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. 设定阈值 (正红、紫红、橙红)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([10, 70, 50])
    upper_orange = np.array([30, 255, 255])

    # 4. 获取掩膜并合并
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_o = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = mask1 | mask2 | mask_o

    # 5. 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 6. 保存结果图（保持和原图一样的名字，后缀加个 _mask）
    # 例如: first_apple.jpg -> first_apple_mask.jpg
    save_name = f"{img_path.stem}_mask.jpg"
    save_path = output_dir / save_name
    cv2.imwrite(str(save_path), mask)
    
    print(f"完成! 已保存为 {save_name}")

def main():
    # 定义输入和输出文件夹
    input_dir = Path("data/raw")
    output_dir = Path("outputs")
    
    # 确保输出文件夹存在
    output_dir.mkdir(exist_ok=True)

    # 查找输入文件夹里所有的 jpg 和 png 图片
    # 注意：为了防止 macOS 的 .DS_Store 等隐藏文件捣乱，过滤一下后缀
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in valid_extensions]

    if not image_files:
        print(f"警告：在 {input_dir} 文件夹里没有找到任何图片！请放入图片再试。")
        return

    print(f"找到 {len(image_files)} 张图片，开始批量处理...\n" + "-"*30)

    # 遍历处理每一张图片
    for img_path in image_files:
        process_image(img_path, output_dir)

    print("-"*30 + f"\n全部处理完成！请去 {output_dir} 文件夹查看结果。")

if __name__ == "__main__":
    main()