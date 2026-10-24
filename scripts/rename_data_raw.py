import os
from pathlib import Path

def batch_rename_images(folder_path):
    # 将字符串路径转换为 Path 对象
    dir_path = Path(folder_path)
    
    if not dir_path.exists():
        print(f"❌ 错误：找不到文件夹 {folder_path}")
        return

    # 支持的图片后缀
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # 获取所有图片文件，并按文件名排序（保证重命名顺序不乱）
    image_files = sorted([f for f in dir_path.iterdir() if f.suffix.lower() in valid_extensions])

    if not image_files:
        print("⚠️ 该文件夹下没有找到图片。")
        return

    print(f"找到 {len(image_files)} 张图片，开始重命名...\n" + "-"*30)

    # 循环重命名
    for index, img_path in enumerate(image_files, start=1):
        # 构造新文件名：apple_1.jpg, apple_2.jpg ...
        new_name = f"apple_{index}.jpg"
        new_path = dir_path / new_name

        # 如果新文件名已经存在（比如之前跑过一次），为了防止覆盖，先跳过或报错
        if new_path.exists() and img_path.name != new_name:
            print(f"跳过: {img_path.name} (目标文件 {new_name} 已存在)")
            continue

        # 执行重命名
        try:
            img_path.rename(new_path)
            print(f"✅ 成功: {img_path.name} -> {new_name}")
        except Exception as e:
            print(f"❌ 失败: {img_path.name} (原因: {e})")

    print("-" * 30 + "\n🎉 全部重命名完成！")

if __name__ == "__main__":
    # 你的图片所在的文件夹路径
    # 注意：如果你的图片直接放在 data/raw 下，就填 "data/raw"
    TARGET_FOLDER = "data/raw" 
    
    batch_rename_images(TARGET_FOLDER)