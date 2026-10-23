import cv2
from pathlib import Path

# 1. 读图（用你之前改好的正确路径）
script_dir = Path(__file__).parent
img_path = script_dir.parent / "data" / "raw" / "first_apple.jpg"
img = cv2.imread(str(img_path))

if img is not None:  
    # 2. 打印形状
    print("彩色图形状:", img.shape)

    # 3. 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("灰度图形状:", gray.shape)

    # 4. 裁剪中间那块（假设你想看苹果的某一部分）
    # 注意：这里的数字不能超过你图片的宽高，不然切出来是空的！
    crop = img[200:400, 300:500]

    cv2.imshow("original", img)          # OpenCV 窗口显示（BGR，正常）
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 5. 把切下来的图存起来看看
    cv2.imwrite("outputs/crop_test.jpg", crop)
    print("裁剪完成！去 outputs 文件夹里看看切出来的图吧。")