"""
Day 9 实操：YOLOv8 概念 + 跑通预训练模型推理
============================================
目标：加载官方预训练 YOLOv8，对一张图做推理，把检测框画出来。
"""
import argparse
import os
import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/raw/test.jpg")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise SystemExit(f"找不到图片: {args.image}（放一张图到 data/raw/ 再试）")

    # 1) 加载官方预训练模型（首次会自动下载 yolov8n.pt，约 6MB）
    model = YOLO("yolov8n.pt")

    # 2) 推理：输入一张图，返回结果列表（这里只有一张）
    results = model.predict(args.image, conf=0.25)
    r = results[0]

    # 3) 读原图，把检测框画上去
    img = cv2.imread(args.image)
    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()      # 左上/右下角坐标(像素)
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            conf = box.conf[0].item()                  # 置信度 0~1
            cls = int(box.cls[0].item())               # 类别 id
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"检测到 {len(r.boxes)} 个目标")
    else:
        print("这张图里预训练模型没检出东西")

    cv2.imshow("yolo infer", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/day9_infer.jpg", img)
    print("结果保存到 outputs/day9_infer.jpg")


if __name__ == "__main__":
    main()
