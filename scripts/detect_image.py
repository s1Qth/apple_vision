import cv2
import matplotlib.pyplot as plt

img = cv2.imread("apple.jpg")        # 读图（失败返回 None，记得判空）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 转灰度

cv2.imshow("original", img)          # OpenCV 窗口显示（BGR，正常）
cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 用 matplotlib 显示时要转 RGB：
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()