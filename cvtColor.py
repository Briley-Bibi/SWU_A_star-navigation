import cv2
import numpy as np

# 定义RGB颜色
color_white = np.uint8([[[255, 255, 255]]])  # 纯白色
color_almost_white = np.uint8([[[229, 229, 230]]])  # 几乎白色

# 将RGB转换为HSV
hsv_white = cv2.cvtColor(color_white, cv2.COLOR_RGB2HSV)
hsv_almost_white = cv2.cvtColor(color_almost_white, cv2.COLOR_RGB2HSV)

# 打印HSV值
print("HSV for White: ", hsv_white)
print("HSV for Almost White: ", hsv_almost_white)

# 设置阈值
# 通常情况下，我们只关心饱和度和明度
lower_bound = np.array([0, 0, np.min([hsv_white[0][0][2], hsv_almost_white[0][0][2]])])
upper_bound = np.array([180, np.max([hsv_white[0][0][1], hsv_almost_white[0][0][1]]), 255])

# 打印阈值
print("Lower Bound for White: ", lower_bound)
print("Upper Bound for White: ", upper_bound)

# 定义新的景观颜色：9CBD8F, C2DEA6, 9CC2CD
# 将这些颜色转换为HSV

# 9CBD8F
color_1 = np.uint8([[[156, 189, 143]]])  # RGB = [156, 189, 143]
hsv_1 = cv2.cvtColor(color_1, cv2.COLOR_RGB2HSV)

# C2DEA6
color_2 = np.uint8([[[194, 222, 166]]])  # RGB = [194, 222, 166]
hsv_2 = cv2.cvtColor(color_2, cv2.COLOR_RGB2HSV)

# 9CC2CD
color_3 = np.uint8([[[156, 194, 205]]])  # RGB = [156, 194, 205]
hsv_3 = cv2.cvtColor(color_3, cv2.COLOR_RGB2HSV)

# 打印这三种颜色的HSV值
print("HSV for 9CBD8F: ", hsv_1)
print("HSV for C2DEA6: ", hsv_2)
print("HSV for 9CC2CD: ", hsv_3)

# 通过查看它们的HSV范围，可以设置大概的阈值
# 设置阈值范围，这里假设我们根据这些颜色的HSV值设置一个适当的范围

# 景观颜色的HSV范围
lower_landscape_1 = np.array([hsv_1[0][0][0] - 10, 50, 50])  # 色相偏移 + 饱和度和亮度的下限
upper_landscape_1 = np.array([hsv_1[0][0][0] + 10, 255, 255])  # 色相偏移 + 饱和度和亮度的上限

lower_landscape_2 = np.array([hsv_2[0][0][0] - 10, 50, 50])  # 色相偏移 + 饱和度和亮度的下限
upper_landscape_2 = np.array([hsv_2[0][0][0] + 10, 255, 255])  # 色相偏移 + 饱和度和亮度的上限

lower_landscape_3 = np.array([hsv_3[0][0][0] - 10, 50, 50])  # 色相偏移 + 饱和度和亮度的下限
upper_landscape_3 = np.array([hsv_3[0][0][0] + 10, 255, 255])  # 色相偏移 + 饱和度和亮度的上限

# 创建景观掩膜
image = cv2.imread('images/original.png')  # 假设你已经有了一个名为 'image' 的图像变量
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 创建白色掩膜
mask_white = cv2.inRange(hsv_image, lower_bound, upper_bound)

# 创建景观掩膜
mask_landscape_1 = cv2.inRange(hsv_image, lower_landscape_1, upper_landscape_1)
mask_landscape_2 = cv2.inRange(hsv_image, lower_landscape_2, upper_landscape_2)
mask_landscape_3 = cv2.inRange(hsv_image, lower_landscape_3, upper_landscape_3)

# 合并景观掩膜
mask_landscape = cv2.bitwise_or(mask_landscape_1, mask_landscape_2)
mask_landscape = cv2.bitwise_or(mask_landscape, mask_landscape_3)

# 结果图像初始化
result = np.zeros_like(image)

# 将白色区域处理为白色
result[mask_white != 0] = [255, 255, 255]

# 将景观区域处理为绿色 (85FFAF的BGR值是 [85, 255, 175])
result[mask_landscape != 0] = [85, 255, 175]  # 绿色 BGR

# 显示结果
cv2.imshow('Original', image)
cv2.imshow('Mask White', mask_white)
cv2.imshow('Mask Landscape', mask_landscape)
cv2.imshow('Result', result)

# 等待按键后关闭窗口
cv2.waitKey(0)

# 保存图像
cv2.imwrite('images/saved_original.png', image)
cv2.imwrite('images/saved_mask_white.png', mask_white)
cv2.imwrite('images/saved_mask_landscape.png', mask_landscape)
cv2.imwrite('images/saved_result.png', result)

# 关闭所有窗口
cv2.destroyAllWindows()