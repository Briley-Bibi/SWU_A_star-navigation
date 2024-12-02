
from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_edt, convolve

def calculate_environmental_quality(image_path):
    # 加载图像
    img = Image.open(image_path).convert('L')
    img_data = np.array(img)

    # 调整景观掩码的阈值
    landscape_mask = img_data > 200

    # 使用距离变换计算每个像素到最近的景观区域的距离
    distances = distance_transform_edt(~landscape_mask)

    # 修改距离的映射范围
    max_effective_distance = 5
    distances = np.clip(distances, 0, max_effective_distance)
    normalized_quality = 1 - distances / max_effective_distance

    # 空间平滑
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    quality_smoothed = convolve(normalized_quality, kernel, mode='nearest')

    # 创建热图
    heatmap = np.zeros((img_data.shape[0], img_data.shape[1], 3), dtype=np.uint8)

    # 定义颜色等级
    colors = np.array([
        [0, 127, 0],      # 深绿色
        [127, 223, 127],  # 中等绿色
        [191, 239, 191],  # 浅绿色
        [255, 255, 255]  # 白色  
    ])

    # 为所有区域分配颜色
    thresholds = [0.75, 0.5, 0.25, 0]
    for i, threshold in enumerate(thresholds):
        if i == 0:
            mask = quality_smoothed >= threshold
        else:
            mask = (quality_smoothed < thresholds[i-1]) & (quality_smoothed >= threshold)
        heatmap[mask] = colors[i]

    heatmap_img = Image.fromarray(heatmap)

    return heatmap_img, quality_smoothed
# Example usage
heatmap_img, quality_array = calculate_environmental_quality("images\saved_mask_landscape.png")
heatmap_img.show()  # Display the heatmap
# 保存热图
heatmap_img.save("images\landscape_heatmap.png")

# 保存质量数组到文件
np.save("images\landscape_array.npy", quality_array)
# 保存数组到txt文件
np.savetxt("images\landscape_array.txt", quality_array, fmt='%.3f')



