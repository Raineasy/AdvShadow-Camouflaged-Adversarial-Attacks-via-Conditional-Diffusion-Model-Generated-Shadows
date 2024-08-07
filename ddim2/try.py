from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def blend_images(image1, image2, alpha=0.5):
    """Blend two images using a specified alpha."""
    return Image.blend(image1, image2, alpha=alpha)

# 加载热力图和原始图像
heat_map_path = 'D:\\tree\\ddim\\ddim2\\heatmap_only.png'  # 热力图路径
original_image_path = 'D:\\tree\\ddim\\ddim2\\original_image_A1.png'  # 原始图像路径

heat_map = Image.open(heat_map_path)
original_image = Image.open(original_image_path)

# 通过PIL调整热力图大小以匹配原始图像
heat_map = heat_map.resize(original_image.size)

# 将PIL图像转换为numpy数组进行处理
heat_map_array = np.array(heat_map)

# 热力图平移
y_shift = -25  # 向上平移10像素
x_shift = 15   # 向右平移20像素
shifted_heatmap_array = np.roll(heat_map_array, shift=y_shift, axis=0)  # 垂直平移
shifted_heatmap_array = np.roll(shifted_heatmap_array, shift=x_shift, axis=1)  # 水平平移

# 将处理后的numpy数组转回PIL图像
shifted_heatmap = Image.fromarray(shifted_heatmap_array)

# 调整热力图的透明度
blended_image = blend_images(original_image, shifted_heatmap, alpha=0.6)  # alpha 设置为0.2以使热力图更透明

# 显示结果
plt.imshow(blended_image)
plt.axis('off')  # 不显示坐标轴
plt.show()
