
import cv2
from PIL import Image, ImageDraw
import numpy as np

# 加载图片和 mask
image_path = "images/Abyssinian_1.jpg"  # 替换为你的图片路径
mask_path = "images_mask/mask_Abyssinian_1.jpg"  # 替换为你的 mask 图片路径
original_image = Image.open(image_path)
mask_image = Image.open(mask_path)
def adjust_shadow_brightness(image, mask, factor=0.43):
    # 转换 PIL 图像到 OpenCV 格式
    image_cv = np.array(image.convert('RGB'))
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # 转换图像到 LAB 色彩空间
    lab_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2Lab)

    # 调整阴影区域的亮度
    mask_cv = np.array(mask.convert('L'))  # 确保 mask 是单通道灰度图
    l, a, b = cv2.split(lab_image)
    l = l.astype(np.float32)  # 转换为 float 以进行计算
    l = np.where(mask_cv, l * factor, l)
    l = np.clip(l, 0, 255).astype(np.uint8)  # 限制范围并转换回 uint8

    # 合并通道并转换回 RGB 色彩空间
    lab_image = cv2.merge([l, a, b])
    adjusted_image_cv = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)
    adjusted_image_cv = cv2.cvtColor(adjusted_image_cv, cv2.COLOR_BGR2RGB)

    # 转换回 PIL 图像格式
    adjusted_image = Image.fromarray(adjusted_image_cv)
    return adjusted_image
# 将 mask 转换为 OpenCV 格式
mask_cv = np.array(mask_image)
if len(mask_cv.shape) == 3 and mask_cv.shape[2] == 3:
    mask_cv = cv2.cvtColor(mask_cv, cv2.COLOR_RGB2GRAY)

# 查找 mask 中的连通区域
contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最大的连通区域
max_contour = max(contours, key=cv2.contourArea)

# 计算边界框
x, y, w, h = cv2.boundingRect(max_contour)

# 在 PIL 图像上创建三角形阴影
triangle_shadow = Image.new('RGBA', original_image.size, (255, 255, 255, 0))
draw = ImageDraw.Draw(triangle_shadow)

# 在边界框中心附近创建三角形
cx, cy = x + w // 2, y + h // 2  # 边界框中心
triangle_size = min(w, h) // 2    # 基于边界框大小调整三角形大小
draw.polygon([(cx, cy - triangle_size), (cx - triangle_size, cy + triangle_size),
              (cx + triangle_size, cy + triangle_size)], fill=(0, 0, 0, 128), outline=None)

# 将三角形阴影覆盖到图片上
combined_image = Image.alpha_composite(original_image.convert('RGBA'), triangle_shadow)
final_image = Image.composite(combined_image, original_image, mask_image)
#final_image = adjust_shadow_brightness(final_image, mask_image)
# 保存修改后的图片
final_image.save("modified_image.png")
