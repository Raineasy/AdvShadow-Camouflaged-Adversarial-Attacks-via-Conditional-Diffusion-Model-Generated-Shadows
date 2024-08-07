import os
import pathlib
import tempfile

from PIL import Image, ImageDraw

from fastai.vision.all import load_learner
import numpy as np
import cv2
import random

from matplotlib import pyplot as plt
from tqdm import tqdm

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# 加载模型
model_path = 'classifer_model/model.pkl'  # 替换为您的 .pkl 文件路径
model = load_learner(model_path)


def generate_triangle_shadow(mask):
    """
    生成一个随机位置的三角形阴影，并确保它与掩码的一个子区域相交。
    """
    mask_cv = np.array(mask)
    if len(mask_cv.shape) == 3:
        mask_cv = cv2.cvtColor(mask_cv, cv2.COLOR_RGB2GRAY)

    contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = random.choice(contours)
    x, y, w, h = cv2.boundingRect(contour)

    # 只在掩码的一个子区域内生成三角形
    sub_x, sub_y, sub_w, sub_h = x + w//4, y + h//4, w//2, h//2  # 取轮廓的中心部分
    triangle_shadow = Image.new('RGBA', mask.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(triangle_shadow)

    # 在子区域内生成三角形
    cx, cy = sub_x + sub_w // 2, sub_y + sub_h // 2
    triangle_size = min(sub_w, sub_h) // 3
    draw.polygon([(cx, cy - triangle_size), (cx - triangle_size, cy + triangle_size),
                  (cx + triangle_size, cy + triangle_size)], fill=(0, 0, 0, 128))

    return triangle_shadow

def adjust_shadow_brightness(image, mask, factor=0.43):
    # 将PIL图像转换为NumPy数组
    image_np = np.array(image)
    mask_np = np.array(mask)

    # 确保遮罩尺寸与图像尺寸相匹配
    mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 确保遮罩是布尔类型
    mask_bool = mask_np.astype(bool)

    # 将图像转换为浮点数以进行计算
    image_float = image_np.astype(np.float32)

    # 仅在遮罩区域应用阴影
    image_float[mask_bool] *= factor

    # 确保所有值都在合理的范围内
    np.clip(image_float, 0, 255, out=image_float)

    # 将图像转换回原始数据类型
    image_shadowed = image_float.astype(np.uint8)

    return Image.fromarray(image_shadowed)

# 函数：在掩码区域内添加阴影
def add_shadow_to_mask_area(image, mask):
    triangle_shadow = generate_triangle_shadow(mask)
    if triangle_shadow is None:
        print("none")
        return image
    # 应用三角形阴影到图像
    shadow_mask = np.array(triangle_shadow.convert('L'))
    mask_cv = np.array(mask.convert('L'))
    intersect_shadow = np.bitwise_and(shadow_mask, mask_cv)
    intersect_shadow = Image.fromarray(intersect_shadow)

    # 结合原图和阴影
    shadow_layer = Image.new('RGBA', image.size, (255, 255, 255, 0))
    shadow_layer.paste(triangle_shadow, mask=intersect_shadow)
    combined_image = Image.alpha_composite(image.convert('RGBA'), shadow_layer).convert('RGB')

    final_image = adjust_shadow_brightness(combined_image, mask)
    return final_image

# 函数：对图像进行分类
def classify_image(image):
    # 检查 image 是否为 PIL.Image.Image 对象
    if isinstance(image, Image.Image):
        # 直接保存 PIL 图像为临时文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
            image.save(temp.name)
            preds, _, probs = model.predict(temp.name)  # 使用文件路径进行预测
    else:
        # 若 image 是文件路径
        preds, _, probs = model.predict(image)

    return preds, probs.max().item()


# 主函数
def main():
    image_dir = 'images'  # 图片文件夹路径
    mask_dir = 'images_mask'  # 掩码文件夹路径
    output_dir = 'shadowed_images'  # 输出文件夹路径

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图片文件名
    image_filenames = os.listdir(image_dir)

    # 使用 tqdm 创建进度条
    for image_filename in tqdm(image_filenames, desc="Processing Images"):
        image_path = os.path.join(image_dir, image_filename)
        mask_filename = f'mask_{image_filename}'
        mask_path = os.path.join(mask_dir, mask_filename)

        try:
            # 尝试打开图片和掩码
            with Image.open(image_path) as img:
                if os.path.exists(mask_path):
                    with Image.open(mask_path) as mask:
                        # 添加阴影并保存结果
                        shadowed_image = add_shadow_to_mask_area(img, mask)
                        shadowed_image.save(os.path.join(output_dir, image_filename))
                else:
                    print(f"No mask found for {image_filename}, skipping.")
        except (IOError, OSError):
            print(f"Error opening {image_filename}, skipping.")


if __name__ == "__main__":
    main()
