# import numpy as np
# import cv2
# import os
#
#
# def add_noise_and_save(image_path, output_dir, steps=500):
#     # Load the original image
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#
#     # Ensure output directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # Define the noise level increment
#     max_noise_level = 500  # You can adjust this level depending on how quickly you want the noise to overwhelm the image
#     noise_increment = max_noise_level / steps
#
#     # Save the original image as the first step
#     cv2.imwrite(os.path.join(output_dir, f"step_0.jpg"), image)
#
#     # Incrementally add noise and save each step
#     for step in range(1, steps + 1):
#         # Calculate current noise level
#         current_noise_level = noise_increment * step
#
#         # Generate Gaussian noise
#         noise = np.random.normal(loc=0, scale=current_noise_level, size=image.shape)
#
#         # Add noise to the original image
#         noisy_image = image + noise
#         noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values remain in [0, 255]
#         noisy_image = noisy_image.astype(np.uint8)  # Convert back to uint8
#
#         # Save the noisy image
#         cv2.imwrite(os.path.join(output_dir, f"step_{step}.jpg"), noisy_image)
#
#
# # Example usage
# add_noise_and_save("E:\\123456\\123.jpg", "E:\\123456\\aaa")
#
import os
from PIL import Image

def resize_images_in_folder(input_folder, output_folder, size=(64, 64)):
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 生成输入文件路径
        input_path = os.path.join(input_folder, filename)
        # 检查文件是否为图像文件
        if input_path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            # 打开图像
            with Image.open(input_path) as img:
                # 调整图像大小
                resized_img = img.resize(size)
                # 生成输出文件路径
                output_path = os.path.join(output_folder, filename)
                # 保存调整大小后的图像
                resized_img.save(output_path)
                print(f"Resized and saved image: {output_path}")

# 输入和输出文件夹路径
input_folder = 'figure'
output_folder = 'attack_IG'

# 调整文件夹中的所有图像大小
resize_images_in_folder(input_folder, output_folder)

