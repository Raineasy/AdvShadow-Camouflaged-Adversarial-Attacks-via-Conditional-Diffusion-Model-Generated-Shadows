import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Preprocessing and transform
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to the same size
    transforms.ToTensor()
])


def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return preprocess(img).numpy()


def calculate_ssim_psnr(image1, image2, win_size=11):
    image1 = np.transpose(image1, (1, 2, 0))  # Convert to HWC format
    image2 = np.transpose(image2, (1, 2, 0))  # Convert to HWC format
    ssim_value = ssim(image1, image2, win_size=win_size, channel_axis=2, gaussian_weights=True,data_range=image1.max() - image1.min())
    psnr_value = psnr(image1, image2, data_range=image1.max() - image1.min())
    return ssim_value, psnr_value


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            images.append(load_image(img_path))
    return images


def compare_folders(folder1, folder2, win_size=7):
    images1 = load_images_from_folder(folder1)
    images2 = load_images_from_folder(folder2)

    if len(images1) != len(images2):
        raise ValueError("Folders must contain the same number of images")

    ssim_values = []
    psnr_values = []

    for img1, img2 in zip(images1, images2):
        ssim_value, psnr_value = calculate_ssim_psnr(img1, img2, win_size=win_size)
        ssim_values.append(ssim_value)
        psnr_values.append(psnr_value)

    mean_ssim = np.mean(ssim_values)
    mean_psnr = np.mean(psnr_values)

    return mean_ssim, mean_psnr


# Paths to the folders containing images
folder1 = 'figure'
folder2 = 'attack_images_IG_64_res34_10'

# Compare the two folders
mean_ssim, mean_psnr = compare_folders(folder1, folder2)
print(f'Mean SSIM between the two folders: {mean_ssim:.4f}')
print(f'Mean PSNR between the two folders: {mean_psnr:.2f} dB')
