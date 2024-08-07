import os
import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from PIL import Image

# Load pre-trained InceptionV3 model
inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model.fc = nn.Identity()  # Remove the final classification layer
inception_model.eval()

# Preprocessing and transform
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_activations(images):
    with torch.no_grad():
        images = torch.stack([preprocess(image) for image in images])
        activations = inception_model(images)
    return activations.cpu().numpy()


def calculate_fid(act1, act2):
    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('RGB')
        images.append(img)
    return images


# Paths to the folders containing images
folder1 = 'D:\\EdgeDownload\\oimages'
folder2 = 'D:\\EdgeDownload\\fgsm1'

# Load images
images1 = load_images_from_folder(folder1)
images2 = load_images_from_folder(folder2)

# Get activations
act1 = get_activations(images1)
act2 = get_activations(images2)

# Calculate FID
fid_value = calculate_fid(act1, act2)
print(f'FID between the two folders: {fid_value}')
