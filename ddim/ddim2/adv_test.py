import os
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np

# 加载预训练模型
model = resnet50(pretrained=True)
model.eval()


# 攻击函数

# FGSM 攻击函数
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# BIM 攻击函数
def bim_attack(image, epsilon, alpha, num_iterations):
    perturbed_image = image.clone().detach().requires_grad_(True)
    for i in range(num_iterations):
        outputs = model(perturbed_image)
        loss = F.cross_entropy(outputs, outputs.max(1)[1])
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image + alpha * data_grad.sign()
        perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = perturbed_image.clone().detach().requires_grad_(True)
    return perturbed_image


# PGD 攻击函数
def pgd_attack(image, epsilon, alpha, num_iterations):
    perturbed_image = image.clone().detach() + torch.empty_like(image).uniform_(-epsilon, epsilon)
    perturbed_image = torch.clamp(perturbed_image, 0, 1).requires_grad_(True)
    for i in range(num_iterations):
        outputs = model(perturbed_image)
        loss = F.cross_entropy(outputs, outputs.max(1)[1])
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image + alpha * data_grad.sign()
        perturbed_image = torch.min(torch.max(perturbed_image, image - epsilon), image + epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = perturbed_image.clone().detach().requires_grad_(True)
    return perturbed_image


# 图像预处理
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 根据模型需求调整大小
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)  # 创建批次维度


# 保存图像
def save_image(tensor, path):
    img = tensor.squeeze().detach().cpu().numpy()
    img = img.transpose(1, 2, 0)  # CHW to HWC
    img = np.clip(img, 0, 1)  # 确保数据在[0, 1]范围内
    img = (img * 255).astype(np.uint8)  # 恢复到[0, 255]范围
    img = Image.fromarray(img)
    img.save(path)


# 攻击和保存图像
def attack_and_save(image_path, epsilon, alpha, num_iterations, save_dir):
    image = preprocess_image(image_path)
    image.requires_grad = True

    # FGSM 攻击
    outputs = model(image)
    loss = F.cross_entropy(outputs, outputs.max(1)[1])
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    fgsm_image = fgsm_attack(image, epsilon, data_grad)
    save_image(fgsm_image, os.path.join(save_dir, 'fgsm111.png'))

    # BIM 攻击
    bim_image = bim_attack(image, epsilon, alpha, num_iterations)
    save_image(bim_image, os.path.join(save_dir, 'bim111.png'))

    # PGD 攻击
    pgd_image = pgd_attack(image, epsilon, alpha, num_iterations)
    save_image(pgd_image, os.path.join(save_dir, 'pgd111.png'))


# 示例使用
image_path = 'E:\\tree\\毕设\\图\\图片4.png'  # 替换为实际的图像路径
save_dir = 'E:\\tree\\ddim\\ddim2\\generated_images'  # 保存攻击后图像的目录
os.makedirs(save_dir, exist_ok=True)

epsilon = 0.05  # 扰动强度
alpha = 0.05  # 每次迭代的步长
num_iterations = 10  # 迭代次数

attack_and_save(image_path, epsilon, alpha, num_iterations, save_dir)
