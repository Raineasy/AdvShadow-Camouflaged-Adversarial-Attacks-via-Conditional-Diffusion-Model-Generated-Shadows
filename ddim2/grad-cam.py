import os
import random
import torch
import numpy as np
from torchvision.models import resnet50
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 加载模型
model = resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 37)  # 修改为对应的类别数
model.load_state_dict(torch.load('D:\\tree\\ddim\\classifer_model\\resnet50\\pytorch_model.bin'))
model.eval()

# 设置 Grad-CAM
target_layer = model.layer4[-1]  # 选择 ResNet50 的最后一个卷积层
cam = GradCAM(model=model, target_layers=[target_layer])

# FGSM 攻击函数
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
# BIM 攻击函数
def bim_attack(model, image, label, epsilon, alpha, iters):
    original_image = image.clone()
    for i in range(iters):
        image.requires_grad = True
        output = model(image)
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        image_grad = image.grad.data
        adv_image = image + alpha * image_grad.sign()
        eta = torch.clamp(adv_image - original_image, min=-epsilon, max=epsilon)
        image = torch.clamp(original_image + eta, min=0, max=1).detach_()
    return image

# PGD 攻击函数
def pgd_attack(model, image, label, epsilon, alpha, iters):
    original_image = image.clone().detach()
    adv_image = image.clone().detach() + torch.randn_like(image) * epsilon

    for i in range(iters):
        adv_image.requires_grad = True
        output = model(adv_image)
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        adv_image_grad = adv_image.grad.data
        adv_image = adv_image.detach() + alpha * adv_image_grad.sign()
        eta = torch.clamp(adv_image - original_image, min=-epsilon, max=epsilon)
        adv_image = torch.clamp(original_image + eta, min=0, max=1).detach_()

    return adv_image
# 图像预处理
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 根据模型需求调整大小
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)  # 创建批次维度

fixed_image_path = 'D:\\tree\\ddim\\ddim2\\original_image_A1.png'  # 替换为实际图片路径
# def select_random_image(folder):
#     files = os.listdir(folder)
#     selected_file = random.choice(files)
#     return os.path.join(folder, selected_file), selected_file

# image_path_A, selected_filename = select_random_image(folder_A)
# image_path_B = os.path.join(folder_B, selected_filename)

# 载入和预处理图像
# input_tensor_A = preprocess_image(image_path_A)
# input_tensor_B = preprocess_image(image_path_B)

input_tensor = preprocess_image(fixed_image_path)
output = model(input_tensor)
pred_label = torch.argmax(output, dim=1)
#
# # 使用 Grad-CAM 生成原始图像的热力图
# grayscale_cam_A = cam(input_tensor=input_tensor_A)[0]
# grayscale_cam_B = cam(input_tensor=input_tensor_B)[0]
#
# # 获取模型的预测结果
# output_A = model(input_tensor_A)
# pred_label_A = torch.argmax(output_A, dim=1)
#
# # 计算损失并进行反向传播
# loss_A = F.cross_entropy(output_A, pred_label_A)
# model.zero_grad()
# loss_A.backward()

# FGSM 攻击

# # 可视化原始图像和攻击后图像的Grad-CAM
def visualize_and_save(image_tensor, grayscale_cam, title, grad_cam_save_path, original_save_path):
    image_for_cam = image_tensor.squeeze(0).detach().cpu().numpy()
    image_for_cam = image_for_cam.transpose(1, 2, 0)  # CHW to HWC
    image_for_cam = np.clip(image_for_cam, 0, 1)  # 确保数据在[0, 1]范围内

    # 保存原图
    plt.imsave(original_save_path, image_for_cam)

    # 使用 Grad-CAM
    visualization = show_cam_on_image(image_for_cam, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.title(title)
    plt.axis('off')
    plt.savefig(grad_cam_save_path)
    plt.show()

# # 可视化和保存图片A的Grad-CAM和原图
# visualize_and_save(input_tensor_A, grayscale_cam_A, 'Original Grad-CAM (C)', 'original_grad_cam_C1.png', 'original_image_C1.png')
#
# # 可视化和保存图片B的Grad-CAM和原图
# visualize_and_save(input_tensor_B, grayscale_cam_B, 'Original Grad-CAM (D)', 'original_grad_cam_D1.png', 'original_image_D1.png')
adv_tensor_bim = bim_attack(model, input_tensor, pred_label, epsilon=0.05, alpha=0.01, iters=10)

# 应用PGD攻击
adv_tensor_pgd = pgd_attack(model, input_tensor, pred_label, epsilon=0.05, alpha=0.01, iters=10)
input_tensor.requires_grad= True
output = model(input_tensor)
loss = F.cross_entropy(output, pred_label)
model.zero_grad()
loss.backward()

adv_tensor_fgsm = fgsm_attack(input_tensor,0.05,input_tensor.grad.data)
# 可视化原图、BIM和PGD攻击后的Grad-CAM
visualize_and_save(input_tensor, cam(input_tensor=input_tensor)[0], 'Original Image', 'grad_cam_original.png', 'original_image.png')
#visualize_and_save(adv_tensor_bim, cam(input_tensor=adv_tensor_bim)[0], 'BIM Attacked Image', 'grad_cam_bim.png', 'bim_attacked_image.png')
#visualize_and_save(adv_tensor_fgsm, cam(input_tensor=adv_tensor_fgsm)[0], 'FGSM Attacked Image', 'grad_cam_fgsm.png', 'fgsm_attacked_image.png')
#visualize_and_save(adv_tensor_pgd, cam(input_tensor=adv_tensor_pgd)[0], 'PGD Attacked Image', 'grad_cam_pgd.png', 'pgd_attacked_image.png')