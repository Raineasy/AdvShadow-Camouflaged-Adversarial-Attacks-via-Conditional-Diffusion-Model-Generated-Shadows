import os
import pathlib
import json
import torch

from fastai.learner import load_learner
from fastai.vision.core import PILImage
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
import os
from PIL import Image
from tqdm import tqdm

# 临时替换 pathlib 的 PosixPath 为 WindowsPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model_path = 'classifer_model/model.pkl'  # 修改为您的 .pkl 文件路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = load_learner(model_path)
model.to(device)
# 还原 pathlib 的 PosixPath
pathlib.PosixPath = temp

# # 准备单个图像进行预测
# image_path = 'Abyssinian_1.jpg'  # 替换为您的图像文件路径
# preds, _, probs = model.predict(image_path)
# # 输出预测结果
#
# print(f"Predicted: {preds}, Probability: {probs.max().item()}")
# 文件夹路径
image_folder = 'shadowed_images'
image_files = os.listdir(image_folder)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
# 生成并保存标签
labels = []
for file in tqdm(image_files):
    image_path = os.path.join(image_folder, file)
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    img_fastai = PILImage.create(image_path)

    with torch.no_grad():
        pred, _, probs = model.predict(img_fastai)
        label = pred

    labels.append(label)

# 保存标签


with open('image_labels.json', 'w') as f:
    json.dump(dict(zip(image_files, labels)), f)