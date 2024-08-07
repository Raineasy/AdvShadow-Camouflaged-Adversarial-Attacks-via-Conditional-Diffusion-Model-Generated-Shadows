import json
import pathlib

import timm
import torch
from fastai.learner import load_learner
from matplotlib import pyplot as plt
from torchvision import models
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 加载标签

with open('config.json', 'r') as f:
    data = json.load(f)

# 获取 id 到 label 的映射
id2label = data['id2label']

# 创建 label 到 id 的映射
label_to_int = {label: int(id) for id, label in id2label.items()}
print(label_to_int)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model_path = 'E:\\tree\\ddim\\classifer_model\\resnet-34-classifier.pkl'  # 修改为分类器 .pkl 文件路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
classifier = load_learner(model_path)
classifier.to(device)
classifier.eval()
# 还原 pathlib 的 PosixPath
pathlib.PosixPath = temp

def load_resnet50_model():
    checkpoint_path = "E:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\resnet50\\pytorch_model.bin"
    model = timm.create_model("resnet50", pretrained=False, num_classes=37)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
    return model
def load_convnext_model():
    weights_path = "E:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\convnext\\pytorch_model.bin"
    model = timm.create_model("convnext_base.fb_in1k", pretrained=False, num_classes=37)
    model.load_state_dict(torch.load(weights_path, map_location='cuda'))
    return model
##########################   swin  ###############################################
def load_swin_model():
    weights_path = "E:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\swin\\pytorch_model.bin"
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=37)
    model.load_state_dict(torch.load(weights_path, map_location='cuda'))
    return model
############################# VGG 16  #########################################
def load_vgg16_model():
    model = models.vgg16()
    # 如果有自己训练的模型权重
    model.classifier[6] = nn.Linear(4096, 37)
    model.load_state_dict(torch.load('E:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\trained_vgg16_pet.pth'))
    model.eval()  # 设置为评估模式
    return model
def load_vgg19_model():
    model = models.vgg19()
    # 如果有自己训练的模型权重
    model.classifier[6] = nn.Linear(4096, 37)
    model.load_state_dict(torch.load('E:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\trained_vgg19_pet.pth'))
    model.eval()  # 设置为评估模式
    return model
def load_resnet18_model():
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    model_path = 'E:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\resnet-18-classifier.pkl'  # 修改为分类器 .pkl 文件路径
    model = load_learner(model_path)
    model.to(device)
    model.eval()
    pathlib.PosixPath = temp
    return model
##########################   ResNet-34  ###############################################
def load_resnet34_model():
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    model_path = 'E:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\resnet-34-classifier.pkl'  # 修改为分类器 .pkl 文件路径
    model = load_learner(model_path)
    model.to(device)
    model.eval()
    pathlib.PosixPath = temp
    return model
def load_vit_model():
    vit_directory = "E:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\vit"
    processor = AutoImageProcessor.from_pretrained(vit_directory)
    model = AutoModelForImageClassification.from_pretrained(vit_directory)
    #pipe = pipeline("image-classification", model=model, feature_extractor=processor)
    return model
def load_vitt_model():
    vit_directory = "E:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\vit_tiny"
    processor = AutoImageProcessor.from_pretrained(vit_directory)
    model = AutoModelForImageClassification.from_pretrained(vit_directory)
    #pipe = pipeline("image-classification", model=model, feature_extractor=processor)
    return model
def load_divcon_model():
    vit_directory = "E:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\dinov2"
    processor = AutoImageProcessor.from_pretrained(vit_directory)
    model = AutoModelForImageClassification.from_pretrained(vit_directory)
    #pipe = pipeline("image-classification", model=model, feature_extractor=processor)
    return model
# 加载和预处理图像
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 根据模型需求调整大小
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image
int_to_label = {v: k for k, v in label_to_int.items()}
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# cnt = 0
# all = 0
# model = load_divcon_model()
# for i in range(37):
#     for j in range(20):
#         image_path = f'E:\\tree\\experiment_indicators\\dfimages1\\resnet50_vgg_convnext\\output{i}\\df_{j}.jpg'  # 测试图像路径
#         image = Image.open(image_path).convert('RGB')
#         image_tensor = transform(image).unsqueeze(0)
#
#         image_tensor = image_tensor.to(device)
#         model = model.to(device)
#
#         output = model(image_tensor)
#         logits = output.logits
#         _,predicted_class = torch.max(logits,1)
#         #predicted_class1 = int_to_label[predicted_class.item()]
#         print(f"#########################{i}")
#         #print(f'Predicted class_name{i}: {predicted_class1}')
#         print(f'Predicted class{i}: {predicted_class}')
#         if predicted_class.item()!=i:
#             cnt+=1
#         all+=1
#
# print(cnt/all)
# print(cnt)
# print(all)
# # 使用模型进行预测
# # cnt = 0
# # all = 0
# # model = load_resnet50_model()
# # for i in range(36):
# #     for j in range(20):
# #         image_path = f'E:\tree\experiment_indicators\dfimages1\output{i}\\df_{j}.jpg'  # 测试图像路径
# #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         model = classifier.to(device)
# #         image = preprocess_image(image_path)
# #         image = image.to(device)
# #         outputs = classifier.model(image)
# #         _,predicted_class = torch.max(outputs,1)
# #         predicted_class1 = int_to_label[predicted_class.item()]
# #         print(f"#########################{i}")
# #         print(f'Predicted class_name{i}: {predicted_class1}')
# #         print(f'Predicted class{i}: {predicted_class}')
# #         if predicted_class.item()!=i:
# #             cnt+=1
# #         all+=1
# #
# # print(cnt/all)
# # print(cnt)
# # print(all)
image_path = f'E:\\tree\\ddim\\ddim2\\generated_images\\图片1.png'  # 测试图像路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = classifier.to(device)
image = preprocess_image(image_path)
image = image.to(device)
outputs = classifier.model(image)
_, predicted_class = torch.max(outputs, 1)
predicted_class1 = int_to_label[predicted_class.item()]
print(f'Predicted class_name{0}: {predicted_class1}')
print(f'Predicted class{0}: {predicted_class}')
