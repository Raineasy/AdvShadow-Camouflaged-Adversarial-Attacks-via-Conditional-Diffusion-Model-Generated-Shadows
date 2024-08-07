import json
import pathlib
import os
import robustbench
import timm
import torch
from fastai.learner import load_learner
from matplotlib import pyplot as plt
from torchvision import models
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from transformers import AutoImageProcessor, AutoModelForImageClassification

def load_resnet50_model():
    checkpoint_path = "D:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\resnet50\\pytorch_model.bin"
    model = timm.create_model("resnet50", pretrained=False, num_classes=37)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
    return model
def load_convnext_model():
    weights_path = "D:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\convnext\\pytorch_model.bin"
    model = timm.create_model("convnext_base.fb_in1k", pretrained=False, num_classes=37)
    model.load_state_dict(torch.load(weights_path, map_location='cuda'))
    return model
##########################   swin  ###############################################
def load_swin_model():
    weights_path = "D:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\swin\\pytorch_model.bin"
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=37)
    model.load_state_dict(torch.load(weights_path, map_location='cuda'))
    return model
############################# VGG 16  #########################################
def load_vgg16_model():
    model = models.vgg16()
    # 如果有自己训练的模型权重
    model.classifier[6] = nn.Linear(4096, 37)
    model.load_state_dict(torch.load('D:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\trained_vgg16_pet.pth'))
    model.eval()  # 设置为评估模式
    return model
def load_vgg19_model():
    model = models.vgg19()
    # 如果有自己训练的模型权重
    model.classifier[6] = nn.Linear(4096, 37)
    model.load_state_dict(torch.load('D:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\trained_vgg19_pet.pth'))
    model.eval()  # 设置为评估模式
    return model
def load_vitt_model():
    vit_directory = "D:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\vit"
    processor = AutoImageProcessor.from_pretrained(vit_directory)
    model = AutoModelForImageClassification.from_pretrained(vit_directory)
    #pipe = pipeline("image-classification", model=model, feature_extractor=processor)
    return model
def load_dinov2_model():
    vit_directory = "D:\\tree\\experiment_indicators\\classifier_model\\classifer_model\\dinov2"
    processor = AutoImageProcessor.from_pretrained(vit_directory)
    model = AutoModelForImageClassification.from_pretrained(vit_directory)
    #pipe = pipeline("image-classification", model=model, feature_extractor=processor)
    return model
def load_efnetv2_model():
    checkpoint_path = "D:\\EdgeDownload\\best_oxford_pets_efficientnetv2.pth"
    model = efficientnet_v2_s(pretrained=False)  # 确保此处与训练时一致
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 37)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model
# 加载标签
with open('config2.json', 'r') as f:
    data = json.load(f)

# 获取 id 到 label 的映射
id2label = data['id2label']

# 创建 label 到 id 的映射
label_to_int = {label: int(id) for id, label in id2label.items()}
print(label_to_int)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model_path = 'D:\\tree\\ddim\\classifer_model\\resnet-18-classifier.pkl'  # 修改为分类器 .pkl 文件路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
classifier = load_learner(model_path)
classifier.to(device)
classifier.eval()

# 还原 pathlib 的 PosixPath
pathlib.PosixPath = temp

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 根据模型需求调整大小
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image

int_to_label = {v: k for k, v in label_to_int.items()}

def compute_asr(folder_path, model, int_to_label):
    total_images = 0
    successful_attacks = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            total_images += 1
            image_path = os.path.join(folder_path, filename)
            true_label = filename.rsplit('_', 1)[0]

            image = preprocess_image(image_path)
            image = image.to(device)
            outputs = model(image)
            #outputs = outputs.logits
            _, predicted_class = torch.max(outputs, 1)
            predicted_label = int_to_label[predicted_class.item()]

            if predicted_label != true_label:
                successful_attacks += 1

            #print(f"Image: {filename}, True Label: {true_label}, Predicted Label: {predicted_label}")

    asr = successful_attacks / total_images
    print(total_images)
    print(successful_attacks)
    return asr


classifier = load_resnet50_model()
classifier=classifier.to(device)
folder_path = 'D:\\tree\\experiment_indicators\\data\\8.3\\vnifgsm_ef'  # 替换为实际的文件夹路径
asr = compute_asr(folder_path, classifier, int_to_label)
print(f'Attack Success Rate (ASR): {(1-asr)*100:.4f}')


classifier = load_vgg19_model()
classifier=classifier.to(device)
folder_path = 'D:\\tree\\experiment_indicators\\data\\8.3\\vnifgsm_ef'  # 替换为实际的文件夹路径
asr = compute_asr(folder_path, classifier, int_to_label)
print(f'Attack Success Rate (ASR): {(1-asr)*100:.4f}')


classifier = load_efnetv2_model()
classifier=classifier.to(device)
folder_path = 'D:\\tree\\experiment_indicators\\data\\8.3\\vnifgsm_ef'  # 替换为实际的文件夹路径
asr = compute_asr(folder_path, classifier, int_to_label)
print(f'Attack Success Rate (ASR): {(1-asr)*100:.4f}')

classifier = load_vgg16_model()
classifier=classifier.to(device)
folder_path = 'D:\\tree\\experiment_indicators\\data\\8.3\\vnifgsm_ef'  # 替换为实际的文件夹路径
asr = compute_asr(folder_path, classifier, int_to_label)
print(f'Attack Success Rate (ASR): {(1-asr)*100:.4f}')


classifier = load_convnext_model()
classifier=classifier.to(device)
folder_path = 'D:\\tree\\experiment_indicators\\data\\8.3\\vnifgsm_ef'  # 替换为实际的文件夹路径
asr = compute_asr(folder_path, classifier, int_to_label)
print(f'Attack Success Rate (ASR): {(1-asr)*100:.4f}')


classifier = load_swin_model()
classifier=classifier.to(device)
folder_path = 'D:\\tree\\experiment_indicators\\data\\8.3\\vnifgsm_res50'  # 替换为实际的文件夹路径
asr = compute_asr(folder_path, classifier, int_to_label)
print(f'Attack Success Rate (ASR): {(1-asr)*100:.4f}')


# classifier = load_dinov2_model()
# classifier=classifier.to(device)
# folder_path = 'D:\\tree\\experiment_indicators\\data\\8.3\\vnifgsm_ef'  # 替换为实际的文件夹路径
# asr = compute_asr(folder_path, classifier, int_to_label)
# print(f'Attack Success Rate (ASR): {(1-asr)*100:.4f}')
#
# classifier = load_vitt_model()
# classifier=classifier.to(device)
# with open('configvit.json', 'r') as f:
#     data = json.load(f)
#
# # 获取 id 到 label 的映射
# id2label = data['id2label']
#
# # 创建 label 到 id 的映射
# label_to_int = {label: int(id) for id, label in id2label.items()}
# print(label_to_int)
# int_to_label = {v: k for k, v in label_to_int.items()}
# folder_path = 'D:\\tree\\experiment_indicators\\data\\8.3\\vnifgsm_ef'  # 替换为实际的文件夹路径
# asr = compute_asr(folder_path, classifier, int_to_label)
# print(f'Attack Success Rate (ASR): {(1-asr)*100:.4f}')

