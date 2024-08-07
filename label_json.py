import os
import json

# 指定包含图片的目录
directory_path = 'E:\\tree\\ddim\\figure'

# 读取目录下的所有文件名
file_names = os.listdir(directory_path)

# 创建一个字典，存储文件名和类别
image_labels = {}
for file_name in file_names:
    # 假设类别是文件名的第一部分，直到第一个下划线
    category = file_name.split('_')[0]
    image_labels[file_name] = category

# 新 JSON 文件的保存路径
new_json_path = 'E:\\tree\\ddim\\image_labels_500.json'

# 写入 JSON 文件
with open(new_json_path, 'w') as file:
    json.dump(image_labels, file, indent=4)

print("JSON file has been created with entries for each image in the directory.")
