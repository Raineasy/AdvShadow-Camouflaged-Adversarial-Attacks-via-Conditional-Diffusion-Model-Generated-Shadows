import json
import pathlib
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
from fastai.learner import load_learner
from diff_model2 import *

from PIL import UnidentifiedImageError

# 临时替换 pathlib 的 PosixPath 为 WindowsPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model_path = 'E:\\大树\\ddim\\classifer_model\\model.pkl'  # 修改为分类器 .pkl 文件路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
classifier = load_learner(model_path)
classifier.to(device)
# 还原 pathlib 的 PosixPath
pathlib.PosixPath = temp

def define_loss(output, label_tensor, model, regularization_factor=0.01):
    cross_entropy_loss = F.cross_entropy(output, label_tensor)
    regularization_term = 0
    for param in model.parameters():
        regularization_term += torch.norm(param)
    total_loss = cross_entropy_loss + regularization_factor * regularization_term
    return total_loss


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_files, labels, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = image_files
        self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        while True:
            try:
                img_name = self.image_files[idx]
                img_path = os.path.join(self.image_dir, img_name)
                mask_name = 'mask_' + img_name
                mask_path = os.path.join(self.mask_dir, mask_name)

                image = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('L')

                label = self.labels[idx]

                if self.transform:
                    image = self.transform(image)
                    mask = self.transform(mask)
                if idx == 0:  # 仅检查第一张图像
                    print("Image range after transform: Min = {}, Max = {}".format(image.min().item(),
                                                                                   image.max().item()))
                return image, mask, label
            except (UnidentifiedImageError, FileNotFoundError):
                # 如果图像或mask加载失败，跳过该样本
                idx = (idx + 1) % len(self.image_files)


if __name__ == "__main__":

    batch_size = 4
    timesteps = 1000

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 根据需要调整大小
        transforms.ToTensor(),
    ])

    image_labels_json = 'E:\\大树\\ddim\\image_labels.json'
    # image_labels_json ='/kaggle/input/oxfordpet/ddim/image_labels.json'
    # 加载 JSON 数据并转换为字典
    with open(image_labels_json, 'r') as file:
        image_labels = json.load(file)

    image_files, labels = zip(*image_labels.items())

    train_files = list(image_labels.keys())
    train_labels = list(image_labels.values())

    dataset_labels_json = 'E:\\大树\\ddim\\config.json'

    # 加载数据集中的JSON文件
    with open(dataset_labels_json, 'r') as file:
        data = json.load(file)

    # 获取 id 到 label 的映射
    id2label = data['id2label']

    # 创建 label 到 id 的映射
    label_to_int = {label: int(id) for id, label in id2label.items()}
    # print(label_to_int)
    # print(train_files[:5])  # 打印前5个训练文件名
    # print(train_labels[:5])  # 打印对应的标签整数
    num_classes = len(label_to_int)

    train_dataset = CustomDataset(
        image_dir='E:\\大树\\ddim\\images',
        mask_dir='E:\\大树\\ddim\\images_mask',  # 假设mask图像存储在'masks'文件夹中
        image_files=train_files,
        labels=train_labels,
        transform=transform
    )

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    model = UNetModel(
        in_channels=3,  # 输入图像通道数
        model_channels=128,  # 模型内部通道数
        out_channels=3,  # 输出图像通道数
        num_res_blocks=2,
        attention_resolutions=(4, 8, 16, 32),
        channel_mult=(1, 1, 2, 2, 4, 4),
        conv_resample=True,
        num_heads=4,
        dropout=0.1
        # in_channels=3,
        # model_channels=128,
        # out_channels=3,
        # num_res_blocks=2,
        # channel_mult=(1, 2, 2, 2),
        # attention_resolutions=(2,),
        # dropout=0.1
    )
    model.to(device)

    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 原5e-4
    #train
    best_loss = float('inf')
    max_timesteps = timesteps
    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (images, masks, labels) in pbar:
            images = images.to(device)
            masks = masks.to(device)

            # 生成批次的时间步长
            timesteps = torch.randint(0, max_timesteps, (images.size(0),), device=device)

            # 优化阴影位置和大小，并收集阴影图像
            shadowed_images_batch = []
            for image, mask, true_label in zip(images, masks, labels):
                # 随机选择一个目标标签（不同于真实标签）
                while True:
                    target_label = torch.randint(0, num_classes, (1,), device=device)
                    if target_label.item() != label_to_int[true_label]:
                        break

                _, _, shadowed_image = gaussian_diffusion.optimize_shadow_position(classifier, image, mask,
                                                                                   target_label, device)
                shadowed_images_batch.append(shadowed_image)

            # 将单个阴影图像合并为批次
            shadowed_images_batch = torch.cat(shadowed_images_batch, dim=0)
            # if batch_idx == 0:  # 仅检查第一批次
            #     print("Output image range: Min = {}, Max = {}".format(shadowed_images_batch.min().item(),
            #                                                           shadowed_images_batch.max().item()))
            # for i, shadowed_image in enumerate(shadowed_images_batch):
            #     # 可以选择只展示部分图像
            #     if i == 0:  # 例如，只展示每个批次的第一个图像
            #         plt.imshow(shadowed_image.cpu().detach().permute(1, 2, 0))
            #         plt.title(f"Shadowed Image from Batch {batch_idx}")
            #         plt.show()
            optimizer.zero_grad()
            #adversarial_losses = []

            # for shadowed_image, label in zip(shadowed_images_batch, labels):
            #     output = classifier.model(shadowed_image.unsqueeze(0))
            #     label_index = label_to_int[label]
            #     label_tensor = torch.tensor([label_index], device=device)
            #     loss = define_loss(output, label_tensor, model)
            #     adversarial_losses.append(loss)

            # 计算平均对抗性损失
            #adversarial_loss = torch.stack(adversarial_losses).mean()
            ddpm_loss = gaussian_diffusion.train_losses(model, shadowed_images_batch, timesteps, device)
            #total_loss = ddpm_loss + 0.0001 * adversarial_loss
            total_loss = ddpm_loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            num_batches += 1
            pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
            pbar.set_postfix(loss=running_loss / num_batches)

        avg_loss = running_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')

        # 生成图像
    torch.save(model.state_dict(), 'best_model.pth')

