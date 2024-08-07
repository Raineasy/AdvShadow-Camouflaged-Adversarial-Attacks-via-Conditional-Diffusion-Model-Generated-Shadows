import json

from sklearn.model_selection import train_test_split
from torchvision.utils import save_image

from diff_model import *


class CustomDataset(Dataset):
    def __init__(self, image_dir, image_files, labels, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = image_files  # 图像文件列表
        self.labels = labels  # 标签列表

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        label = self.labels[idx]  # 获取对应的标签

        if self.transform:
            image = self.transform(image)

        return image, label



if __name__ == "__main__":

    batch_size = 4
    timesteps = 1000

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 根据需要调整大小
        transforms.ToTensor(),
    ])

    image_labels_json = 'image_labels.json'
    #image_labels_json ='/kaggle/input/oxfordpet/ddim/image_labels.json'
    # 加载 JSON 数据并转换为字典
    with open(image_labels_json, 'r') as file:
        image_labels = json.load(file)
    image_files, labels = zip(*[(k, v) for k, v in image_labels.items()])
    train_files, test_files, train_labels, test_labels = train_test_split(
        image_files, labels, test_size=0.2, random_state=42
    )
    # 创建训练和测试数据集
    train_dataset = CustomDataset(
        image_dir='images',
        image_files=train_files,
        labels=train_labels,
        transform=transform
    )
    test_dataset = CustomDataset(
        image_dir='images',
        image_files=test_files,
        labels=test_labels,
        transform=transform
    )

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    model = UNetModel(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(2,),
        dropout=0.1
        # in_channels=3,  # 输入图像通道数
        # model_channels=128,  # 模型内部通道数
        # out_channels=3,  # 输出图像通道数
        # num_res_blocks=3,
        # attention_resolutions=(4, 8, 16),
        # channel_mult=(1, 2, 4, 8),
        # conv_resample=True,
        # num_heads=4
    )
    model.to(device)

    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)#原5e-4
    #train
    # best_loss = float('inf')
    # max_timesteps = timesteps
    # epochs = 50
    # for epoch in range(epochs):
    #     model.train()
    #     running_loss = 0.0
    #     num_batches = 0
    #     for (images, labels) in tqdm(train_loader, total=len(train_loader),desc=f"Epoch {epoch + 1}/{epochs}"):
    #         images = images.to(device)
    #         timesteps = torch.randint(0, max_timesteps, (images.size(0),), device=device)
    #         optimizer.zero_grad()
    #         loss = gaussian_diffusion.train_losses(model, images, timesteps)
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss += loss.item()
    #         num_batches += 1
    #     avg_loss = running_loss / num_batches
    #     print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    #     if avg_loss < best_loss:
    #         best_loss = avg_loss
    #         torch.save(model.state_dict(), 'best_model.pth')
        # 生成图像
    model.load_state_dict(torch.load('E:\\大树\\ddim\\oxfordpet-32-cs.pth'))
    model.eval()
    cnt1 = 0
    save_dir = 'generated_images'
    max_images = 1
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc='Generating Images'):
            images = images.to(device)
            original_images = images.to(device)
            generated_images = gaussian_diffusion.sample(model, 32, batch_size=batch_size, channels=images.shape[1])
            for i, (img_tensor, original_img_tensor) in enumerate(zip(generated_images[-1], original_images)):
                plt.figure()

                # 处理生成的图像
                if img_tensor.ndim == 4:
                    img_tensor = np.squeeze(img_tensor, axis=0)
                if img_tensor.shape[0] in [1, 3]:
                    img_tensor = img_tensor.transpose(1, 2, 0)  # 将通道移到最后

                    # 确保图像数据在正确的范围内
                img_tensor = np.clip(img_tensor, 0, 1)  # 用于浮点数

                # 处理原始图像
                if isinstance(original_img_tensor, torch.Tensor):
                    original_img_tensor = original_img_tensor.permute(1, 2, 0).cpu().numpy()
                original_img_tensor = np.clip(original_img_tensor, 0, 1)

                # 显示和保存原始图像
                plt.subplot(1, 2, 1)  # 第一个子图
                plt.imshow(original_img_tensor)
                plt.axis('off')

                # 显示和保存生成的图像
                plt.subplot(1, 2, 2)  # 第二个子图
                plt.imshow(img_tensor)
                plt.axis('off')

                # 保存图像
                img_path = os.path.join(save_dir, f'combined_image_{i+cnt1}.png')
                plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                cnt1 += 1  # 更新已处理的图像数量

            if cnt1 >= max_images:
                break  # 确保在处理了足够多的图像后跳出外层循环

    # 保存生成图片的文件夹
    # 创建保存图像的目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 只保存最后一步的图像
    final_image = generated_images[-1]  # 获取最后一个图像
