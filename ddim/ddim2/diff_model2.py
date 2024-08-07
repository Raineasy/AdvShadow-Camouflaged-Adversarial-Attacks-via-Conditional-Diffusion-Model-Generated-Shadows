import os
import math
from abc import abstractmethod
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from fastai.vision.core import PILImage


class PretrainedResNet50:
    def __init__(self, weight_path, device):
        """
        初始化预训练的ResNet50模型。

        Parameters:
        - weight_path: 预训练权重的路径。
        - device: 将模型加载到指定的设备（'cuda' 或 'cpu'）。
        """
        self.model = models.resnet50(pretrained=False)
        self.model.load_state_dict(torch.load(weight_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()  # 设置为评估模式

    def predict(self, images):
        """
        使用预训练的ResNet50模型对图像进行分类。

        Parameters:
        - images: 要分类的图像张量。

        Returns:
        - 分类结果。
        """
        with torch.no_grad():
            return self.model(images)


# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# define TimestepEmbedSequential to support `time_emb` as extra input
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# use GN for norm layer
def norm_layer(channels):
    return nn.GroupNorm(32, channels)


# Residual block
class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


# Attention block with shortcut
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x


# upsample
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


# downsample
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.op(x)


# The full UNet model with attention and timestep embedding
class UNetModel(nn.Module):
    def __init__(
            self,
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=3,  # 第一次调整，原为2
            attention_resolutions=(4, 8, 16, 32),  # 第一次调整，原为（8，16）
            dropout=0.1,  # 第一次调整，原为0.1
            channel_mult=(1, 2, 4, 8),  # 第一次调整，原为（1，2，2，2）
            conv_resample=True,
            num_heads=4 # 第一次调整，原为3
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        # time step embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h, emb)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, emb)
        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        return self.out(h)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion:
    def __init__(
            self,
            timesteps=1000,
            beta_schedule='linear'
    ):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        # self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )

        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
        )

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        # predict noise using model
        pred_noise = model(x_t, t)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True):
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                                 clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    # denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        batch_size = shape[0]
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long))
            imgs.append(img.cpu().numpy())
        return imgs

    # sample new images
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=8, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

    def optimize_shadow_position(self, classifier, original_image, mask, target_label, device, lr=1e-1, iterations=10,
                                 ):
        """
        使用梯度上升优化阴影位置，并在阴影区域施加基于梯度的对抗性扰动，同时考虑阴影的自然性损失。

        Parameters:
        - classifier: 分类器模型。
        - original_image: 原始图像。
        - mask: original_image的mask
        - target_label: 目标标签（我们希望模型错误分类为此标签）。
        - device: cpu/gpu
        - lr: 学习率。
        - iterations: 迭代次数。

        Returns:
        - 最优阴影位置。
        """
        # 初始化阴影半径
        #shadow_radius = torch.nn.Parameter(torch.rand(1) * 20 + 10, requires_grad=True)  # 调整半径初始值和大小
        mask_indices = torch.nonzero(mask)
        mask_center = mask_indices.float().mean(0)[1:]  # 取 y 和 x 坐标的平均值
        shadow_center = torch.nn.Parameter(mask_center.clone(), requires_grad=True)

        # 初始化阴影半径为较小的值
        shadow_radius = torch.nn.Parameter(torch.tensor(20.0), requires_grad=True)
        original_image = original_image.to(device)
        mask = mask.to(device)
        # 优化器同时优化阴影中心和半径
        optimizer = torch.optim.Adam([shadow_center, shadow_radius], lr=lr)
        model = classifier.model
        model = model.to(device)
        for iteration in range(iterations):
            optimizer.zero_grad()

            # 应用阴影
            shadowed_image = self.apply_shadow(image = original_image, shadow_center=shadow_center, shadow_radius=shadow_radius, feature_mask=mask, classifier=classifier, target_label=target_label, device=device)
            shadow_mask = self.create_shadow_mask(original_image.size(), shadow_center, shadow_radius,device)
            # 确保 shadowed_image 是适当的输入格式
            if shadowed_image.dim() == 4:
                shadowed_image_for_loss = shadowed_image.squeeze(0)
            else:
                shadowed_image_for_loss = shadowed_image

            shadowed_image_for_loss = shadowed_image_for_loss.to(device)

            # 对模型进行前向传播，获取原始输出
            output_tensor = model(shadowed_image_for_loss.unsqueeze(0))
            # 计算交叉熵损失
            adversarial_loss = -F.cross_entropy(output_tensor, target_label)
            # 阴影自然性损失
            natural_loss = F.mse_loss(shadowed_image_for_loss, original_image)
            regularization = (shadow_center - mask_center).pow(2).sum() + shadow_radius.pow(2)
            # 总损失
            loss = adversarial_loss + natural_loss + 0.1 * regularization
            loss.backward()
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Loss={loss.item()}, adv_loss={adversarial_loss.item()}")
                fig, axs = plt.subplots(1, 4, figsize=(20, 5))

                # 显示原始图像
                axs[0].imshow(original_image.detach().cpu().numpy().transpose(1, 2, 0))
                axs[0].set_title('Original Image')
                axs[0].axis('off')

                # 显示 feature_mask
                axs[1].imshow(mask.detach().cpu().numpy().squeeze(), cmap='gray')  # 使用 squeeze() 去除多余的维度
                axs[1].set_title('Feature Mask')
                axs[1].axis('off')

                # 显示应用阴影后的图像
                axs[2].imshow(shadowed_image_for_loss.detach().cpu().numpy().transpose(1, 2, 0))
                axs[2].set_title('Shadowed Image')
                axs[2].axis('off')

                axs[3].imshow(shadow_mask.detach().cpu().numpy().squeeze(), cmap='gray')  # 同样使用 squeeze() 去除多余的维度
                axs[3].set_title('Shadow Mask')
                axs[3].axis('off')

                plt.suptitle(f"Iteration {iteration}")
                plt.show()
            print("shadow loss: ")
            print(loss.item())
            if shadow_center.grad is not None and shadow_radius.grad is not None:
                optimizer.step()
            else:
                print("No gradient for shadow parameters at iteration", iteration)
            if device == 'cuda':
                torch.cuda.empty_cache()
            # 保证阴影中心和半径在合理范围内
            with torch.no_grad():
                shadow_center.clamp_(min=0, max=original_image.size(2))
                shadow_radius.clamp_(min=0, max=min(original_image.size(1), original_image.size(2)) / 2)

        return shadow_center.detach(), shadow_radius.detach(), shadowed_image

    def create_shadow_mask(self, image_size, shadow_center, shadow_radius, device):
        """
        创建阴影遮罩。

        Parameters:
        - image_size: 图像大小。
        - shadow_center: 阴影中心。
        - shadow_radius: 阴影半径。

        Returns:
        - 阴影遮罩。
        """
        C, H, W = image_size
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        Y = Y.to(device)
        X = X.to(device)
        dist_from_center = torch.sqrt((X - shadow_center[0]) ** 2 + (Y - shadow_center[1]) ** 2)
        shadow_mask = (dist_from_center <= shadow_radius).float()
        return shadow_mask

    def apply_adversarial_perturbation(self, classifier, image, target_label, device, epsilon=0.00001):
        """
        根据分类器的梯度生成并应用对抗性扰动。

        Parameters:
        - classifier: 分类器模型。
        - image: 输入图像。
        - target_label: 目标标签（我们希望模型错误分类为此标签）。
        - device: cpu/gpu
        - epsilon: 对抗性扰动的强度。

        Returns:
        - 带有对抗性扰动的图像。
        """
        model = classifier.model

        # 需要将图像包装为可求导的变量
        image = image.unsqueeze(0)
        image = image.to(device)
        image.requires_grad = True

        # 正向传播以计算损失
        output = model(image)
        loss = F.cross_entropy(output, target_label)

        # 反向传播以计算图像梯度
        model.zero_grad()
        loss.backward()

        # 收集数据梯度
        data_grad = image.grad.data

        # 使用梯度创建对抗性扰动
        perturbation = epsilon * data_grad.sign()

        # 将扰动应用到原始图像上
        perturbed_image = image + perturbation

        # 保证图像值在有效范围内
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image.detach()  # 需要调用detach()来停止对此变量的梯度跟踪

    def apply_shadow(self, image, shadow_center, shadow_radius, feature_mask, classifier, target_label, device,
                     shadow_intensity=0.33, epsilon=0.01):
        """
        在图像上应用特定位置和大小的阴影，并在特定区域施加对抗性扰动。

        Parameters:
        - image: 输入图像。
        - shadow_center: 阴影中心位置，格式为 (x, y)。
        - shadow_radius: 阴影半径。
        - feature_mask: 特征遮罩，定义了需要施加阴影的特征轮廓。
        - device: cpu/gpu
        - shadow_intensity: 阴影强度，值越大阴影越深。
        - epsilon: 对抗性扰动的强度。

        Returns:
        - 带有阴影和对抗性扰动的图像。
        """
        image = image.to(device)
        feature_mask = feature_mask.to(device)
        C, H, W = image.shape
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        Y = Y.to(device)
        X = X.to(device)
        dist_from_center = torch.sqrt((X - shadow_center[0]) ** 2 + (Y - shadow_center[1]) ** 2)
        shadow_mask = (dist_from_center <= shadow_radius).float().to(image.device)

        # 计算阴影mask2与特征mask的交集
        combined_mask = shadow_mask * feature_mask

        # 在交集区域应用阴影
        shadowed_image = image * (1 - combined_mask) + combined_mask * (image * (1 - shadow_intensity))

        # 在交集区域施加对抗性扰动
        adversarial_image = self.apply_adversarial_perturbation(classifier, shadowed_image, target_label, device,
                                                                epsilon)
        perturbed_shadow_image = image * (1 - combined_mask) + adversarial_image * combined_mask

        # 保证图像值在有效范围内
        perturbed_shadow_image = torch.clamp(perturbed_shadow_image, 0, 1)
        return perturbed_shadow_image

    def train_losses(self, model, x_start, t, device):
        """
        计算训练损失，同时考虑对抗性阴影和DDPM损失。

        Parameters:
        - model: 训练的生成模型
        - classifier: 用于分类的模型
        - x_start: 原始未加噪声的图像
        - t: 时间步
        - shadow_params: 阴影参数，包含阴影中心和半径等
        - target_label: 目标标签
        - device: 设备 (CPU或GPU)

        Returns:
        - 计算得到的总损失
        """
        # DDPM损失：评估噪声重建能力
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t)
        ddpm_loss = F.mse_loss(noise, predicted_noise)

        total_loss = ddpm_loss
        return total_loss

