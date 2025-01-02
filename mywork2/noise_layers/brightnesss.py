import torchvision.transforms.functional as F
import torch

def adjust_brightness(img, brightness_factor):
    """
    调整图像亮度，支持梯度计算。

    参数:
        img (torch.Tensor): 输入图像 tensor，形状为 [C, H, W] 或 [B, C, H, W]，值在 [0,1] 之间。
        delta_brightness (float或torch.Tensor): 亮度调整的增量，可以是浮点数或与 img 同批次的 tensor。

    返回:
        torch.Tensor: 亮度调整后的图像 tensor。
    """
    # 确保亮度调整系数是 tensor 类型
    if isinstance(brightness_factor, float):
        brightness_factor = torch.tensor(brightness_factor, device=img.device)
    # 调整亮度
    adjusted_img = F.adjust_brightness(img, brightness_factor)
    # 可选: 裁剪到 [0,1] 范围
    adjusted_img = torch.clamp(adjusted_img, 0, 1)
    return adjusted_img