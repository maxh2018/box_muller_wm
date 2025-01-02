import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def rotate_image(image, angle : float):
    """
    对图像进行旋转，旋转角度可导。

    参数:
        image (torch.Tensor): 输入图像 tensor，形状为 (B, C, H, W)。
        angle (torch.Tensor): 旋转角度 tensor，形状为 (B,)，单位为度。

    返回:
        torch.Tensor: 旋转后的图像 tensor，形状与输入相同。
    """
    # 将角度从度转换为弧度
    angle = torch.tensor([angle], requires_grad=True)  # 旋转角度为45度
    angle_rad = torch.deg2rad(angle)
    # 创建旋转矩阵
    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)
    rotation_matrix = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1).view(-1, 2, 2)
    # 创建仿射网格
    grid = F.affine_grid(rotation_matrix, image.size(), align_corners=False)
    # 应用网格采样
    rotated_image = F.grid_sample(image, grid, align_corners=False)
    return rotated_image

# 测试代码
if __name__ == "__main__":
    # 创建测试图像
    image = None
    angle = torch.tensor([45.0], requires_grad=True)  # 旋转角度为45度

    # 旋转图像
    rotated_image = rotate_image(image, angle)

    # 可视化原始图像和旋转后的图像
    img_np = image.squeeze(0).permute(1, 2, 0).detach().numpy()
    rot_img_np = rotated_image.squeeze(0).permute(1, 2, 0).detach().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(rot_img_np)
    plt.title(f'Rotated Image ({angle.item()} degrees)')
    plt.show()

    # 验证可导性
    loss = rotated_image.mean()
    loss.backward()

    # 检查梯度
    print("Image gradient:", image.grad)
    print("Angle gradient:", angle.grad)