import torch
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0.0, std=0.1):
    """
    向图像添加可导的高斯噪声。
    
    :param image: 输入图像，类型为tensor，范围[0, 1]
    :param mean: 高斯噪声的均值
    :param std: 高斯噪声的标准差
    :return: 添加高斯噪声后的图像
    """
    # 使用torch生成标准差为std、均值为mean的高斯噪声
    noise = torch.normal(mean=mean, std=std, size=image.size()).to(image.device)
    
    # 添加噪声到图像并保证噪声添加过程是可导的
    noisy_image = image + noise
    
    # 保证图像的值仍然在[0, 1]范围内
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)

    return noisy_image

# 示例使用
if __name__ == "__main__":
    # 假设我们有一个灰度图像
    image = plt.imread('example_image.jpg')  # 假设图像路径是example_image.jpg
    image = image.mean(axis=-1)  # 转换为灰度图
    image = torch.tensor(image, dtype=torch.float32) / 255.0  # 转为[0, 1]范围的Tensor

    # 将图像尺寸调整为PyTorch支持的格式 (batch_size, channels, height, width)
    image = image.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度

    # 添加高斯噪声
    noisy_image = add_gaussian_noise(image, mean=0.0, std=0.1)
    
    # 展示图像
    plt.figure(figsize=(8, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(0).squeeze(0).numpy(), cmap='gray')
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_image.squeeze(0).squeeze(0).detach().numpy(), cmap='gray')
    plt.title("Noisy Image")
    
    plt.show()
