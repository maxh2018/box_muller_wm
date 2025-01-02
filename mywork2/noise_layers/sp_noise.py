import torch

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob, temperature=0.01):
    """
    向图像添加可导的椒盐噪声。
    
    :param image: 输入图像，PyTorch tensor，形状为 [batch_size, channels, height, width]，值在 [0,1] 之间
    :param salt_prob: 盐噪声的概率
    :param pepper_prob: 胡椒噪声的概率
    :param temperature: 控制掩码接近0或1的程度，值越小，掩码越接近0或1
    :return: 添加椒盐噪声后的图像
    """
    noise_salt = torch.rand_like(image)
    noise_pepper = torch.rand_like(image)
    
    # 计算阈值
    threshold_salt = -torch.log(torch.tensor((1 - salt_prob) / salt_prob))
    threshold_pepper = -torch.log(torch.tensor((1 - pepper_prob) / pepper_prob))
    
    # 生成掩码
    mask_salt = torch.sigmoid((noise_salt - threshold_salt) / temperature)
    mask_pepper = torch.sigmoid((noise_pepper - threshold_pepper) / temperature)
    
    # 避免同一像素同时被设置为盐和胡椒
    mask_pepper = mask_pepper * (1 - mask_salt)
    
    # 添加噪声
    noisy_image = image * (1 - mask_salt - mask_pepper) + mask_salt * 1 + mask_pepper * 0
    return noisy_image

# 测试代码
if __name__ == "__main__":
    # 创建一个简单的图像tensor
    image = torch.randn(1, 1, 8, 8, requires_grad=True)
    image = torch.sigmoid(image)  # 确保图像值在 [0,1] 之间
    
    # 添加椒盐噪声
    noisy_image = add_salt_and_pepper_noise(image, salt_prob=0.1, pepper_prob=0.1)
    
    # 计算损失并反向传播
    loss = noisy_image.mean()
    loss.backward()
    
    # 检查梯度
    print(image.grad)