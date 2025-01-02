import torch
import torch.nn as nn

# 定义带有残差连接的卷积块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 确保输入与输出形状相同，如果需要，可以使用1x1卷积调整通道数
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # 残差连接
        out = self.relu(out)
        return out


class Encoder(nn.Module):  # 4*64*64
    def __init__(self):
        super(Encoder, self).__init__()

        # 编码部分，使用残差连接的卷积块
        self.conv1 = ResidualBlock(4, 32)  # 4 * 64 * 64 -> 32 * 64 * 64
        self.conv2 = ResidualBlock(32, 32)  # 32 * 64 * 64 -> 32 * 64 * 64
        self.conv3 = ResidualBlock(32, 64, stride=2)  # 32 * 64 * 64 -> 64 * 32 * 32
        self.conv4 = ResidualBlock(64, 128, stride=2)  # 64 * 32 * 32 -> 128 * 16 * 16
        self.conv5 = ResidualBlock(128, 256, stride=2)  # 128 * 16 * 16 -> 256 * 8 * 8

        # 解码部分（上采样并融合）
        self.up6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 256 * 8 * 8 -> 128 * 16 * 16
        self.conv6 = ResidualBlock(256, 128)  # 128 * 16 * 16 -> 128 * 16 * 16

        self.up7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 128 * 16 * 16 -> 64 * 32 * 32
        self.conv7 = ResidualBlock(128, 64)  # 64 * 32 * 32 -> 64 * 32 * 32

        self.up8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 64 * 32 * 32 -> 32 * 64 * 64
        self.conv8 = ResidualBlock(64, 32)  # 32 * 64 * 64 -> 32 * 64 * 64

        # 最后的卷积层，输出为最终的 residual
        self.conv9 = nn.Conv2d(68, 32, kernel_size=3, padding=1)
        self.residual = nn.Conv2d(32, 4, kernel_size=1, padding=0)  # 输出4个通道，保持尺寸一致

    def forward(self, image):
        inputs = image  # 输入图像 4 * 64 * 64
        
        # 编码器部分
        conv1 = self.conv1(inputs)  # 32 * 64 * 64
        conv2 = self.conv2(conv1)   # 32 * 64 * 64
        conv3 = self.conv3(conv2)   # 64 * 32 * 32
        conv4 = self.conv4(conv3)   # 128 * 16 * 16
        conv5 = self.conv5(conv4)   # 256 * 8 * 8

        # 解码器部分，跳跃连接与上采样
        up6 = self.up6(conv5)  # 128 * 16 * 16
        merge6 = torch.cat([conv4, up6], dim=1)  # 拼接 128 * 16 * 16 + 128 * 16 * 16 = 256 * 16 * 16
        conv6 = self.conv6(merge6)  # 128 * 16 * 16

        up7 = self.up7(conv6)  # 64 * 32 * 32
        merge7 = torch.cat([conv3, up7], dim=1)  # 拼接 64 * 32 * 32 + 64 * 32 * 32 = 128 * 32 * 32
        conv7 = self.conv7(merge7)  # 64 * 32 * 32

        up8 = self.up8(conv7)  # 32 * 64 * 64
        merge8 = torch.cat([conv2, up8], dim=1)  # 拼接 32 * 64 * 64 + 32 * 64 * 64 = 64 * 64 * 64
        conv8 = self.conv8(merge8)  # 32 * 64 * 64

        up9 = self.up9(conv8)  # 32 * 64 * 64
        merge9 = torch.cat([conv1, up9, inputs], dim=1)  # 拼接 32 * 64 * 64 + 32 * 64 * 64 + 4 * 64 * 64 = 68 * 64 * 64
        conv9 = self.conv9(merge9)  # 32 * 64 * 64

        residual = self.residual(conv9)  # 4 * 64 * 64 (输出与输入相同的尺寸)
        
        return residual


# 测试代码
if __name__ == "__main__":
    model = Encoder()  # 创建模型
    print(model)

    # 测试网络的前向传播
    x = torch.randn(1, 4, 64, 64)  # 假设输入图像大小为64x64，通道数为4
    output = model(x)
    print("Output shape:", output.shape)  # 输出形状应该是(1, 4, 64, 64)
