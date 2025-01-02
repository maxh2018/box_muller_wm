import sys
# sys.path.append("PerceptualSimilarity\\")
import os
# import utils
import torch
import numpy as np
from torch import nn
import torchgeometry
from kornia import color
import torch.nn.functional as F
import warnings
from options import HiDDenConfiguration
warnings.filterwarnings('ignore')
# from unet import unet_parts as UNet
from torchvision import transforms


class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs): #torch.Size([1, 32, 32, 32])
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU()(outputs)#inplace=True
            else:
                raise NotImplementedError
        return outputs


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class Encoder(nn.Module):#4*64*64
    def __init__(self):
        super(Encoder, self).__init__()
        # self.secret_dense = Dense(256, 1024, activation='relu', kernel_initializer='he_normal')###

        self.conv1 = Conv2D(4, 32, 3, activation='relu')
        self.conv2 = Conv2D(32, 32, 3, activation='relu', strides=2)
        self.conv3 = Conv2D(32, 64, 3, activation='relu', strides=2)
        self.conv4 = Conv2D(64, 128, 3, activation='relu', strides=2)
        self.conv5 = Conv2D(128, 256, 3, activation='relu', strides=2)
        self.up6 = Conv2D(256, 128, 3, activation='relu')
        self.conv6 = Conv2D(256, 128, 3, activation='relu')
        self.up7 = Conv2D(128, 64, 3, activation='relu')
        self.conv7 = Conv2D(128, 64, 3, activation='relu')
        self.up8 = Conv2D(64, 32, 3, activation='relu')
        self.conv8 = Conv2D(64, 32, 3, activation='relu')
        self.up9 = Conv2D(32, 32, 3, activation='relu')
        self.conv9 = Conv2D(68, 32, 3, activation='relu')
        self.residual = Conv2D(32, 4, 1, activation=None)

    def forward(self, image):#, secrect):# 4*32*32

        inputs = image # 4*64*64 #torch.cat([secrect_enlarged, image], dim=1)
        conv1 = self.conv1(inputs)#32*64*64
        conv2 = self.conv2(conv1)#32*32*32
        conv3 = self.conv3(conv2)#64*16*16
        conv4 = self.conv4(conv3)#128*8*8
        conv5 = self.conv5(conv4)#256*4*4
        up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5))#128*8*8
        merge6 = torch.cat([conv4, up6], dim=1)#256*8*8
        conv6 = self.conv6(merge6)#128*8*8
        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6)) #64*16*16
        merge7 = torch.cat([conv3, up7], dim=1) #128*16*16
        conv7 = self.conv7(merge7) #64*16*16
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7)) #32*32*32
        merge8 = torch.cat([conv2, up8], dim=1) #64*32*32
        conv8 = self.conv8(merge8) #32*32*32
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))#32*64*64
        merge9 = torch.cat([conv1, up9, inputs], dim=1)#68*64*64
        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)
        return residual




class SpatialTransformerNetwork(nn.Module):
    def __init__(self, ):#config: HiDDenConfiguration
        super(SpatialTransformerNetwork, self).__init__()
        self.localization = nn.Sequential(#4*64*64
            Conv2D(4, 32, 3, strides=2, activation='relu'), #32*32*32
            Conv2D(32, 64, 3, strides=2, activation='relu'),#64*16*16
            Conv2D(64, 128, 3, strides=2, activation='relu'),#128*8*8
            Flatten(),#
            Dense(8192, 128, activation='relu'),
            nn.Linear(128, 6)
        )
        self.localization[-1].weight.data.fill_(0)
        self.localization[-1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, image): # 4*32*32
        theta = self.localization(image)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        transformed_image = F.grid_sample(image, grid, align_corners=False)
        return transformed_image


class Decoder(nn.Module):
    def __init__(self, config: HiDDenConfiguration, secret_size=256):
        super(Decoder, self).__init__()
        self.secret_size = secret_size
        self.stn = SpatialTransformerNetwork()
        self.decoder = nn.Sequential(#4*64*64
            Conv2D(4, 32, 3, strides=2, activation='relu'),#32*32*32
            Conv2D(32, 32, 3, activation='relu'),#32*32*32
            Conv2D(32, 64, 3, strides=2, activation='relu'),#64*16*16
            Conv2D(64, 64, 3, activation='relu'),#64*16*16
            Conv2D(64, 64, 3, strides=2, activation='relu'),#64*8*8
            Conv2D(64, 128, 3, strides=2, activation='relu'),#128*4*4
            Conv2D(128, 128, 3, strides=2, activation='relu'),#128*2*2
            Flatten(),#
            Dense(128, 512, activation='relu'),#
            Dense(512, secret_size, activation=None))

    def forward(self, image):
        image = image - .5
        transformed_image = self.stn(image)
        return torch.sigmoid(self.decoder(transformed_image))




