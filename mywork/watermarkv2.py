import torch
from scipy.stats import norm,truncnorm
from functools import reduce
from scipy.special import betainc
import numpy as np
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
from pydantic import BaseModel
from typing import *

class  param(BaseModel):
    ch_factor: Optional[int]
    hw_factor: Optional[int]
    fpr: Optional[float]
    user_number: Optional[int]
    channel: int = 4
    height: int = 64
    width: int = 64
    length: int = 1


class Gaussian_Shading_chacha:
    def __init__(self, para : param):
        self.ch = para.ch_factor
        self.hw = para.hw_factor
        self.l = para.length
        self.nonce = None
        self.key = None
        self.U2 = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * para.user_number
            if fpr_onebit <= para.fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= para.fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    def stream_key_encrypt(self, sd):
        self.key = get_random_bytes(32)
        self.nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0**self.l
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(0,self.latentlength,self.l):
            #截取比特串中一段长度为l的子串，转换为十进制
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + self.l])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        if self.l == 1:
            z = torch.from_numpy(z).reshape(1, 4, 64, 64).to(dtype=torch.float32)
        else:
            z = torch.from_numpy(z).reshape(1, 4, 64, 64, self.l).to(dtype=torch.float32)
        
        return z.cuda()

    #下面函数将离散值转成连续值，记为连续化
    def DAC(self, message):#DAC:Digital to Analog Converter
        l = len(message)
        #产生长度为l的[0,1]均匀分布的随机数
        u = np.random.rand(l)
        denominator = int(2.0**self.l)
        ret = (message + u) / denominator
        return ret

    #利用Box-Muller方法生成高斯分布随机数
    def Box_Muller(self, ret): #ret: np.array
        # np.random.seed(0)
        self.U2 = np.random.rand(len(ret))
        # self.U2 = np.random.uniform(0, 1, len(ret))
        #计算ret的以e为底的对数
        R = -2*np.log(ret) 
        #计算ret的余弦值
        theta = 2*np.pi*self.U2
        #计算高斯分布随机数
        Z0 = np.sqrt(R)*np.cos(theta)
        # Z1 = np.sqrt(R)*np.sin(theta)
        if self.l == 1:
            z = torch.from_numpy(Z0).reshape(1, 4, 64, 64).half()#.to(dtype=torch.float32)
        else:
            z = torch.from_numpy(Z0).reshape(1, 4, 64, 64, self.l).half()#.to(dtype=torch.float32)
        return z
        
    
    def create_watermark_and_return_w(self):
        if self.l == 1:
            self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()
            sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
        else:
            self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw, self.l]).cuda()
            sd = self.watermark.repeat(1,self.ch,self.hw,self.hw,1)
        m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
        m = self.DAC(m)
        w = self.Box_Muller(m)
        #产生一个形状为(1,4,64,64)的标准高斯分布连续取值噪声
        # w = torch.randn(1, 4, 64, 64).half()
        
        return w

    def stream_key_decrypt(self, reversed_m):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        if self.l == 1:
            sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, 64).to(torch.uint8)
        else:
            sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, 64, self.l).to(torch.uint8)
        return sd_tensor.cuda()

    def extract_watermark(self, reversed_w):
        #将tensor转换为numpy数组
        reversed_w = reversed_w.cpu().numpy()
        U2 = self.U2.reshape((1, 4, 64, 64))
        denominator = np.cos(2*np.pi*U2)
        #将reverse_w的每一个元素转换除以denominator，然后平方
        R = (reversed_w / denominator)**2
        #计算e的-1/2 *R次方
        U1 = np.exp(-0.5*R)
        x = (2.0**self.l*U1).astype(int)
        x = x
        return x
    def diffusion_inverse(self,watermark_r):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_w):
        reversed_m = self.extract_watermark(reversed_w)  #(reversed_w > 0).int()
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten())
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count




class Gaussian_Shading:
    def __init__(self, ch_factor, hw_factor, fpr, user_number):
        self.ch = ch_factor
        self.hw = hw_factor
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.cuda()

    def create_watermark_and_return_w(self):
        self.key = torch.randint(0, 2, [1, 4, 64, 64]).cuda()
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()
        sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
        m = ((sd + self.key) % 2).flatten().cpu().numpy()
        w = self.truncSampling(m)
        return w

    def diffusion_inverse(self,watermark_sd):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_sd, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_m):
        reversed_m = (reversed_m > 0).int()
        reversed_sd = (reversed_m + self.key) % 2
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count


