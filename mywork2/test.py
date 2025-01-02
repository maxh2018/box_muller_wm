# import argparse
import copy
from tqdm import tqdm
import torch,json
from transformers import CLIPModel, CLIPTokenizer
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
import open_clip
from optim_utils import *
from io_utils import *
from image_utils import *
from watermark_train import *
from pydantic import BaseModel
from typing import *
from PIL import Image
from model import Encoder as decoder
import torch.nn.functional as F
import logging
import utils
from model_unet_res import Encoder as modify_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class param(param):
    device: str = 'cuda'
    model_path: str = 'stabilityai/stable-diffusion-2-1-base'
    reference_model: Optional[str] = None
    reference_model_pretrain: Optional[str] = None
    dataset_path: str = 'Gustavosta/Stable-Diffusion-Prompts'
    channel_copy: int = 1
    hw_copy: int = 8
    fpr: float = 0.000001
    user_number: int = 1000000
    output_path: str = './output/'
    chacha: bool = False
    num: int = 1000
    image_length: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    num_inversion_steps: Optional[int] = None
    gen_seed: int = 0
    jpeg_ratio: Optional[int] = None
    random_crop_ratio: Optional[float] = None
    random_drop_ratio: Optional[float] = None
    gaussian_blur_r: Optional[int] = None
    median_blur_k: Optional[int] = None
    resize_ratio: Optional[float] = None
    gaussian_std: Optional[float] = None
    sp_prob: Optional[float] = None
    brightness_factor: Optional[float] = None
    save_image: bool = False
    save_distortion: bool = False
    epoch: int = 10
    experiment_name: str = 'box_muller_wm'
    eval_steps : int = 100
    save_steps :int = 1000
    resume: Union[str,bool] = False
    load_checkpoint: Union[str,bool] = False





def main():
   
    device = "cuda"
    scheduler = DPMSolverMultistepScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            scheduler=scheduler,
            torch_dtype=torch.float32,
            revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    unet = pipe.unet
    
    # 保存 unet 的模型参数
    torch.save(unet.state_dict(), "/home/maxiaohui/box_muller_wm/mywork2/model_dict/unet_state_dict.pth")

    # 此时，unet_weights.pth 包含了 unet 的所有模型参数。

    # 如果需要加载保存的权重，可以使用以下代码：

    # 加载保存的 unet 参数
    # state_dict = torch.load("./model_dict/unet_weights.pth")
    # unet.load_state_dict(state_dict)
    
    # 获取 unet 配置
    unet_config = unet.config
    import json
    # 保存配置为 JSON 文件
    
    with open("/home/maxiaohui/box_muller_wm/mywork2/model_dict/unet_config.json", "w") as f:
        json.dump(unet_config, f)
        
    from diffusers import UNet2DConditionModel
    import json

    # 加载配置
    with open("/home/maxiaohui/box_muller_wm/mywork2/model_dict/unet_config.json", "r") as f:
        unet_config = json.load(f)

    # 重建模型
    unet = UNet2DConditionModel(**unet_config)
    
    # 方法 2：保存完整结构代码

    # 你也可以直接保存 unet 对象为 PyTorch 的完整模型文件，这样既包含参数也包含模型结构。

    # 保存完整模型（结构 + 参数）
    torch.save(unet, "/home/maxiaohui/box_muller_wm/mywork2/model_dict/unet_model.pth")

    # 加载时：

    # 加载完整模型
    unet = torch.load("/home/maxiaohui/box_muller_wm/mywork2/model_dict/unet_model.pth")
    
if __name__ == '__main__':
    # args = param()
    main()