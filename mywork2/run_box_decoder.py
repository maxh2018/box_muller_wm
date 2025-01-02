# import argparse
import copy
from tqdm import tqdm
import torch
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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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





def main(args: param):
   
    device = args.device
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    #reference model for CLIP Score
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model,
                                                                                  pretrained=args.reference_model_pretrain,
                                                                                  device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    # class for watermark
    args.ch_factor,args.hw_factor= args.channel_copy, args.hw_copy
    if args.chacha:
        watermark = Gaussian_Shading_chacha(args)
    else:
        #a simple implement,
        watermark = Gaussian_Shading(args)

    os.makedirs(args.output_path, exist_ok=True)

    # assume at the detection time, the original prompt is unknown
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    #acc
    acc = []
    #CLIP Scores
    clip_scores = []

    #test
    for i in tqdm(range(args.num)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        #generate with watermark
        set_random_seed(seed)
        init_latents_w = watermark.create_watermark_and_return_w()
        outputs = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
        )
        image_w = outputs.images[0]
        # img = pipe.numpy_to_pil(image_w)
        #保存图像
        if args.save_image:
            image_w.save(args.output_path + 'image_w_' + str(i) + '.png')

        # distortion
        image_w_distortion,type_info = image_distortion(image_w, seed, args)
        if args.save_distortion:
            savepath = os.path.join(args.output_path, 'distortion')
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            image_w_distortion.save(savepath + f'/image_distortion_{type_info}' + str(i) + '.png')

        # reverse img
        image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.num_inversion_steps,
        )

        #acc metric
        acc_metric = watermark.eval_watermark(reversed_latents_w)
        acc.append(acc_metric)

        #CLIP Score
        if args.reference_model is not None:
            socre = measure_similarity([image_w], current_prompt, ref_model,
                                              ref_clip_preprocess,
                                              ref_tokenizer, device)
            clip_socre = socre[0].item()
        else:
            clip_socre = 0
        clip_scores.append(clip_socre)

    #tpr metric
    tpr_detection, tpr_traceability = watermark.get_tpr()
    #save metrics
    save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores)

def loss_fn(x, y):
    #计算张量a和b之间的均方误差和l1损失
    mse_loss = F.mse_loss(x, y)
    l1_loss = F.l1_loss(x, y)
    return mse_loss + l1_loss
def cal_acc(x, gt):
    #二值化
    x = (x > 0.5).float()
    #计算准确率
    acc = (x == gt).float().mean()
    return acc
    

def train(args: param):
    
    device = args.device
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float32,
            # revision='fp16',
    )
    unet_decoder = torch.load("/home/maxiaohui/box_muller_wm/mywork2/model_dict/unet_model.pth")
    pipe.decoder_unet = unet_decoder
    pipe.safety_checker = None
    pipe = pipe.to(device)

    #reference model for CLIP Score
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model,
                                                                                  pretrained=args.reference_model_pretrain,
                                                                                  device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    # class for watermark
    args.ch_factor,args.hw_factor= args.channel_copy, args.hw_copy
    if args.chacha:
        watermark = Gaussian_Shading_chacha(args)
    else:
        #a simple implement,
        watermark = Gaussian_Shading(args)

    os.makedirs(args.output_path, exist_ok=True)

    # assume at the detection time, the original prompt is unknown
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    model = pipe.decoder_unet
    #acc
    acc = []
    #CLIP Scores
    clip_scores = []
    # if args.resume:
    #     #从断点继续训练
    #     model = modify_model.to(device)
    #     optimizer = torch.optim.Adam(model.parameters())
    #     checkpoint = torch.load(args.resume)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # elif args.load_checkpoint:
    #     #加载预训练参数继续训练
    #     model = modify_model().to(device)
    #     checkpoint = torch.load(args.load_checkpoint)
    #     model.load_state_dict(checkpoint)
    #     #加载AdamW优化器
    #     # optimizer = torch.optim.AdamW(model.parameters())
    #     optimizer = torch.optim.Adam(model.parameters())
    # else:
    #     #从头开始训练
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    import wandb
    wandb.login(key="e03222d8dab298e41bce668868777103a0031750")
    wandb.init(project=args.experiment_name, dir= "./wandb",config=args)
    for i in tqdm(range(args.epoch)):
        for j in tqdm(range(args.num)):
            optimizer.zero_grad()
            seed = j + args.gen_seed
            current_prompt = dataset[j][prompt_key]

            #generate with watermark
            set_random_seed(seed)
            init_latents_w= watermark.create_watermark_and_return_w().to(device)#利用box-muller生成的随机高斯噪声
            outputs = pipe(
                current_prompt,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=init_latents_w,
            )
            image_w = outputs.images[0]#.clone().detach().requires_grad_(True).to(device)#梯度传递从这里开始

            # distortion
            #设计随机加噪声，加旋转
            # distortion = {
            #     # "jpeg_ratio": [90,70],
            #     # "random_crop_ratio": [0.9,0.7,0.5],
            #     # "random_drop_ratio": [0.1,0.3],
            #     "identity": [0],
            #     "gaussian_blur_r": [2,4],
            #     # "median_blur_k": [3,7],
            #     "gaussian_std": [0.05,0.1,0.15],
            #     "sp_prob": [0.05,0.1,0.15],  
            #     # "resize_ratio": [0.9,0.7],
            #     "brightness_factor": [2,4],
            #     "rotate": [0,15,30,45],
            # }
            # possibility ={"identity": 0.4, "gaussian_blur_r": 0.1, "gaussian_std": 0.3, "sp_prob": 0.1, "brightness_factor": 0.1}
            # #按照possibility中的概率随机选择噪声种类和噪声强度
            # noise_type = random.choices(list(possibility.keys()), weights=list(possibility.values()))[0]
            # noise_intensity = random.choice(distortion[noise_type])
            # image_w_distortion,type_info = distort_image(image_w, noise_type, noise_intensity)
            image_w_distortion,type_info = image_distortion(image_w, seed, args)
            image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).clone().detach().requires_grad_(True).to(device)#梯度传递从这里开始
            image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False).to(dtype=torch.float32)
            image_latents_w = image_latents_w#.clone().detach().requires_grad_(True).to(device)#梯度传递从这里开始
            wm_logits = pipe.forward_diffusion_unet(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.num_inversion_steps,
        )
            init_latents_w = init_latents_w.clone().detach().requires_grad_(True).to(device)
            loss = loss_fn(wm_logits, init_latents_w)
            acc = watermark.eval_watermark(wm_logits.clone().detach())
            logger.info(f'Epoch {i}, Step {j}, Loss: {loss.item()}, Acc: {acc}, TPR:')
            loss = loss.to(dtype=torch.float32)
            loss.backward()
            optimizer.step()
            steps = i * args.num + j
            train_log = {
                'epoch': i,
                'loss': loss.item(),
                'decode_acc': acc
            }
            wandb.log(train_log, step=steps)
            #评估
            if  args.eval_steps is not None and steps>0 and steps % args.eval_steps == 0:
                eval_log = eval(model, args, pipe,steps)
                wandb.log(eval_log, step=steps)
                model.train()
            else:
                pass
            if args.save_steps is not None and steps>0 and steps % args.save_steps == 0:
                logging.info('saving checkpoints on  steps {}'.format(steps))
                ##保存模型权重
                checkpoint = {
                        'epoch': i,
                        "steps": steps,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                }
                utils.save_checkpoint(model, args.experiment_name, i, steps, os.path.join(args.output_path, 'checkpoints'), optimizer, loss)
        #save model
        torch.save(model.state_dict(), os.path.join(args.output_path, args.experiment_name,  f'checkpoints/model_epoch_{i}.pth'))
    wandb.finish()


def eval(model, args, pipe,steps=0):
    model.eval()
    if args.chacha:
        watermark = Gaussian_Shading_chacha(args)
    else:
        #a simple implement,
        watermark = Gaussian_Shading(args)
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # os.makedirs(args.output_path, exist_ok=True)
    dataset, prompt_key = get_dataset(args)
    Loss,Acc = [],[]
    for i in tqdm(range(args.num,args.num+100)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]
        set_random_seed(seed)
        init_latents_w= watermark.create_watermark_and_return_w().to(args.device)
        outputs = pipe(
                current_prompt,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=init_latents_w,
            )
        image_w = outputs.images[0]

            # distortion
            #设计随机加噪声，加旋转
        distortion = {
                "jpeg_ratio": [90,70],
                "random_crop_ratio": [0.9,0.7,0.5],
                "random_drop_ratio": [0.1,0.3],
                "gaussian_blur_r": [2,4],
                "median_blur_k": [3,7],
                "gaussian_std": [0.05,0.1,0.15],
                "sp_prob": [0.05,0.1,0.15],  
                "resize_ratio": [0.9,0.7],
                "brightness_factor": [2,4],
            }
        image_w_distortion,type_info = image_distortion(image_w, seed, args)

        # reverse img
        image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(args.device)
        image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False).to(dtype=torch.float32)
            #DDIM_Inversion被省略，直接用decoder获取高斯噪声
        image_latents_w = image_latents_w.clone().detach().requires_grad_(True).to(args.device)#梯度传递从这里开始
        wm_logits = model(image_latents_w)#解码出来的高斯噪声对应 init_latents_w
        loss = loss_fn(wm_logits, init_latents_w)
            # acc = cal_acc(wm_logits, msg_bits)
        acc = watermark.eval_watermark(wm_logits.clone().detach())
        Loss.append(loss.item())
        Acc.append(acc)
    
    eval_step = steps//args.eval_steps
    loss,acc = float(sum(Loss)/len(Loss)),float(sum(Acc)/len(Acc))
    logger.info(f'Eval_Step {eval_step}, Loss: {loss}, Acc: {acc}, TPR:')
    eval_log = {
                'eval_step': eval_step,
                'eval_loss': loss,
                'eval_acc': acc
        }
    return eval_log
        

if __name__ == '__main__':
    dict_args = {}
    with open("/home/maxiaohui/box_muller_wm/mywork/config/args_train_new.json", "r", encoding="utf-8") as f:
        dict_args = json.load(f)
    args = param(**dict_args)

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps
    train(args)
    # main(args)
