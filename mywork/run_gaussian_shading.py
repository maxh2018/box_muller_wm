"""
为了调试方便，部分位置用了绝对路径
"""
import copy
from tqdm import tqdm
import torch, sys, os
from transformers import CLIPModel, CLIPTokenizer
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
import open_clip
from optim_utils import *
from io_utils import *
from image_utils import *
from watermark import *
from pydantic import BaseModel
from typing import *
from PIL import Image

parent_dir = os.path.abspath("/home/maxiaohui/box_muller_wm")
sys.path.append(parent_dir)
# from DeamNet.denoise import main as denoise
from MaskedDenoising.denoise import  denoise as denoise_
from MaskedDenoising.denoise import  init, param_denoise

DENOISE = True

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
    DENOISE : bool= False
    





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
        
    if  args.DENOISE is True:
        with open("/home/maxiaohui/box_muller_wm/MaskedDenoising/config/args.json", "r", encoding='utf-8') as f:
            json_dict = json.load(f)
    
        args_denoise = param_denoise(**json_dict)
        denoise_model,window_size = init(args_denoise)
        def denoise(image, imgname=None):
            return denoise_(denoise_model, window_size, image, args_denoise,imgname)

    # dataset
    dataset, prompt_key = get_dataset(args)

    # class for watermark
    args.ch_factor,args.hw_factor= args.channel_copy, args.hw_copy
    if args.chacha:
        watermark = Gaussian_Shading_chacha(args)
    else:
        #这一部分暂未实现
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
        image_w = outputs.images[0] #PIL Image对象
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
        if args.DENOISE is True:  
            image_w_distortion = denoise(image_w_distortion)
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


if __name__ == '__main__':
    dict_args = {}
    with open("/home/maxiaohui/box_muller_wm/mywork/config/args.json", "r", encoding="utf-8") as f:
        dict_args = json.load(f)
    args = param(**dict_args)

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    main(args)
