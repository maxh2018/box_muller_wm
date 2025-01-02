python /home/maxiaohui/box_muller_wm/MaskedDenoising/image_denoise1228.py \
        --model_path model_zoo/input_mask_80_90.pth \
        --name input_mask_80_90/McM_poisson_20 \
        --opt model_zoo/input_mask_80_90.json \
        --folder_gt /home/maxiaohui/box_muller_wm/MaskedDenoising/testset/McM/truth \
        --folder_lq /home/maxiaohui/box_muller_wm/MaskedDenoising/testset/McM/Low
#把图片放到testset/McM/McM_poisson_20路径下面，得到的去噪图像在results中进行的储存。其他代码是官网的。