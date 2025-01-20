export data_Flickr8k_path="/data/FinAi_Mapping_Knowledge/TOG3/maxiaohui/cache/FID数据集/Flicker8k_Dataset"
export data_COCO_path="/data/FinAi_Mapping_Knowledge/TOG3/maxiaohui/cache/FID数据集/val2017"
export coco_saved_npz="/home/maxiaohui/box_muller_wm/mywork/FID/coco/asaved.npz"
export flickr8k_saved_npz="/home/maxiaohui/box_muller_wm/mywork/FID/Flickr8k/asaved.npz"

export generated_COCO_with_wm="/home/maxiaohui/box_muller_wm/mywork/results/output_0118_coco_with_wm_0"
export generated_Flickr8k_with_wm="/home/maxiaohui/box_muller_wm/mywork/results/output_0118_Flickr8k_with_wm_0"
export generated_COCO_without_wm="/home/maxiaohui/box_muller_wm/mywork/results/output_0118_coco_without_wm_0"
export generated_Flickr8k_without_wm="/home/maxiaohui/box_muller_wm/mywork/results/output_0118_Flickr8k_without_wm_0"

export generated_COCO_with_wm_npz="/home/maxiaohui/box_muller_wm/mywork/results/output_0118_coco_with_wm_0/asaved.npz"
export generated_Flickr8k_with_wm_npz="/home/maxiaohui/box_muller_wm/mywork/results/output_0118_Flickr8k_with_wm_0/asaved.npz"
export generated_COCO_without_wm_npz="/home/maxiaohui/box_muller_wm/mywork/results/output_0118_coco_without_wm_0/asaved.npz"
export generated_Flickr8k_without_wm_npz="/home/maxiaohui/box_muller_wm/mywork/results/output_0118_Flickr8k_without_wm_0/asaved.npz"


# python /home/maxiaohui/box_muller_wm/mywork/pytorch_fid/fid_score.py  "$generated_COCO_without_wm" /home/maxiaohui/box_muller_wm/mywork/results/output_0118_coco_without_wm_0/asaved.npz  --device cuda:2 --save-stats
# python /home/maxiaohui/box_muller_wm/mywork/pytorch_fid/fid_score.py  "$generated_Flickr8k_without_wm" /home/maxiaohui/box_muller_wm/mywork/results/output_0118_Flickr8k_without_wm_0/asaved.npz  --device cuda:1 --save-stats

python /home/maxiaohui/box_muller_wm/mywork/pytorch_fid/fid_score.py  "$coco_saved_npz" "$generated_COCO_with_wm_npz"  --device cuda:1
python /home/maxiaohui/box_muller_wm/mywork/pytorch_fid/fid_score.py  "$coco_saved_npz" "$generated_COCO_without_wm_npz"  --device cuda:1

python /home/maxiaohui/box_muller_wm/mywork/pytorch_fid/fid_score.py  "$flickr8k_saved_npz" "$generated_Flickr8k_with_wm_npz"  --device cuda:2
python /home/maxiaohui/box_muller_wm/mywork/pytorch_fid/fid_score.py  "$flickr8k_saved_npz" "$generated_Flickr8k_without_wm_npz"  --device cuda:2

# python /home/maxiaohui/box_muller_wm/mywork/pytorch_fid/fid_score.py  /home/maxiaohui/box_muller_wm/mywork/output_0114_with_wm/saved.npz /home/maxiaohui/box_muller_wm/mywork/output_0114_with_wm/saved.npz  --device cuda:0 