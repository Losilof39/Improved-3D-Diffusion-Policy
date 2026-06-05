# bash scripts/vis_dataset.sh

dataset_path=/home/luke/MSMD/repos/Improved-3D-Diffusion-Policy/Improved-3D-Diffusion-Policy/data/g1_empty_bucket_v2

vis_cloud=1
cd Improved-3D-Diffusion-Policy
python vis_dataset.py --dataset_path $dataset_path \
                    --use_img 1 \
                    --vis_cloud ${vis_cloud} \
                    --use_pc_color 0 \
                    --downsample 0 \