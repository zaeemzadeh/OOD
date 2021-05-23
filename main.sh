#!/usr/bin/env bash
gpu_flag=1
path='UoS'
rate=0.1


# # Training
CUDA_VISIBLE_DEVICES=$gpu_flag python train.py --name $path --droprate $rate

# # Testing
out_dataset='Imagenet_crop'
CUDA_VISIBLE_DEVICES=$gpu_flag python extract_features.py  --path $path --out_dataset $out_dataset  --droprate $rate
python evaluate.py --path $path --out_dataset $out_dataset

out_dataset='Imagenet_resize'
CUDA_VISIBLE_DEVICES=$gpu_flag python extract_features.py  --path $path --out_dataset $out_dataset  --droprate $rate --no_in_dataset
python evaluate.py --path $path --out_dataset $out_dataset

out_dataset='LSUN_crop'
CUDA_VISIBLE_DEVICES=$gpu_flag python extract_features.py  --path $path --out_dataset $out_dataset  --droprate $rate --no_in_dataset
python evaluate.py --path $path --out_dataset $out_dataset

out_dataset='LSUN_resize'
CUDA_VISIBLE_DEVICES=$gpu_flag python extract_features.py  --path $path --out_dataset $out_dataset  --droprate $rate --no_in_dataset
python evaluate.py --path $path --out_dataset $out_dataset
