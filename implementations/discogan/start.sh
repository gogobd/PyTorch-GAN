#!/bin/sh

python3 discogan.py --dataset_name pix --batch_size 4 --n_cpu 20 --img_width 512 --img_height 512 --sample_interval 100 --checkpoint_interval 1 --lambda_gan 25 --lambda_cyc 20 --lambda_pix 1.5 --lambda_id 1.5 --n_epoch 500 --plot_interval 50 --epoch 352 --load_model 352_2651735
