# -*- coding: utf-8 -*-
# env: python=3.6
# author: lm 

python lr_finder.py \
       --train=./tfrecords/cityscapes/1024x2048/train/ \
       --eval=./tfrecords/cityscapes/1024x2048/test/ \
       --net=small_seg \
       --height=1024 \
       --width=2048 \
       --crop_height=512 \
       --crop_width=1024 \
       --num_labels=19 \
       --epoches=100 \
       --eval_epoches=1 \
       --batch_size=12 \
       --eval_batch_size=4 \
       --weight_decay=2e-4 \
       --lr=0.000001 \
       --max_lr=1 \
       --step_size=1600 \
       --ckpt_iters=250 \
       --show_iters=10 \
       --use_bn=1 \
       --class_weights=weights_1024x2048.npy \
       --model_dir=models/mbv3_seg_small_ea_1024x2048 \
       --log_path=logs/mbv3_seg_small_ea_1024x2048_1109.txt \
       --restore=1 \
       --reset_lr=1  

# FILE END.
