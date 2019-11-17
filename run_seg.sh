# -*- coding: utf-8 -*-
# env: python=3.6
# author: lm 

python train_mbv3_seg.py \
       --train=./tfrecords/cityscapes/1024x2048/train/ \
       --eval=./tfrecords/cityscapes/1024x2048/test/ \
       --net=large_seg \
       --height=1024 \
       --width=2048 \
       --crop_height=512 \
       --crop_width=1024 \
       --num_labels=19 \
       --epoches=100 \
       --eval_epoches=1 \
       --batch_size=12 \
       --eval_batch_size=1 \
       --weight_decay=2e-4 \
       --lr=0.000001 \
       --max_lr=0.000006 \
       --step_size=1482 \
       --ckpt_iters=247 \
       --show_iters=10 \
       --use_bn=1 \
       --class_weights=class_weights/weights_1024x2048.npy \
       --model_dir=models/mbv3_seg_large_drop0.2_ea_1024x2048 \
       --log_path=logs/mbv3_seg_large_drop0.2_ea_1024x2048_1108.txt \
       --restore=1 \
       --reset_lr=1  

# FILE END.
