# -*- coding: utf-8 -*-
# env: python=3.6
# author: lm 

python train_mbv3.py \
       --train=./tfrecords/cifar-100/224x224/train/ \
       --eval=./tfrecords/cifar-100/224x224/test/ \
       --net=small \
       --height=224 \
       --width=224 \
       --num_labels=100 \
       --epoches=100 \
       --eval_epoches=1 \
       --batch_size=256 \
       --eval_batch_size=100 \
       --weight_decay=2e-4 \
       --lr=0.0001 \
       --max_lr=0.0006 \
       --step_size=1170 \
       --ckpt_iters=195 \
       --show_iters=10 \
       --use_bn=1 \
       --model_dir=models/mbv3_small_ea_cifar100_224x224 \
       --log_path=logs/mb3_small_ea_224x224_cifar100_1105.txt \
       --restore=1 \
       --reset_lr=1  

# FILE END.
