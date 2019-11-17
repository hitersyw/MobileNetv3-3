# -*- coding: utf-8 -*-
# author: lm
# define training options for this model.

import argparse

parser = argparse.ArgumentParser()

# define options
# for data
parser.add_argument("--train", required = True, type = str, 
    help = "Path to the train data.")
parser.add_argument("--eval", required = True, type = str,
    help = "Path to the eval data.")

# for model input 
parser.add_argument("--height", type = int, default = 224,
    help = "Height of input images.")
parser.add_argument("--width", type = int, default = 224,
    help = "Width of input images.")
parser.add_argument("--channel", type = int, default = 3,
    help = "Channels of input images.")
parser.add_argument("-n", "--num_labels", type = int, default = 1000,
    help = "How many classes to classify.")
parser.add_argument("--crop_height", type = int, default = 512,
    help = "The crop height of sub image for segmentation.")
parser.add_argument("--crop_width", type = int, default = 1024,
    help = "The crop width of sub image for segmentation.")
    
# for learning(training)
parser.add_argument("--net", type = str, default = 'large',
    help = "The network mode, support `large` and `small`.")
parser.add_argument("--epoches", type = int, default = 300,
    help = "Max training epoches.")
parser.add_argument("--lr", type = float, default = 0.001,
    help = "Base learning rate.")
parser.add_argument("--max_lr", type = float, default = 0.006,
    help = "Maximum learning rate for CLR.")
parser.add_argument("--step_size", type = int, default = 2000,
    help = "The step size of CLR.")
parser.add_argument("--lr_decay", type = float, default = 0.99,
    help = "The decay param of learning rate.")
parser.add_argument("--batch_size", type = int, default = 64,
    help = "Batch size used for training model.")
parser.add_argument("--class_weights", type = str, default = "weights.npy",
    help = "The class weights file as numpy format.")
parser.add_argument("--weight_decay", type = float, default = 0.0002,
    help = "The weight decay used for regularization.")
parser.add_argument("--moving_ave_decay", type = float, default = 0.99,
    help = "The moving average decay param for training the moving average model.")
parser.add_argument("--use_bn", type = int, default = 0,
    help = "Whether contain BatchNorm Layer in model.")
parser.add_argument("--restore", type = int, default = 1,
    help = "Whether to restore weights from ckpt if ckpt existed.")
parser.add_argument("--reset_lr", type = int, default = 0,
    help = "if not 0 reset the learning rate to base learning rate, \
           otherwise set the learning rate by global step in the saved model.")

# for evaluation
parser.add_argument("--eval_epoches", type = int, default = 1,
    help = "Evaluation per epoches.")
parser.add_argument("--eval_batch_size", type = int, default = 1,
    help = "Batch size used for evaluation.")

# for experiment
parser.add_argument("--log_path", type = str, default = "./logs/log.txt",
    help = "Path to the log file.")
parser.add_argument("--model_dir", type = str, default = "./models/",
    help = "Path to model folder, where stores the trained folder.")
parser.add_argument("--save_model_name", type = str, default = "model",
    help = "Checkpoint name to save.")
parser.add_argument("--ckpt_iters", type = int, default = 2000,
    help = "Model checkpoint iter for saving model.")
parser.add_argument("--show_iters", type = int, default = 100,
    help = "After how many iters, show training state.")
parser.add_argument("--eval_iters", type = int, default = 2000,
    help = "Evaluation period.")

# FILE END    
