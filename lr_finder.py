# -*- coding: utf-8 -*- 
# train.py
# env: python=3.6
# author: lm 

import time
import math
import os
import random
import logging # for save train log
import numpy as np 
import tensorflow as tf 
from tfrecords import tf_records
from tqdm import tqdm 
import cv2 as cv
from skimage import color 
from matplotlib import pyplot as plt 

from network import seg_loss
from network import seg_accuracy
from network import lr_policy 
from network.data_aug import seg_augmentation
from network.cityscapes_colors import id2colors, draw_label
from network import train_options
from network.mobilenet_v3 import MobileNetv3



args = train_options.parser.parse_args()
print (args)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # allocate as need.

logging.basicConfig(filename = 'lr_finder', level = logging.DEBUG,
    format = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logging.info("Save log to `{}`.".format(args.log_path))
logging.info(args)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)


def get_file_list(root_dir, exts = 'tfr'):
    if os.path.isfile(root_dir) and root_dir.lower().endswith(exts):
        return [root_dir]
    raw_files = os.listdir(root_dir)
    tfrs = []
    for each in raw_files:
        if each.lower().endswith(exts):
            tfrs.append(os.path.join(root_dir, each))
    return tfrs


np.random.seed(seed = 101)


class LRFinder:
    def __init__(self):
        self.losses = [] # record loss 
        self.should_stop = False 
        self.lrs = [] # record lr 
        self.best_loss = 1e9 # init to very big 
        self.regularizer = tf.contrib.layers.l2_regularizer(args.weight_decay)
        self.global_step = tf.Variable(0, trainable = False, name = "global_step")
        self.class_weights = seg_loss.load_class_weights(args.class_weights)
        self.IGNORE_LABEL = 255 
        self.epochs = 1 
        self.load_data()
        self.load_model()
        self.find() 
        
    def load_data(self):
        self.xin = tf.placeholder(dtype = tf.float32,
            shape = [None, args.crop_height, args.crop_width, 3],
            name = "xin")
        self.yin = tf.placeholder(dtype = tf.int32, 
            shape = [None, args.crop_height, args.crop_width], name = "yin")
        self.images, self.labels = tf_records.read_tfrecords_by_data_v2(
            get_file_list(args.train),
            (args.height, args.width), 
            3, 
            batch_size = args.batch_size,
            buffer_size = 256,
            label_type = 'segmentation')
        self.num_examples = tf_records.get_number_examples(get_file_list(args.train))
        self.train_batches_per_epoch = self.num_examples // args.batch_size
        self.lr_mult = tf.constant((args.max_lr / args.lr) ** (1. / self.train_batches_per_epoch))
        print("lr mult:", self.lr_mult)
        self.learning_rate = lr_policy.finder_lr(args.lr, self.lr_mult, self.global_step)
        print("There are {} examples in train dataset. Iterate {} steps per epoch.".format(self.num_examples, self.train_batches_per_epoch))
        
            
    def load_model(self):
        # you should load your model here, define the loss 
        mbv3 = MobileNetv3(args.num_labels, self.regularizer, 'mbv3')
        if 'large_seg' == args.net:
            self.logits = mbv3.large_seg(self.xin)
        elif 'small_seg' == args.net:
            self.logits = mbv3.small_seg(self.xin)
        print("logits' shape:", self.logits.shape)
        cross_entropy_mean = seg_loss.weighted_sparse_softmax_cross_entropy_with_logits(
            self.yin, self.logits, self.class_weights, self.IGNORE_LABEL)
        regularizer_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = cross_entropy_mean + regularizer_loss
        if args.use_bn:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_ops = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, self.global_step)
        else:
            self.train_ops = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, self.global_step)
            
    def find(self):
        with tf.Session() as sess:
            if args.restore:
                ckpt = tf.train.get_checkpoint_state(args.model_dir)
                if ckpt and os.path.exists(ckpt.model_checkpoint_path + '.meta'):
                    vars_restore = tf.trainable_variables()
                    vars_global = tf.global_variables()
                    bn_moving_vars = [g for g in vars_global if 'moving_mean' in g.name]
                    bn_moving_vars += [g for g in vars_global if 'moving_variance' in g.name]
                    vars_restore += bn_moving_vars
                    vars_restore += [self.global_step]
                    self.saver = tf.train.Saver(vars_restore, max_to_keep=5)
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                    vars_init = [var for var in tf.global_variables() if var not in vars_restore]
                    init_op = tf.variables_initializer(vars_init)
                    # initialize the untrainable variables 
                    sess.run(init_op) 
                    logging.info("restored from {}".format(ckpt.model_checkpoint_path))
                    if args.reset_lr:
                        sess.run(self.global_step.initializer) # init to zero.
                else:
                    sess.run(tf.global_variables_initializer())
            else:
                sess.run(tf.global_variables_initializer())
            print("Starting from global step:", sess.run(self.global_step))
            for epoch in range(self.epochs):
                if self.should_stop:
                    break 
                for it in range(self.train_batches_per_epoch):
                    img_batch, lab_batch = sess.run([self.images, self.labels])
                    img_batch, lab_batch = seg_augmentation(img_batch, lab_batch, args.crop_height, args.crop_width) 
                    img_batch = (img_batch / 255. - 0.5) * 2
                    train_feed = {
                        self.xin: img_batch,
                        self.yin: lab_batch
                    }
                    _, loss_value, lr = sess.run([self.train_ops, self.loss, self.learning_rate], feed_dict = train_feed)
                    # do something on batch end. 
                    if it > 5 and math.isnan(loss_value) or loss_value > self.best_loss * 4:
                        self.should_stop = True 
                        break 
                    print("it: {}, loss: {:.4f}, lr: {:.10f}".format(it, loss_value, lr))
                    self.lrs.append(lr)
                    self.losses.append(loss_value)
                    # update best_loss 
                    if loss_value < self.best_loss:
                        self.best_loss = loss_value
                    
                    
            # after training 
            best_lr = self.get_best_lr(sma = 20)
            print("got best lr:", best_lr)
            self.plot_loss_change(sma = 20)
            # self.plot_loss_change(sma)
            
    def get_derivatives(self, sma = 1):
        '''sma: smooth moving average'''
        assert sma >= 1 
        derivatives = [0] * sma 
        for i in range(sma, len(self.lrs)):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        return derivatives
    
    def get_best_lr(self, sma = 5, n_skip_begining = 10, n_skip_end = 5):
        derivatives = self.get_derivatives(sma)
        best_der_idx = np.argmin(derivatives[n_skip_begining : - n_skip_end])
        print("best_der_idx", best_der_idx)
        return self.lrs[n_skip_begining : - n_skip_end][best_der_idx]
        
        
    def plot_loss_change(self, sma, n_skip_begining = 10, n_skip_end = 5):
        derivatives = self.get_derivatives(sma)[n_skip_begining : - n_skip_end]
        lrs = self.lrs[n_skip_begining : - n_skip_end]
        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(lrs, derivatives)
        plt.xscale('log')
        plt.ylim((min(derivatives) - 0.01, max(derivatives) + 0.01))
        # plt.show() 
        # save the picture 
        plt.savefig('lr_finder.jpg')


if __name__ == "__main__":
    lrfinder = LRFinder()
    

# FILE END.
