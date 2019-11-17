# -*- coding: utf-8 -*- 
# train.py
# env: python=3.6
# author: lm 

import tensorflow as tf 
import os 
import numpy as np 
import time
import math 
from tqdm import tqdm 
import cv2 as cv
import random
import logging # for save train log

from tfrecords import tf_records
from network.data_aug import seq_aug
from network import train_options
from network.mobilenet_v3 import MobileNetv3
from network.lr_policy import cyclic_lr




args = train_options.parser.parse_args()
print (args)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # allocate as need.

logging.basicConfig(filename = args.log_path, level = logging.DEBUG,
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



def train():
    xin = tf.placeholder(dtype = tf.float32,
        shape = [None, args.height, args.width, args.channel],
        name = "xin")
    yin = tf.placeholder(dtype = tf.int32,
        shape = [None, args.num_labels],
        name = "yin")
    is_training = tf.placeholder(dtype = tf.bool)
    global_step = tf.Variable(0, trainable = False, name = "global_step")
    epoch = tf.Variable(0, trainable = False, name = 'epoch')
    regularizer = tf.contrib.layers.l2_regularizer(args.weight_decay)
    # compute graph 
    mbv3 = MobileNetv3(xin, args.num_labels, regularizer, 'mbv3')
    if args.net == 'large':
        logits = mbv3.large()
        test_logits = mbv3.large(False, True)
    elif args.net == 'small':
        logits = mbv3.small()
        test_logits = mbv3.small(False, True)
    print ("logits.shape: {}".format(logits.shape))
    # for evaluation
    scores = tf.nn.softmax(logits)
    classes = tf.argmax(scores, 1)
    test_scores = tf.nn.softmax(test_logits)
    test_class = tf.argmax(test_scores, 1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits = logits,
        labels = yin)

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = cross_entropy_mean + regularizer_loss 
    '''
    learning_rate = tf.train.exponential_decay(
        learning_rate = args.lr,
        global_step = global_step,
        decay_steps = 100,
        decay_rate = args.lr_decay)
    '''
    '''
    learning_rate = tf.train.piecewise_constant(
        epoch, 
        boundaries = [30, 60, 90],
        values = [args.lr, args.lr / 10, args.lr / 100, args.lr / 1000])
    '''
    learning_rate = cyclic_lr(args.lr, args.max_lr, global_step, args.step_size, 'triangular2')
    
    #  TRAIN STEP, to use AdamOptimizer
    if args.use_bn:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_ops = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
            # train_ops = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step)
    else:
        train_ops = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
    # calculate accuracy by once. 
    train_accu = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(yin, 1), tf.argmax(logits, 1)), tf.float32))

    images, labels = tf_records.read_tfrecords_by_data(
        get_file_list(args.train),
        (args.height, args.width), 
        3, 
        batch_size = args.batch_size,
        buffer_size = 256)

    images = tf.image.random_brightness(images, 0.3)
    labels = tf.one_hot(labels, args.num_labels, 1, 0) # should out of session. 
    
    eval_images, eval_labels = tf_records.read_tfrecords_by_data(
        get_file_list(args.eval), 
        (args.height, args.width), 
        3, 
        batch_size = args.eval_batch_size)
    eval_images = tf.cast(eval_images, tf.float32)

    vars_restore = tf.trainable_variables()
    vars_global = tf.global_variables()
    
    bn_moving_vars = [g for g in vars_global if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in vars_global if 'moving_variance' in g.name]
    vars_restore += bn_moving_vars
    vars_restore += [global_step]
    print("*" * 80)
    print("The following vars may be restored.")
    total_param = 0
    for var in vars_restore:
        if 'weights' in var.name:
            print(var.name, var.shape)
            cnt = 1
            for dim in var.shape:
                cnt *= dim
            total_param += cnt
    print("Total params: {}".format(total_param))
    print("*" * 80) 
    saver = tf.train.Saver(vars_restore, max_to_keep=5) # only save trainable vars for smaller checkpoints.
    save_model_path = os.path.join(args.model_dir, args.save_model_name + ".ckpt")
    best_saver = tf.train.Saver(vars_restore, max_to_keep = 1) # for save the best model.
    with tf.Session(config = config) as sess:
        # use tensorboard
        tf.set_random_seed(101)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('train_accuracy', train_accu)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs/', sess.graph)
        if args.restore:
            ckpt = tf.train.get_checkpoint_state(args.model_dir)
            if ckpt and os.path.exists(ckpt.model_checkpoint_path + '.meta'):
                saver.restore(sess, ckpt.model_checkpoint_path)
                vars_init = [var for var in tf.global_variables() if var not in vars_restore]
                init_op = tf.variables_initializer(vars_init)
                # initialize the untrainable variables 
                sess.run(init_op) 
                logging.info("restored from {}".format(ckpt.model_checkpoint_path))
                if args.reset_lr:
                    sess.run(global_step.initializer) # init to zero.
            else:
                sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())
        print("Global step:", sess.run(global_step))
        logging.info("images.shape: {}, labels: {}".format(images.shape, labels))
        logging.info("training...")
        # train loop
        num_examples_train = tf_records.get_number_examples(get_file_list(args.train))
        train_batches_per_epoch = num_examples_train // args.batch_size
        # train_batches_per_epoch = math.ceil(num_examples_train / args.batch_size)
        print("There are {} examples in train dataset. Iterate {} batches per epoch.".format(num_examples_train, train_batches_per_epoch))
        num_examples_val = tf_records.get_number_examples(get_file_list(args.eval))
        eval_batches_per_epoch = num_examples_val // args.eval_batch_size
        batch_begin = sess.run(global_step) % train_batches_per_epoch
        best_val_accu = 0.0  
        saturation_epoch_count = 0 
        for epoch in range(sess.run(global_step) // train_batches_per_epoch, args.epoches):
            train_accuracies = []
            for it in range(batch_begin, train_batches_per_epoch):
                img_batch, label_batch = sess.run([images, labels])
                # use data augmentation.
                img_batch = seq_aug.augment_images(img_batch) 
                img_batch = 2 * img_batch / 255. - 1.0
                train_feed = {xin: img_batch, 
                              yin: label_batch}
                _, loss_value, step, lr, accuracy, summary = sess.run(
                        [train_ops, loss, global_step, learning_rate, train_accu, merged_summary], 
                        feed_dict = train_feed)
                train_accuracies.append(accuracy)
                if (it) % args.show_iters == 0:
                    writer.add_summary(summary, step)
                    cur_time = time.strftime('%H:%M:%S')
                    show_info = "{}, Epoch {}, Step {}, Loss {:.4f}, lr: {:.4e}, train_accu {:.4f}" \
                                .format(cur_time, epoch, it, loss_value, lr, accuracy)
                    print(show_info)
                    logging.info(show_info)
                if (step) % args.ckpt_iters == 0:
                    saver.save(sess, save_model_path, global_step = global_step)
                    show_info = "model saved to {}-{}".format(save_model_path, step)
                    print(show_info)
                    logging.info(show_info)
            show_info = "Train accuracy: {:.4f}".format(np.mean(train_accuracies))
            print(show_info)
            logging.info(show_info)
            # should do evaluation on val_dataset.
            if (epoch) % args.eval_epoches == 0: 
                pred_right = 0.0
                pred_total = 0.0
                for _ in tqdm(range(eval_batches_per_epoch)):
                    eval_image_batch, eval_label_batch = sess.run([eval_images, eval_labels])
                    eval_image_batch = 2 * eval_image_batch / 255. - 1.0
                    eval_feed = {
                        xin: eval_image_batch
                    } # only feed val image
                    pred_classes = sess.run(test_class, feed_dict = eval_feed)
                    equal = (pred_classes == eval_label_batch).astype(np.float32)
                    pred_right += sum(equal)
                    pred_total += args.eval_batch_size
                val_accu = 0.0
                if pred_total > 0:
                    val_accu = float(pred_right) / pred_total
                    show_info = "Epoch {}, Global Step {}, val accuracy: {:.4f} ({}/{})".format(epoch, step, val_accu, pred_right, pred_total)
                    print(show_info)
                    logging.info(show_info)
                if val_accu > best_val_accu:
                    saturation_epoch_count = 0 
                    best_saver_model_path = os.path.join(args.model_dir, 'best', 'best_{}.ckpt'.format(val_accu))
                    show_info = "Epoch {}, val accuracy improved from {} to {}, save best model to {}." \
                                .format(epoch, best_val_accu, val_accu, best_saver_model_path)
                    best_val_accu = val_accu # update best accuracy 
                    best_saver.save(sess, best_saver_model_path, global_step = global_step)
                    
                else:
                    saturation_epoch_count += 1 
                    show_info = "Saturation Epoch {}, val accuracy did not improve from {}.".format(saturation_epoch_count, best_val_accu)
                print(show_info)
                logging.info(show_info)
            batch_begin = 0 # start a new epoch 
            
def main(argv = None):
    print ("main function begin!")
    train()
    print ("optimize done!")

if __name__ == "__main__":
    main() 

# FILE END.
