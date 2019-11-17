# -*- coding: utf-8 -*- 
# env: python=3.6
# author: lm 

import tensorflow as tf 
import numpy as np 

def load_class_weights(weights_path):
    weights = np.load(weights_path).tolist()
    # to tensor 
    class_weights = tf.constant(weights, tf.float32)
    print("Load class weights:", weights)
    return class_weights 



def weighted_sparse_softmax_cross_entropy_with_logits(labels, logits, class_weights = None, ignore_label = 255):
    '''
    Args:
        labels: (N, H, W)
        logits: (N, H, W, num_class)
    '''
    shape = tf.shape(logits) # (N, H, W, num_class)
    N, H, W, C = tf.unstack(shape, axis = 0)
    
    labels_flatten = tf.reshape(labels, [-1])
    logits = tf.reshape(logits, [N*H*W, C])
    
    not_ignore = tf.not_equal(labels_flatten, ignore_label)
    labels_flatten = tf.boolean_mask(labels_flatten, not_ignore)
    logits = tf.boolean_mask(logits, not_ignore)
    
    
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = labels_flatten,
        logits = logits)
    if class_weights is not None:
        weights = tf.gather(class_weights, labels_flatten)
        losses = tf.reduce_mean(losses * weights, axis = 0)
    else:
        losses = tf.reduce_mean(losses, axis = 0)
    # get loss 
    return losses 


def ohem_weighted_sparse_softmax_cross_entropy_with_logits(
        labels, logits, preds, class_weights = None, ignore_label = 255,
        thres = 0.7, min_kept = 100000):
    '''
    Using OHEM 
    '''
    
    pass 

# FILE END.