# -*- coding: utf-8 -*-
# author: lm
"""
This file defines the common layers or layer module.
"""
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib as tf_contrib


def unpack(size):
    """
    params:\n
        size: discrible the size. (size_h, size_w)
    return: size_h, size_w
    """
    assert type(size) in {int, tuple, list}, "please check the type of size used."
    if int == type(size):
        return size, size # h, w
    else:
        assert len(size) > 0 and len(size) < 3, "please check the size. but {} got".format(size)
        if 1 == len(size):
            return size[0], size[0]
        else:
            return size[0], size[1]

def get_weight_variable(name, shape, initializer, regularizer = None, lc = None):
    weights = tf.get_variable(
        name = name,
        shape = shape,
        initializer = initializer)

    if regularizer is not None:
        if lc is None:
            print("add reg loss to", tf.GraphKeys.REGULARIZATION_LOSSES)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer(weights))
        else:
            tf.add_to_collection(lc, regularizer(weights))
    return weights

def conv(name_scope, input_tensor, output_channels, ksize, strides = [1, 1], 
        padding = "SAME", bias = True, trainable = True, weight_init = None, 
        bias_init = tf.zeros_initializer(), regularizer = None, lc = None):
    """
    Convolution layer
    Args:
        weight_init: if weight_init is `None` (the default), the default initializer passed 
                     in the variable scope will be used. If that one is `None` too, a 
                     `glorot_uniform_initializer` also named `Xavier_normal_initializer` 
                     will be used.
    """
    input_channels = input_tensor.get_shape()[-1] # get channles of the input tensor
    kh, kw = unpack(ksize)
    sh, sw = unpack(strides)
    with tf.variable_scope(name_scope):
        weights = tf.get_variable("weights", 
            [kh, kw, input_channels, output_channels], 
            dtype = tf.float32, trainable = trainable, 
            initializer = weight_init)
        if None != regularizer:
            if lc is None:
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer(weights))
            else:
                tf.add_to_collection(lc, regularizer(weights))
        if bias:
            biases = tf.get_variable("biases", 
                [output_channels], dtype = tf.float32, 
                trainable = trainable, initializer = bias_init)
        conv_out = tf.nn.conv2d(input_tensor, weights, [1, sh, sw, 1], padding = padding)
        if bias:
            conv_res = tf.nn.bias_add(conv_out, biases)
            return conv_res 
        else: 
            return conv_out 
   
def depthwise_conv(name_scope, input_tensor, output_channels, ksize, strides = [1, 1],
    padding = "SAME", bias = False, trainable = True, weight_init = None, regularizer = None, 
    lc = None, bias_init = tf.zeros_initializer()):
    """
    Depthwise Convolution layer.
    Args:
    
    """
    input_channels = input_tensor.get_shape()[-1]
    kh, kw = unpack(ksize)
    sh, sw = unpack(strides)
    with tf.variable_scope(name_scope):
        weights = tf.get_variable("weights", [kh, kw, input_channels, output_channels // input_channels],
                dtype = tf.float32, trainable = trainable, initializer = weight_init)
        if None != regularizer:
            if lc is None:
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer(weights))
            else:
                tf.add_to_collection(lc, regularizer(weights))
        conv_out = tf.nn.depthwise_conv2d(input_tensor, weights, [1, sh, sw, 1], padding = padding)
        if bias:
            biases = tf.get_variable("biases", [output_channels], dtype = tf.float32,
                                    trainable = trainable, initializer = bias_init)
            conv_out = tf.nn.bias_add(conv_out, biases)
        return conv_out 
   

# batch normalization implementated by tf.nn.layers.batch_norm 
def bn(name_scope, input_tensor, is_training = True, trainable = True, center = True, 
    scale = True, momentum = 0.99, epsilon = 1e-5):
    """BatchNormalization layer, if trainning model please set 'is_training' to True, else False.
    \n NOTE: if trainable = True, then bn's params will be added to graph collection 
             `tf.GraphKeys.TRAINABLE_VARIABLES`.
    \n How to update bn's params, eg:
    \n x_norm = bn("bn_namescope", x, True, True) # define bn
    \n update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # define update ops.
    \n train_op = tf.train.optimizer.minimize(loss) # define train op
    \n train_op = tf.group([train_op, update_ops]) # group update_ops with train_op
    \n when save parameters of this model, please save the parameters of batch norm layer.
    """
    with tf.variable_scope(name_scope):
        bn_out = tf.layers.batch_normalization(
            input_tensor,
            momentum = momentum,
            epsilon = epsilon,
            training = is_training,
            trainable = trainable,
            center = center,
            scale = scale,
            name = "bn")
        return bn_out

### Batch Normalization
def batch_norm(input_tensor, decay = 0.9, is_training=True, scope='batch_norm', center = True, scale = True):
    """"BatchNorm layer"""
    bn_out =  tf_contrib.layers.batch_norm(input_tensor, decay=decay, epsilon=1e-05, 
    center=center, scale=scale, updates_collections=None, is_training=is_training, scope=scope)
    return bn_out
    
def relu(name_scope, input_tensor):
    """ReLU layer"""
    with tf.variable_scope(name_scope):
        relu_out =  tf.nn.relu(input_tensor)
        return relu_out

def relu6(name_scope, input_tensor):
    """ReLU6 layer"""
    with tf.variable_scope(name_scope):
        relu6_out = tf.nn.relu6(input_tensor)
        return relu6_out

def sigmoid(name_scope, input_tensor):
    with tf.variable_scope(name_scope):
        sigmoid_out = tf.nn.sigmoid(input_tensor)
        return sigmoid_out

'''

'''

def hsigmoid(input_tensor):
    """Hard sigmoid in MobileNetv3"""
    return tf.nn.relu6(input_tensor + 3.0) / 6.0

def hswish(name_scope, input_tensor):
    """Hard swish in MobileNetv3"""
    with tf.variable_scope(name_scope):
        return input_tensor * tf.nn.relu6(input_tensor + 3.0) / 6.0

'''
def hswish(input_tensor):
    with tf.name_scope('hswish'):
        return input_tensor * tf.nn.relu6(input_tensor + np.float32(3)) * np.float32(1. / 6.)
'''

def pool(name_scope, input_tensor, ksize, strides, padding = "SAME", method = "max"):
    """Pooling layer"""
    kh, kw = unpack(ksize)
    sh, sw = unpack(strides)
    with tf.variable_scope(name_scope):
        if "max" == method.lower():
            pooled = tf.nn.max_pool(
                value = input_tensor,
                ksize = [1, kh, kw, 1],
                strides = [1, sh, sw, 1],
                padding = padding)
        else: 
            pooled = tf.nn.avg_pool(
                value = input_tensor,
                ksize = [1, kh, kw, 1],
                strides = [1, sh, sw, 1],
                padding = padding)
        return pooled 

def global_avg_pool(name_scope, input_tensor):
    """Global average pooling layer. Please check the shape of input_tensor is [b, h, w, c]."""
    with tf.variable_scope(name_scope + "/global_ave_pooling"):
        global_avg_pool = tf.reduce_mean(input_tensor, axis = [1, 2], keepdims = True,
                                         name = "global_avg_pool")
        return global_avg_pool

def fc(name_scope, input_tensor, num_output, trainable = True, weight_init = None, 
        bias = True, bias_init = tf.zeros_initializer(), regularizer = None, lc = "losses"):
    """Fully connected layer"""
    input_shape = input_tensor.get_shape().as_list()
    size = 1
    for each in input_shape[1:]:
        size *= each 
    flatten = tf.reshape(input_tensor, [-1, size])
    with tf.variable_scope(name_scope):
        weights = tf.get_variable(
            name = "weights",
            shape = [size, num_output],
            dtype = tf.float32,
            initializer = weight_init,
            trainable = trainable)
        if None != regularizer:
            tf.add_to_collection(lc, regularizer(weights))
        fc_out = tf.matmul(flatten, weights)
        if bias:
            biases = tf.get_variable(
                name = "biases",
                shape = [num_output],
                dtype = tf.float32,
                initializer = bias_init,
                trainable = trainable)
            return fc_out + biases
        else:
            return fc_out 

def concat(name_scope, tensors, axis = -1):
    """
    concat the input tensors by specified axis, by default is the last axis.
    params:
        tensors: a tensor list, which contains the tensors to be concated.
        axis: the concat axis of tensors, default is the last axis which means channel-axis for 4D tensors.
    """
    with tf.variable_scope(name_scope):
        concat_out =  tf.concat(tensors, axis)
        return concat_out

def sum_tensors(name_scope, tensors):
    '''
    sum tensors.
    '''
    with tf.variable_scope(name_scope):
        sum_out =  sum(tensors)
        return sum_out

def interp(name_scope, input_tensor, size, align_corners = True, name = None):
    """
    resize the `input_tensor` to specified `size` using bilinear interpolation.
    """
    with tf.variable_scope(name_scope):
        interp_out =  tf.image.resize_bilinear(input_tensor, size, align_corners, name)
        return interp_out

def dropout(name_scope, input_tensor, rate, is_training = True):
    '''dropout layer
    rate: dropout rate 
    '''
    with tf.variable_scope(name_scope):
        return tf.layers.dropout(inputs = input_tensor, rate = rate, 
                training = is_training, name = name_scope)

# other layer
# space_to_depth

# for more quickly
        
def conv_relu(name_scope, input_tensor, output_channels, ksize, strides,
        padding = "SAME", bias = True, trainable = True, weight_init = None, 
        bias_init = tf.zeros_initializer(), regularizer = None, lc = None):
    """Convolution Layer + ReLU layer"""
    conv_layer = conv(name_scope + "/conv", input_tensor, output_channels, 
        ksize, strides, padding, bias, trainable, 
        weight_init, bias_init, regularizer, lc)
    relu_layer = relu(name_scope + "/relu", conv_layer)
    return relu_layer

def fc_relu(name_scope, input_tensor, num_output, trainable = True, weight_init = None, 
        bias = True, bias_init = tf.zeros_initializer(), regularizer = None, lc = None):
    """Fully connected layer + ReLU layer."""
    fc_layer = fc(name_scope + "/fc", input_tensor, num_output, trainable, weight_init,
        bias, bias_init, regularizer, lc)
    relu_layer = relu(name_scope + "/relu", fc_layer)
    return relu_layer

def conv_bn(name_scope, input_tensor, output_channels, ksize, strides = [1, 1],
        padding = "SAME", bias = False, is_training = True, trainable = True, 
        weight_init = None, bias_init = tf.zeros_initializer(), scale = False, 
        momentum = 0.8999, regularizer = None, lc = None):
    """Convolution Layer + BN layer"""
    conv_layer = conv(name_scope, input_tensor, output_channels,
        ksize, strides, padding, bias, trainable,
        weight_init, bias_init, regularizer, lc)
    conv_layer_norm = batch_norm(conv_layer, decay = momentum, scale = scale,
        is_training=is_training, scope = name_scope + "/bn")
    return conv_layer_norm

def conv_bn_relu(name_scope, input_tensor, output_channels, ksize, strides,
        padding = "SAME", bias = False, is_training = True, trainable = True, 
        weight_init = None, bias_init = tf.zeros_initializer(), scale = False, 
        momentum = 0.999, regularizer = None, lc = None):
    """Convolution Layer + BN layer + ReLU layer"""
    conv_layer = conv(name_scope , input_tensor, output_channels,
        ksize, strides, padding, bias, trainable, 
        weight_init, bias_init, regularizer, lc)
    conv_layer_norm = batch_norm(conv_layer, decay = momentum, scale = scale,
        is_training=is_training, scope = name_scope + "/bn")
    relu_layer = relu(name_scope + "/relu", conv_layer_norm)     
    return relu_layer
    
        

## FOR residual network
# 3x3
# 3x3
def res_block_a(name_scope, input_tensor, output_channels, ksize, strides,
        padding = "SAME", bias = False, is_training = False, trainable = True, 
        weight_init = None, bias_init = tf.zeros_initializer(), regularizer = None, lc = "losses"):
    """
    None-bottle neck residual block.
    """
    # conv3x3 + conv3x3
    conv1 = conv_bn_relu(name_scope + "/conv1_3x3", input_tensor, output_channels, ksize, 
        strides, padding, bias, is_training, trainable, weight_init, bias_init, False,
        0.899999976158, regularizer, lc)
    conv2 = conv_bn(name_scope + "/conv2_3x3", conv1, output_channels, ksize, [1, 1], 
        padding, bias, is_training, trainable, weight_init, bias_init, False,
        0.899999976158, regularizer, lc)
    # if skip:
    # use skip to math output_channels and input channels or space dimension of two different tensor.
    if (output_channels != input_tensor.get_shape()[-1]) or (sum(unpack(strides)) > 2):
        conv_skip = conv_bn(name_scope + "/skip", input_tensor, output_channels, [1, 1], strides,
            padding, bias, is_training, trainable, weight_init, bias_init, False,
            0.899999976158, regularizer, lc)
        res_sum = sum_tensors(name_scope + "/sum", [conv2, conv_skip])
    else:
        res_sum = sum_tensors(name_scope + "/sum", [conv2 + input_tensor]) 
    res_relu = relu(name_scope, res_sum)
    return res_relu 

# 1x1
# 3x3
# 1x1
def res_block_b(name_scope, input_tensor, output_channels, ksize, strides,
        padding = "SAME", bias = False, is_training = True, trainable = True, 
        weight_init = None, bias_init = tf.zeros_initializer(), multiple = 4, 
        regularizer = None, lc = "losses"):
    """
    bottleneck residual block.
    """
    # conv1x1_reduce + conv3x3 + conv1x1_expand
    # stage 1. conv1x1 reduce channel size
    conv_reduce = conv_bn_relu(name_scope + "/reduce", input_tensor, 
        int(output_channels / multiple), [1, 1], [1, 1], padding, bias, is_training, 
        trainable, weight_init, bias_init, False, 0.899999976158, regularizer, lc)
    conv_3x3 = conv_bn_relu(name_scope + "/3x3", conv_reduce, int(output_channels / 4),
        ksize, strides, padding, bias, is_training, trainable, weight_init,
        bias_init, False, 0.899999976158, regularizer, lc)
    conv_expand = conv_bn(name_scope + "/expand", conv_3x3, output_channels, 
        [1, 1], [1, 1], padding, bias, is_training, trainable, weight_init, 
        bias_init, False, 0.899999976158, regularizer, lc)
    # if skip: # need skip layer
    if (output_channels != input_tensor.get_shape()[-1]) or (sum(unpack(strides)) > 2):
        conv_skip = conv_bn(name_scope + "/skip", input_tensor, output_channels, 
            [1, 1], strides, padding, bias, is_training, trainable, weight_init, 
            bias_init, False, 0.899999976158, regularizer, lc)
        res_sum = sum_tensors(name_scope + "/sum", [conv_expand + conv_skip])
    else:
        res_sum = sum_tensors(name_scope + "/sum", [conv_expand + input_tensor])
    res_relu = relu(name_scope, res_sum)
    return res_relu


# for feature fusion.
def fn(name_scope, tensor_h, tensor_l, is_training = False, weight_init = None, 
    bias_init = tf.zeros_initializer(), regularizer = None, lc = "losses"):
    """
    defines the Fusion Node of Partial Order Pruning.
    params:
        tensor_h: shape = (2H, 2W, Ch)
        tensor_l: shape = (H, W, Cl)
    """
    _, hh, wh, ch = tensor_h.shape.as_list()
    conv1 = conv_bn_relu(name_scope + "/1x1", tensor_l, ch, 1, 1, is_training = is_training,
                        weight_init = weight_init, bias_init = bias_init, regularizer = regularizer,
                        lc = lc)
    conv1_upx2 = interp(name_scope + "/interp", conv1, [hh, wh]) # up 2x
    fn_concat = concat(name_scope + "/concat", [tensor_h, conv1_upx2])
    fn = conv_bn_relu(name_scope + "/fusion", fn_concat, ch, [3, 3], [1, 1], is_training = is_training,
                    weight_init = weight_init, bias_init = bias_init, regularizer = regularizer, lc = lc)
    return fn

# PSP pooling
def psp_module(name_scope, input_tensor, num_output, up_size, is_training = False, fm = concat,
        psizes = [[32, 32], [16, 16], [8, 8], [4, 4]],
        strides = [[32, 32], [16, 16], [8, 8], [4, 4]]):
    """
    Structure: pooling + conv + interp + concat/sum, fm specified the feature fuse function.
    """
    psp = []
    for i in range(len(psizes)):
        psp_pool = pool(name_scope + "/pool{}".format(i), input_tensor, psizes[i], strides[i], method = "avg")
        psp_conv = conv_bn_relu(name_scope + "/conv{}".format(i), psp_pool, num_output, [1, 1], [1, 1],
                                is_training = is_training)
        psp_interp = interp(name_scope + "/interp", psp_conv, up_size)
        psp.append(psp_interp)
    if concat == fm:
        psp_fusion = fm(name_scope + "/concat", psp)
    else:
        psp_fusion = fm(name_scope + "/sum", psp)
    return psp_fusion
        


# FILE END.
