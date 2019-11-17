# -*- coding: utf-8 -*- 
# env: python=3.6
# author: lm 

import tensorflow as tf 
from network import layers as L 

def squeeze_excite(name_scope, x, reduction = 4):
    with tf.variable_scope(name_scope):
        in_size     = x.get_shape()[-1]
        gap         = L.global_avg_pool('gap', x)
        conv1_relu1 = L.conv_relu('squeeze', gap, in_size // reduction, (1, 1), (1, 1))
        conv2       = L.conv("excite", conv1_relu1, in_size, (1, 1), (1, 1))
        hsigmoid1   = L.hsigmoid(conv2)
        return x * hsigmoid1 

def conv_bn_relu6(name_scope, x, out_size, ksize, strides, padding = "SAME", 
    bias = False, is_training = True, trainable = True, weight_init = None, 
    bias_init = tf.zeros_initializer(), scale = False, momentum = 0.99, 
    regularizer = None, lc = None, nl = 'RE'):
    '''Convolution + BatchNorm + ReLU6'''
    conv_layer = L.conv(name_scope , x, out_size,
        ksize, strides, padding, bias, trainable, 
        weight_init, bias_init, regularizer, lc)
    bn_out = L.bn(name_scope, conv_layer, is_training, momentum = 0.99, epsilon = 1e-5)
    if 'RE' == nl:
        relu_out = L.relu(name_scope, bn_out)
    elif 'HS' == nl:
        relu_out = L.hswish(name_scope, bn_out)
    elif 'R6' == nl:
        relu_out = L.relu6(name_scope, bn_out)
    else:
        print("nolinear layer {} is not implementate yeat.".format(nl))
    return relu_out 
    
def conv_bn_drop_relu6(name_scope, x, out_size, ksize, strides, padding = "SAME", 
    bias = False, is_training = True, trainable = True, weight_init = None, 
    bias_init = tf.zeros_initializer(), scale = False, momentum = 0.99, 
    regularizer = None, lc = None, rate = 0.0, nl = 'RE'):
    '''Convolution + BatchNorm + DropOut + ReLU6'''
    conv_layer = L.conv(name_scope , x, out_size,
        ksize, strides, padding, bias, trainable, 
        weight_init, bias_init, regularizer, lc)
    bn_out = L.bn(name_scope, conv_layer, is_training, momentum = 0.99, epsilon = 1e-5)
    drop_out = L.dropout(name_scope, bn_out, is_training, rate)
    if 'RE' == nl:
        relu_out = L.relu(name_scope, drop_out)
    elif 'HS' == nl:
        relu_out = L.hswish(name_scope, drop_out)
    elif 'R6' == nl:
        relu_out = L.relu6(name_scope, drop_out)
    else:
        print("nolinear layer {} is not implementate yeat.".format(nl))
    return relu_out 

def depthwise_bn_relu6(name_scope, x, out_size, ksize, strides, padding = "SAME", 
    bias = False, is_training = True, trainable = True, weight_init = None, 
    bias_init = tf.zeros_initializer(), scale = False, momentum = 0.99, 
    regularizer = None, lc = None, nl = 'RE'):
    '''DeptwiseConvolution + BatchNorm + ReLU6'''
    depthwise_conv_layer = L.depthwise_conv(name_scope, x, out_size,
        ksize, strides, padding, bias, trainable, weight_init, regularizer, lc)
    bn_out = L.bn(name_scope, depthwise_conv_layer, 
        is_training, momentum = momentum, epsilon = 1e-5)
    if 'RE' == nl:
        relu_out = L.relu(name_scope, bn_out)
    elif 'HS' == nl:
        relu_out = L.hswish(name_scope, bn_out)
    elif 'R6' == nl:
        relu_out = L.relu6(name_scope, bn_out)
    else:
        print("nolinear layer {} is not implementate yeat.".format(nl))
    return relu_out 

def depthwise_bn(name_scope, x, out_size, ksize, strides, padding = "SAME", 
    bias = False, is_training = True, trainable = True, weight_init = None, 
    bias_init = tf.zeros_initializer(), scale = False, momentum = 0.99, 
    regularizer = None, lc = None):
    '''DepthwiseConvolution + BatchNorm'''
    depthwise_conv_layer = L.depthwise_conv(name_scope, x, out_size,
        ksize, strides, padding, bias, trainable, weight_init, regularizer, lc)
    bn_out = L.bn(name_scope, depthwise_conv_layer, 
        is_training, momentum = momentum, epsilon = 1e-5)
    return bn_out

def bneck(name, x, ksize, exp_size, out_size, se = 'F', nl = 'RE', strides = 1, it = True):
    in_size = x.get_shape()[-1]
    conv1 = conv_bn_relu6(name+'/expand', x, exp_size, (1, 1), (1, 1), is_training = it, nl = nl)
    conv2 = depthwise_bn_relu6(name+"/depthwise", conv1, exp_size, ksize, strides, is_training = it, nl = nl)
    if se == 'T':
        conv2 = squeeze_excite(name, conv2)
    conv4 = conv_bn_relu6(name+"/reduce", conv2, out_size, (1, 1), (1, 1), is_training = it, nl = nl)
    if 1 == strides and in_size == out_size:
        # whether add dropout in bnet when training.
        conv4 = L.dropout(name, conv4, 0.2, it) # whether is training 
        return x + conv4
    else:
        return conv4


######################################## MobileNetV3 ##################################################
class MobileNetv3:
    def __init__(self, num_labels, reg = None, name = 'mbv3'):
        self.num_labels = num_labels 
        self.name = name 
        self.reg = reg
        self.weight_init = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.09)
    
        
    def large(self, inputs, is_training = True, reuse = False):
        with tf.variable_scope(self.name+'_large', reuse = reuse, initializer = self.weight_init, regularizer = self.reg):
            #      ksize, exp_size, out_size, SE, NL,  # 224x224x3  -> 112x112x16   16
            BNETS = [[(3, 3), 16,  16,  'F', 'RE', 1], # 112x112x16 -> 112x112x16   16
                     [(3, 3), 64,  24,  'F', 'RE', 2], # 112x112x16 -> 56x56x24     8 
                     [(3, 3), 72,  24,  'F', 'RE', 1], # 56x56x24   -> 56x56x24     8
                     [(5, 5), 72,  40,  'T', 'RE', 2], # 56x56x24   -> 28x28x40     4
                     [(5, 5), 120, 40,  'T', 'RE', 1], # 28x28x40   -> 28x28x40     4
                     [(5, 5), 120, 40,  'T', 'RE', 1], # 28x28x40   -> 28x28x40     4
                     [(3, 3), 240, 80,  'F', 'HS', 2], # 28x28x40   -> 14x14x80     2
                     [(3, 3), 200, 80,  'F', 'HS', 1], # 14x14x80   -> 14x14x80     2
                     [(3, 3), 184, 80,  'F', 'HS', 1], # 14x14x80   -> 14x14x80     2
                     [(3, 3), 184, 80,  'F', 'HS', 1], # 14x14x80   -> 14x14x80     2
                     [(3, 3), 480, 112, 'T', 'HS', 1], # 14x14x80   -> 14x14x112    2
                     [(3, 3), 672, 112, 'T', 'HS', 1], # 14x14x112  -> 14x14x112    1
                     [(5, 5), 672, 160, 'T', 'HS', 2], # 14x14x112  -> 7x7x160      1
                     [(5, 5), 960, 160, 'T', 'HS', 1], # 7x7x160    -> 7x7x160      1
                     [(5, 5), 960, 160, 'T', 'HS', 1]] # 7x7x160    -> 7x7x160      1
            x = conv_bn_relu6('conv1', inputs, 16, (3, 3), (2, 2), is_training = is_training, nl = 'HS')
            for idx, (ksize, exp_size, out_size, se, nl, strides) in enumerate(BNETS):
                name = "bneck{}".format(idx+1)
                x = bneck(name, x, ksize, exp_size, out_size, se, nl, strides, is_training)
            x = conv_bn_relu6('conv2', x, 960, (1, 1), (1, 1), is_training = is_training, nl = 'HS')
            x = L.global_avg_pool('gap', x)
            x = L.conv('conv3', x, 1280, (1, 1), (1, 1))
            x = L.hswish('conv3/hswich', x)
            x = L.dropout('dropout', x, 0.2, is_training = is_training)
            x = L.conv('conv4', x, self.num_labels, (1, 1), (1, 1))
            x = tf.squeeze(x, [1, 2])
            return x 
            
    def small(self, inputs, is_training = True, reuse = False):
        with tf.variable_scope(self.name+'_small', reuse = reuse, initializer = self.weight_init, regularizer = self.reg):
            #           k     e    o    SE    NL   s   # 224x224x3  -> 112x12x16   16   index
            BNETS = [[(3, 3), 16,  16,  'T', 'RE', 2], # 112x112x16 -> 56x56x16    8    0
                     [(3, 3), 72,  24,  'F', 'RE', 2], # 56x56x16   -> 28x28x24    4    1
                     [(3, 3), 88,  24,  'F', 'RE', 1], # 28x28x24   -> 28x28x24    4    2   +
                     [(5, 5), 96,  40,  'T', 'HS', 2], # 28x28x24   -> 14x14x40    2    3   
                     [(5, 5), 240, 40,  'T', 'HS', 1], # 14x14x40   -> 14x14x40    2    4   +
                     [(5, 5), 240, 40,  'T', 'HS', 1], # 14x14x40   -> 14x14x40    2    5   +
                     [(5, 5), 120, 48,  'T', 'HS', 1], # 14x14x40   -> 14x14x48    2    6
                     [(5, 5), 144, 48,  'T', 'HS', 1], # 14x14x48   -> 14x14x48    2    7   +
                     [(5, 5), 288, 96,  'T', 'HS', 2], # 14x14x48   -> 7x7x96      1    8   
                     [(5, 5), 576, 96,  'T', 'HS', 1], # 7x7x96     -> 7x7x96      1    9   +
                     [(5, 5), 576, 96,  'T', 'HS', 1]] # 7x7x96     -> 7x7x96      1    10  +
            
            x = conv_bn_relu6('conv1', inputs, 16, (3, 3), (2, 2), is_training = is_training, nl = 'HS')
            for idx, (ksize, exp_size, out_size, se, nl, strides) in enumerate(BNETS):
                name = "bneck{}".format(idx+1)
                x = bneck(name, x, ksize, exp_size, out_size, se, nl, strides, is_training)
            x = conv_bn_relu6('conv2', x, 576, (1, 1), (1, 1), is_training = is_training, nl = 'HS')
            x = L.global_avg_pool('gap', x)
            x = L.conv('conv3', x, 1024, (1, 1), (1, 1)) # should be 1024
            # dropout ? 
            x = L.dropout('dropout', x, 0.2, is_training = is_training)
            x = L.hswish('conv3/hswich', x)
            x = L.conv('conv4', x, self.num_labels, (1, 1), (1, 1))
            x = tf.squeeze(x, [1, 2])
            return x 
            
    def LR_ASPP(self, x1, x2, num, it = True):
        '''
        Args:
            x1: outputs of 1/16 resolution, for upper stream.
            x2: outputs of 1/8 resolution, for bottom stream.
            num: number of labels.
            it: is_training.
        '''
        with tf.variable_scope("seg_head"):
            # upper stream
            _, H, W, _ = x1.get_shape()
            x11 = conv_bn_relu6('conv1', x1, 128, (1, 1), (1, 1), is_training = it, nl = 'RE')
            pool1 = L.pool('pool', x1, (49, 49), (16, 20), method = 'avg')
            x12 = L.conv('conv2', pool1, 128, (1, 1), (1, 1))
            sig = L.sigmoid('sigmoid', x12)
            up1 = L.interp('up1', sig, (H, W))
            fused = x11 * up1 
            up2 = L.interp('up2', fused, (2*H, 2*W))
            x13 = L.conv('conv3', up2, num, (1, 1), (1, 1))
            # bottom stream
            x21 = L.conv('conv4', x2, num, (1, 1), (1, 1))
            seg = x13 + x21 
            return seg  
            
        
    def small_seg(self, inputs, is_training = True, reuse = False):
        with tf.variable_scope(self.name+"_small_seg", reuse = reuse, initializer = self.weight_init, regularizer = self.reg):
            BNET1 = [[(3, 3), 16,  16,  'T', 'RE', 2], # 1/4  
                     [(3, 3), 72,  24,  'F', 'RE', 2], # 1/8
                     [(3, 3), 88,  24,  'F', 'RE', 1]] # 1/8
                     
            BNET2 = [[(5, 5), 96,  40,  'T', 'HS', 2], # 1/16 
                     [(5, 5), 240, 40,  'T', 'HS', 1], #  
                     [(5, 5), 240, 40,  'T', 'HS', 1], #  
                     [(5, 5), 120, 48,  'T', 'HS', 1], # 
                     [(5, 5), 144, 48,  'T', 'HS', 1], # 
                     [(5, 5), 288, 96,  'T', 'HS', 1], # 1/16, modify stride to 1, then got 1/16 resolution.
                     [(5, 5), 576, 96,  'T', 'HS', 1], # 
                     [(5, 5), 576, 96,  'T', 'HS', 1]] # 
            x = conv_bn_relu6('conv1', inputs, 16, (3, 3), (2, 2), is_training = is_training, nl = 'HS')
            for idx, (ksize, exp_size, out_size, se, nl, strides) in enumerate(BNET1):
                name = "bneck{}".format(idx+1)
                x = bneck(name, x, ksize, exp_size, out_size, se, nl, strides, is_training)
            x2 = x 
            for idx, (ksize, exp_size, out_size, se, nl, strides) in enumerate(BNET2):
                name = "bneck{}".format(idx + len(BNET1) + 1)
                x = bneck(name, x, ksize, exp_size, out_size, se, nl, strides, is_training)
            seg = self.LR_ASPP(x, x2, self.num_labels, is_training)
            _, H, W, _ = seg.get_shape()
            seg_logits = L.interp('up3', seg, (8*H, 8*W))
            return seg_logits
            
    def large_seg(self, inputs, is_training = True, reuse = False):
        with tf.variable_scope(self.name+"_large_seg", reuse = reuse, initializer = self.weight_init, regularizer = self.reg):
            BNET1 = [[(3, 3), 16,  16,  'F', 'RE', 1], # 1/2
                     [(3, 3), 64,  24,  'F', 'RE', 2], # 1/4 
                     [(3, 3), 72,  24,  'F', 'RE', 1], # 1/4
                     [(5, 5), 72,  40,  'T', 'RE', 2], # 1/8
                     [(5, 5), 120, 40,  'T', 'RE', 1], # 1/8
                     [(5, 5), 120, 40,  'T', 'RE', 1]] # 1/8
                     
            BNET2 = [[(3, 3), 240, 80,  'F', 'HS', 2], # 1/16
                     [(3, 3), 200, 80,  'F', 'HS', 1], # 1/16
                     [(3, 3), 184, 80,  'F', 'HS', 1], # 1/16
                     [(3, 3), 184, 80,  'F', 'HS', 1], # 1/16
                     [(3, 3), 480, 112, 'T', 'HS', 1], # 1/16
                     [(3, 3), 672, 112, 'T', 'HS', 1], # 1/16
                     [(5, 5), 672, 160, 'T', 'HS', 1], # 1/16, modify stride to 1, then got 1/16 resolution.
                     [(5, 5), 960, 160, 'T', 'HS', 1], # 1/16
                     [(5, 5), 960, 160, 'T', 'HS', 1]] # 1/16
            x = conv_bn_relu6('conv1', inputs, 16, (3, 3), (2, 2), is_training = is_training, nl = 'HS')
            for idx, (ksize, exp_size, out_size, se, nl, strides) in enumerate(BNET1):
                name = "bneck{}".format(idx+1)
                x = bneck(name, x, ksize, exp_size, out_size, se, nl, strides, is_training)
            x2 = x 
            for idx, (ksize, exp_size, out_size, se, nl, strides) in enumerate(BNET2):
                name = "bneck{}".format(idx + len(BNET1) + 1)
                x = bneck(name, x, ksize, exp_size, out_size, se, nl, strides, is_training)
            seg = self.LR_ASPP(x, x2, self.num_labels, is_training)
            _, H, W, _ = seg.get_shape()
            seg_logits = L.interp('up3', seg, (8*H, 8*W))
            return seg_logits
    
if __name__ == "__main__":
    x = tf.zeros([128, 224, 224, 3])
    mbv3 = MobileNetv3(x, 1000)
    small_logits = mbv3.small()
    print(small_logits.shape)
    
    vars_restore = tf.trainable_variables()
    vars_global = tf.global_variables()
    
    bn_moving_vars = [g for g in vars_global if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in vars_global if 'moving_variance' in g.name]
    vars_restore += bn_moving_vars
    print("*" * 80)
    print("The following vars may be restored.")
    total_param = 0
    for var in vars_restore:
        print(var.name, var.shape)
        cnt = 1
        for dim in var.shape:
            cnt *= dim
        total_param += cnt
    print("Total params: {}".format(total_param))
    print("*" * 80) 
# FILEEND.