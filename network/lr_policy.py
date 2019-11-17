# -*- coding: utf-8 -*- 
# env: python3.6
# author: lm 

import tensorflow as tf 
import numpy as np 


def fixed_lr(base_lr):
    '''
    Args:
        base_lr: the specified base learning rate.
    Return:
        lr: a TF tensor which value is specified by the base learning rate.
    '''
    return tf.cast(base_lr, tf.float32)
    
def step_lr(base_lr, step_size, global_step, gamma = 0.1):
    '''
    Args:
        base_lr: the specified base learning rate.
        step: learning decay step.
        global_step: the global step should passed in.
        gamma: learning rate will decayed by multiply gamma.
    '''
    return base_lr * (gamma ** (global_step // step_size))

def step_lr_by_epoch(base_lr, epoch_size, global_epoch, gamma):
    return base_lr * (gamma ** (global_epoch // epoch_size))

def multistep_lr(base_lr, step_sizes, global_step, gamma):
    '''
    Args:
        base_lr: the base learning rate.
        steps: a list which specified the step to decay learning rate.
        global_step: the global step should be passed in.
        gamma: learning rate will decayed by multiply gamma.
    Return:
        lr: the updated learning rate.
    '''
    pass 
    
def multistep_lr_by_epoch(base_lr, epoch_sizes, global_eopch, gamma):
    pass 
    
def exp_lr(base_lr, global_step, gamma):
    return tf.cast(base_lr * (gamma ** global_step), tf.float32)

def poly_lr(base_lr, global_step, max_step, power):
    return tf.cast(base_lr * (1 - global_step / max_step) ** power, tf.float32)

def inv_lr(base_lr, global_step, gamma, power):
    return tf.cast(base_lr * (1 + gamma * global_step) ** (-power), tf.float32)


def sigmoid_lr(base_lr, step_size, global_step, gamma):
    return tf.cast(base_lr * (1 / (1 + exp(-gamma * (global_step - step_size)))), tf.float32)


def finder_lr(lr, lr_mult, global_step):
    return lr * tf.pow(lr_mult, tf.cast(global_step, dtype = tf.float32))

def cyclic_lr(base_lr, max_lr, clr_iterations, step_size = 2000, mode = 'triangular', gamma = 1.,
    scale_fn = None, scale_mode = 'cycle'):
    '''
    https://arxiv.org/abs/1506.01186
    Args:
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr:upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: 'triangular', 'triangular2', 'exp_range'.
            `triangular`:
                A basic triangular cycle w/ no amplitude scaling.
            `triangular2`:
                A basic triangular cycle that scales initial amplitude by half each cycle.
            `exp_range`:
                A cycle that scales initial amplitude by gamma**(cycle iterations) at each
                cycle iteration.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
        
        For more detail, please see paper.
    Return:
        updated learning rate.
    '''
    if scale_fn is None:
        if mode == 'triangular':
            scale_fn = lambda x: 1.0
            scale_mode = 'cycle'
        elif mode == 'triangular2':
            # scale half by each ecpch
            scale_fn = lambda x: 1 / (2. ** (x - 1))
            scale_mode = 'cycle'
        elif mode == 'exp_range':
            scale_fn = lambda x: gamma ** (x)
            scale_mode = 'iterations'
    else:
        # using the specified scale_fn and scale_mode.
        pass 
    cycle = tf.cast(tf.floor(1 + clr_iterations / (2 * step_size)), tf.float32)
    x = tf.abs(tf.cast(clr_iterations / step_size, tf.float32) - 2 * cycle + 1)
    if scale_mode == 'cycle':
        print(cycle)
        return base_lr + (max_lr - base_lr) * tf.maximum(0.0, (1 - x) * scale_fn(cycle))
    else:
        return base_lr + (max_lr - base_lr) * tf.maximum(0.0, (1 - x) * scale_fn(clr_iterations)) 
            
    
    