# -*- coding: utf-8 -*- 
# env: python=3.6
# author: lm 

import tensorflow as tf 
import numpy as np 


def pixel_accuracy(seg, label):
    '''
    Args:
        seg: segmentation resutls with shape = (N, H, W)
        label: the relative label of seg, which with shape = (N, H, W)
    '''
    return tf.reduce_mean(tf.to_float(tf.equal(seg, label)), axis = [0, 1, 2])
    
def pixel_accuracy_np(pred, label):
    return np.mean(np.equal(pred, label))


def mean_iou(seg, label, num_labels):
    '''
    Args:
        seg: segmentation results with shape = (N, H, W)
        label: the relative label of seg, which with shape = (N, H, W)
        num_labels: integer, the total amount of class.
    '''
    unique_labels = tf.range(num_labels, dtype = tf.int32)
    x = tf.equal(tf.expand_dims(seg, 3), unique_labels)
    y = tf.equal(tf.expand_dims(label, 3), unique_labels)
    inter = tf.to_float(tf.logical_and(x, y)) # N, H, W, num_labels
    union = tf.to_float(tf.logical_or(x, y))  # N, H, W, num_labels
    inter = tf.reduce_sum(inter, axis = [1, 2]) # intersection sum 
    union = tf.reduce_sum(union, axis = [1, 2]) # union sum 
    union = tf.maximum(1.0, union) # advoid divide zero. 
    return tf.reduce_mean(inter / union, axis = [0, 1]) # miou 
    

def hist_once(pred, label, num_labels = 19, ignore_label = 255):
    keep = np.logical_not(label == ignore_label)
    merge = pred[keep] * num_labels + label[keep]
    hist = np.bincount(merge, minlength = num_labels ** 2)
    hist = hist.reshape((num_labels, num_labels))
    return hist 

def mean_iou_hist(hist):
    IOUs = np.diag(hist) / np.clip((np.sum(hist, axis = 0) + np.sum(hist, axis = 1) - np.diag(hist)), 1, np.inf) # avoid divide by zero 
    mIOU = np.mean(IOUs)
    return mIOU 
    
'''    
    def __call__(self, net):
        ## evaluate
        hist_size = (self.cfg.n_classes, self.cfg.n_classes)
        hist = np.zeros(hist_size, dtype=np.float32)
        if dist.is_initialized() and dist.get_rank()!=0:
            diter = enumerate(self.dl)
        else:
            diter = enumerate(tqdm(self.dl))
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.cfg.n_classes, H, W))
            probs.requires_grad = False
            for sc in self.cfg.eval_scales:
                new_hw = [int(H*sc), int(W*sc)]
                with torch.no_grad():
                    im = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)
                    im = im.cuda()
                    out = net(im)
                    out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
                    prob = F.softmax(out, 1)
                    probs += prob.cpu()
                    if self.cfg.eval_flip:
                        out = net(torch.flip(im, dims=(3,)))
                        out = torch.flip(out, dims=(3,))
                        out = F.interpolate(out, (H, W), mode='bilinear',
                                align_corners=True)
                        prob = F.softmax(out, 1)
                        probs += prob.cpu()
                    del out, prob
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)

            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once
        if self.distributed:
            hist = torch.tensor(hist).cuda()
            dist.all_reduce(hist, dist.ReduceOp.SUM)
            hist = hist.cpu().numpy().astype(np.float32)
        IOUs = np.diag(hist) / (np.sum(hist, axis=0)+np.sum(hist, axis=1)-np.diag(hist))
        mIOU = np.mean(IOUs)
        return mIOU
'''
# FILE END.