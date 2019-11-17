# -*- coding: utf-8
# env: python=3.6 
# author: lm 

import imgaug as ia # 0.3.0 
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np 


# cifar-10 and cicar-100 32x32
# data aug for cifar-10 and cifar-100
# ia.seed(101)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq_aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.2),
    # sometimes(iaa.GaussianBlur(sigma = (0, 2.0))),
    sometimes(iaa.CropAndPad(
        percent = (-0.1, 0.1),
        pad_mode = ia.ALL,
        pad_cval = (0, 255),
        keep_size = True)),
      
    # sometimes(iaa.CoarseDropout(
    #                     (0.03, 0.15), size_percent=(0.02, 0.05),
    #                     per_channel=0.2
    #                 )),
    # sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.2)),
      
    # sometimes(iaa.ContrastNormalization((0.75, 1.5), per_channel=0.5)), 
    ],
    random_order = True)


# data augmentation for cityscapes 
def seg_seq_augmenter(crop_h = 512, crop_w = 1024):
    seg_seq_aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            sometimes(iaa.GaussianBlur(sigma = (0.0, 2.0))),
            
            # sometimes(iaa.)
            sometimes(iaa.Sharpen((0.0, 1.0))),
            sometimes(iaa.SomeOf(2, [
                iaa.Affine(scale = [0.5, 0.75, 1.0, 1.25, 1.5]),
                iaa.Affine(rotate = (-15, 15)),
                iaa.Affine(translate_px = (-2, 2)),
                iaa.Affine(shear = (-15, 15))
                ])),
            sometimes(iaa.ElasticTransformation(alpha = 10, sigma = 5)),
            sometimes(iaa.contrast.LinearContrast((0.75, 1.5), per_channel = 0.5)),
            sometimes(iaa.Multiply((0.9, 1.1), per_channel = 0.2)),
            iaa.CropToFixedSize(width = crop_w, height = crop_h),
            # sometimes(iaa.Dropout([0.05, 0.2])),
        ], random_order = False)
    return seg_seq_aug 
    
def seg_augmentation(imgs, labs, crop_h = 512, crop_w = 1024):
    aug_imgs, aug_labs = [], []
    for img, lab in zip(imgs, labs):
        img, lab = seg_seq_augmenter(crop_h, crop_w)(image = img, segmentation_maps = SegmentationMapsOnImage(lab, shape = img.shape))
        lab = lab.get_arr().astype(np.uint8)
        aug_imgs.append(img)
        aug_labs.append(lab)
    return np.array(aug_imgs), np.array(aug_labs)
    
if __name__ == "__main__":
    import cv2 as cv 
    img = cv.imread('')
    lab = cv.imread('')
    aimg, alab = seg_augmentation(img, lab)
    
    

# FILE END.
