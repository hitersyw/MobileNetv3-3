# -*- coding: utf-8 -*- 
# python=3.6
# author: lm 

import tensorflow as tf 
import numpy as np 
import cv2 as cv 
from tf_records import read_tfrecords_by_data_v2, get_number_examples
from skimage import color 

id2colors = [
    (128, 64,  128), # 0 road
    (244, 35,  232), # 1 sidewalk
    (70,  70,  70 ), # 2 building
    (102, 102, 156), # 3 wall
    (190, 153, 153), # 4 fence
    (153, 153, 153), # 5 pole
    (250, 170, 30 ), # 6 traffic light
    (220, 220, 0  ), # 7 traffic sign
    (107, 142, 35 ), # 8 vegetation
    (152, 251, 152), # 9 terrain
    (70,  130, 180), # 10 sky
    (220, 20,  60 ), # 11 person
    (255, 0,   0  ), # 12 rider
    (0,   0,   142), # 13 car
    (0,   0,   70 ), # 14 truck
    (0,   60,  100), # 15 bus
    (0,   80,  100), # 16 train
    (0,   0,   230), # 17 motorcycle
    (119, 11,  32 ), # 18 bicycle
    ]

def test_tf_records_py(tfr_path):
    batch_size = 2
    image_label_pairs = read_tfrecords_by_data_v2(tfr_path, (1024, 2048), 3, batch_size = batch_size, label_type = "segmentation")
    print("There are {} examples in dataset.".format(get_number_examples(tfr_path)))
    with tf.Session() as sess:
        for i in range(10):
            images, labels = sess.run(image_label_pairs) 
            for j in range(batch_size):
                cv.imwrite('image_{}_{}.jpg'.format(i, j), images[j])
                cv.imwrite('label_{}_{}.png'.format(i, j), color.label2rgb(labels[j], images[j], colors = id2colors))
                cv.imwrite('label_{}_{}.png'.format(i, j), color.label2rgb(labels[j], colors = id2colors))
                # cv.imwrite('label_{}_{}.png'.format(i, j), color.label2rgb(labels[j], images[j]))

if __name__ == "__main__":
    test_tf_records_py('./cityscapes/1024x2048/train/train_gtFine.tfr')    

# FILE END.
