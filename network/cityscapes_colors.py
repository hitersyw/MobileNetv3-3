# -*- coding: utf-8 -*- 
# env: python=3.6
# author: lm 

import labelme
import numpy as np 

# rgb for PIL Image save image.
# here define the color for each cityscapes class.
# id2colors = [
#     (128, 64,  128), # 0 road
#     (244, 35,  232), # 1 sidewalk
#     (70,  70,  70 ), # 2 building
#     (102, 102, 156), # 3 wall
#     (190, 153, 153), # 4 fence
#     (153, 153, 153), # 5 pole
#     (250, 170, 30 ), # 6 traffic light
#     (220, 220, 0  ), # 7 traffic sign
#     (107, 142, 35 ), # 8 vegetation
#     (152, 251, 152), # 9 terrain
#     (70,  130, 180), # 10 sky
#     (220, 20,  60 ), # 11 person
#     (255, 0,   0  ), # 12 rider
#     (0,   0,   142), # 13 car
#     (0,   0,   70 ), # 14 truck
#     (0,   60,  100), # 15 bus
#     (0,   80,  100), # 16 train
#     (0,   0,   230), # 17 motorcycle
#     (119, 11,  32 ), # 18 bicycle
#     ]

label_names = [
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle']
    
label_names = label_names + ['ignore'] * (256 - len(label_names))

# bgr for opencv save image.
id2colors = [
    (128, 64,  128), # 0 road
    (232, 35,  244), # 1 sidewalk
    (70,  70,  70 ), # 2 building
    (156, 102, 102), # 3 wall
    (153, 153, 190), # 4 fence
    (153, 153, 153), # 5 pole
    (30 , 170, 250 ), # 6 traffic light
    (0  , 220, 220), # 7 traffic sign
    (35 , 142, 107), # 8 vegetation
    (152, 251, 152), # 9 terrain
    (100, 130, 70 ), # 10 sky
    (60 , 20,  220), # 11 person
    (0  , 0,   255), # 12 rider
    (142, 0,   0  ), # 13 car
    (70,  0,   0  ), # 14 truck
    (100, 60,  0  ), # 15 bus
    (100, 80,  0  ), # 16 train
    (230, 0,   0  ), # 17 motorcycle
    (32 , 11,  119), # 18 bicycle
    ]    
    
id2colors = np.array(id2colors + [(255, 255, 255)] * (256 - len(id2colors))) / 255.0

def draw_label(label, img = None, label_names = label_names, colormap = id2colors):
    return labelme.utils.draw_label(label, img, label_names, colormap)

if __name__ == "__main__":
    print("length of color_names:", len(label_names))
    print("length of id2colors:", len(id2colors))
    assert len(label_names) == len(id2colors)
    for idx in range(len(label_names)):
        print("{} {} {}".format(idx, id2colors[idx], label_names[idx]))
        
        