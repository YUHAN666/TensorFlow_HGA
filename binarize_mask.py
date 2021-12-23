import os
import cv2
import numpy as  np


image_root = "E:/CODES/TensorFlow_HGA/Datasets/ST8-COIL-1/TRAIN/P/MASKS/"

image_list = [i[2] for i in os.walk(image_root)][0]

for path in image_list:
    mask = cv2.imread(os.path.join(image_root, path), 0)

    # mask = np.where(mask>60*np.ones_like(mask), 0, mask)
    mask[0, :] = 0
    mask[1, :] = 0
    cv2.imwrite(os.path.join(image_root, path), mask)