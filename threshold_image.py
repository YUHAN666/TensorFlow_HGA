import os
import cv2
import numpy as np

root = "E:/CODES/TensorFlow_HGA/Datasets/PZT-DN/TRAIN/N/MASKS/"
image_paths = [i[2] for i in os.walk(root)][0]

for path in image_paths:
	image = cv2.imread(os.path.join(root, path), 0)

	image = np.where(image>60, np.zeros_like(image), image)
	cv2.imwrite(os.path.join(root, path), image)