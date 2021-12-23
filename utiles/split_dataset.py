import os
import numpy as np

image_root = "E:/DATA/MG08/P08/train/p/image"
mask_root = "E:/DATA/MG08/P08/train/p/mask"
dst_root_1 = "E:/DATA/MG08/P08/valid/p/image"
dst_root_2 = "E:/DATA/MG08/P08/valid/p/mask"
scale = 0.2

image_list = [i[2] for i in os.walk(image_root)][0]
val_num = int(len(image_list)*0.2)
np.random.shuffle(image_list)

val_list = image_list[:val_num]

for path in val_list:
	os.rename(os.path.join(image_root, path), os.path.join(dst_root_1, path))
	os.rename(os.path.join(mask_root, path), os.path.join(dst_root_2, path))

