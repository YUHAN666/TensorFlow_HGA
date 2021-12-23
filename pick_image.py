import os


src_root = "F:/HGAImage/IMAGES/ST12/CUT-2-EM"
dst_root = "F:/HGAImage/IMAGES/ST12/123"

ref_root = "E:/CODES/TensorFlow_HGA/Datasets/ST12-COIL-2/VALID/P/IMAGES"

image_list = [i[2] for i in os.walk(src_root)][0]
ref_list = [i[2] for i in os.walk(ref_root)][0]

for path in image_list:

	if path in ref_list:
		os.rename(os.path.join(src_root, path), os.path.join(dst_root, path))