import os


image_root = "F:\\20211208\\PZT-SIDE\\"
# mask_root = "D:/HGAImage/IMAGES/PZT/DN-LABEL/"

image_list = [i[2] for i in os.walk(image_root)][0]

# image_mask_list = [(os.path.join(image_root, i), os.path.join(mask_root,i)) for i in image_list]

for i in range(len(image_list)):
	image_name = image_list[i]
	# new_image_name = image_name.rstrip('jpg')+"bmp"
	new_image_name = str(1467+i) + '.bmp'
	image_src = os.path.join(image_root, image_name)
	image_dst = os.path.join(image_root, new_image_name)
	# mask_src = os.path.join(mask_root, image_name)
	# mask_dst = os.path.join(mask_root, new_image_name)


	os.rename(image_src, image_dst)
	# os.rename(mask_src, mask_dst)


