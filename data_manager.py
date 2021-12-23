import os
import cv2
import numpy as np
import tensorflow as tf


class DataManager(object):
	def __init__(self, data_root, batch_size, image_height, image_width, shuffle=True, balance=True, augmentor=None):

		self.data_root = data_root
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.balance = balance
		self.image_height = image_height
		self.image_width = image_width
		self.p_image_mask_list, self.n_image_mask_list = self.get_image_list()
		self.num_data = len(self.p_image_mask_list)+len(self.n_image_mask_list)
		if self.balance:
			self.num_batch = max(len(self.p_image_mask_list),len(self.n_image_mask_list))*2//self.batch_size +  1
		else:
			self.num_batch = self.num_data//self.batch_size + 1
		self.augmentor = augmentor
		self.next_batch = self.get_next()

	def get_next(self):
		""" Encapsulate generator into TensorFlow DataSet"""
		dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.float32, tf.float32, tf.string))
		dataset = dataset.repeat()
		dataset = dataset.batch(self.batch_size)
		iterator = dataset.make_one_shot_iterator()
		out_batch = iterator.get_next()

		return out_batch

	def generator(self):
		"""
		Generator of image, mask, label, and image_root
		Should be revised according to the saving path of data
		:return:  image
				  mask
				  label
				  image_path
		"""

		p_index = 0
		n_index = 0
		index = 0
		if self.shuffle:
			np.random.shuffle(self.p_image_mask_list)
			np.random.shuffle(self.n_image_mask_list)
		if not self.balance:
			image_mask_list = self.n_image_mask_list+self.p_image_mask_list
		while True:
			if self.balance:
				if p_index == len(self.p_image_mask_list):
					p_index = 0
					if self.shuffle:
						np.random.shuffle(self.p_image_mask_list)
				if n_index == len(self.n_image_mask_list):
					n_index = 0
					if self.shuffle:
						np.random.shuffle(self.n_image_mask_list)
				prob = np.random.uniform()
				if prob > 0.5:
					path = self.p_image_mask_list[p_index]
					p_index += 1
					label = np.array([1.0])
				else:
					path = self.n_image_mask_list[n_index]
					n_index += 1
					label = np.array([0.0])
			else:
				if index < len(self.n_image_mask_list):
					path = image_mask_list[index]
					label = np.array([0.0])
				elif index<len(image_mask_list):
					path = image_mask_list[index]
					label = np.array([1.0])
				else:
					index = 0
					path = image_mask_list[index]
				index += 1



			image = cv2.imread(path[0], 0)
			mask = cv2.imread(path[1], 0)
			image = cv2.resize(image, (self.image_width, self.image_height))
			mask = cv2.resize(mask, (self.image_width, self.image_height))
			_, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
			mask = mask // 255

			if self.augmentor:
				image, mask = self.augmentor.transform_seg(image, mask)
			image = image / 255.0
			mask = mask.astype(np.float)

			# cv2.imshow("image", image)
			# cv2.imshow("mask", mask)
			# cv2.waitKey()
			# cv2.destroyAllWindows()

			if len(image.shape) == 2:
				image = (np.array(image[:, :, np.newaxis]))
			if len(mask.shape) == 2:
				mask = (np.array(mask[:, :, np.newaxis]))

			yield image, mask, label, path[0]


	@staticmethod
	def check_mask(image_names, mask_names, image_root, mask_root):
		# 检查是否每一张图片都有对应的mask
		image_mask_paths = []
		for i in image_names:
			# if i[:-4] + "_label.bmp" not in mask_names:          # -4 is length of .bmp
			if i[:-4] + ".bmp" not in mask_names:  # -4 is length of .bmp
				raise FileNotFoundError(i[:-4] + "_label.bmp not found")
			else:
				image_mask_paths.append((os.path.join(image_root, i),
				                         # os.path.join(mask_root, i[:-4] + "_label.bmp")
				                         os.path.join(mask_root, i[:-4] + ".bmp")
				                         ))
		return image_mask_paths

	def get_image_list(self):

		p_root = os.path.join(self.data_root, "P")
		n_root = os.path.join(self.data_root, "N")

		p_image_root = os.path.join(p_root, "IMAGES")
		p_mask_root = os.path.join(p_root, "MASKS")

		n_image_root = os.path.join(n_root, "IMAGES")
		n_mask_root = os.path.join(n_root, "MASKS")

		p_image_files = [i[2] for i in os.walk(p_image_root)][0]
		p_mask_files = [i[2] for i in os.walk(p_mask_root)][0]
		n_image_files = [i[2] for i in os.walk(n_image_root)][0]
		n_mask_files = [i[2] for i in os.walk(n_mask_root)][0]

		p_image_mask = self.check_mask(p_image_files, p_mask_files, p_image_root, p_mask_root)
		n_image_mask = self.check_mask(n_image_files, n_mask_files, n_image_root, n_mask_root)
		return p_image_mask, n_image_mask

