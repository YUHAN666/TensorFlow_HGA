from tensorflow.python.platform import gfile
import tensorflow as tf
import cv2
import numpy as np
from timeit import default_timer as timer
from utiles.concat_image import concatImage
import os

# IMAGE_SIZE = [224, 576]  # PZT

IMAGE_SIZE = [256, 1408]
# IMAGE_SIZE = [256, 512]  # ST2-COIL
# IMAGE_SIZE = [192, 480]    # ST2-BASE-1
# IMAGE_SIZE = [192, 352]    # ST2-BASE-2
# IMAGE_SIZE = [384, 608]    # ST3-COIL-1
# IMAGE_SIZE = [224, 704]    # ST3-COIL-2
# IMAGE_SIZE = [608, 1056]    # ST3-BASE - 0.8
# IMAGE_SIZE = [160, 1120]    # ST8-COIL-1
# IMAGE_SIZE = [448, 1248]    # ST8-COIL-2
# IMAGE_SIZE = [736, 288]    # ST8-COIL-3
# IMAGE_SIZE = [256, 960]    # ST12-COIL-1 - 0.5
# IMAGE_SIZE = [288, 960]    # ST12-COIL-2 - 0.5
# IMAGE_SIZE = [288, 1056]    # ST11-COIL - 0.5


pb_file_path = './pbModel/pzt_side_model.pb'
image_root = './PZT-SIDE/'
vis_root = './visualization/'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

if not os.path.exists("./p/"):
	os.mkdir("./p/")
if not os.path.exists("./n/"):
	os.mkdir("./n/")
if not os.path.exists(vis_root):
	os.mkdir(vis_root)


with gfile.FastGFile(pb_file_path, 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')

sess.run(tf.global_variables_initializer())
decision_out = sess.graph.get_tensor_by_name('decision_out:0')
mask_out = sess.graph.get_tensor_by_name('mask_out1:0')
input_image = sess.graph.get_tensor_by_name('image_input:0')
image_names = [i[2] for i in os.walk(image_root)][0]

for i in image_names:
	image_path = os.path.join(image_root, i)
	image = cv2.imread(image_path, 0)
	image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
	batch_image = image/255
	batch_image = np.array(batch_image[np.newaxis,:, :, np.newaxis])

	batch_image = np.concatenate([batch_image, batch_image], axis=0)
	start=timer()
	decision = sess.run([decision_out], feed_dict={input_image: batch_image})
	mask = sess.run([mask_out], feed_dict={input_image: batch_image})


	end = timer()
	print(end-start)


	if decision[0][0][0]>0.5:

		img_visual = concatImage([cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(mask[0][:, :, 0]*255, cv2.ROTATE_90_COUNTERCLOCKWISE)])
		visualization_path = os.path.join(vis_root, 'p_'+i)
		img_visual.save(visualization_path)
		os.rename(image_path, os.path.join('./P/', i))
	else:
		img_visual = concatImage([cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(mask[0][:, :, 0]*255, cv2.ROTATE_90_COUNTERCLOCKWISE)])
		visualization_path = os.path.join(vis_root, 'n_'+i)
		img_visual.save(visualization_path)

