""" Implementation of BiFPN, FPN, PAN """

from tensorflow import keras
from model_components.conv_block import ConvBatchNormRelu as CBR
from model_components.conv_block import DepthwiseConvBlock


def bifpn_neck(features, num_channels, momentum, mode, data_format, is_training=False, activation='relu', reuse=None):
	"""
	:param features:    [P3_in, P4_in, P5_in, P6_in, P7_in] P7_in is downsample 32
	:param num_channels:  inter channels of bifpn
	:param momentum:    momentum of batch norm layers
	:param mode:        mode for CBR
	:param data_format:
	:param is_training: for batch norm layers
	:param activation:  activation function
	:param reuse:

	:return:        P3_out, P4_out, P5_out, P6_out, P7_out;  P7_out is downsample 32
	"""

	P3_in, P4_in, P5_in, P6_in, P7_in = features

	P3_in = CBR(P3_in, num_channels, 1, 1, is_training, momentum=momentum, mode=mode, name='BiFPN_P3',
				padding='same', data_format=data_format, activation=activation, bn=True)
	P4_in = CBR(P4_in, num_channels, 1, 1, is_training, momentum=momentum, mode=mode, name='BiFPN_P4',
				padding='same', data_format=data_format, activation=activation, bn=True)
	P5_in = CBR(P5_in, num_channels, 1, 1, is_training, momentum=momentum, mode=mode, name='BiFPN_P5',
				padding='same', data_format=data_format, activation=activation, bn=True)
	P6_in = CBR(P6_in, num_channels, 1, 1, is_training, momentum=momentum, mode=mode, name='BiFPN_P6',
				padding='same', data_format=data_format, activation=activation, bn=True)
	P7_in = CBR(P7_in, num_channels, 1, 1, is_training, momentum=momentum, mode=mode, name='BiFPN_P7',
				padding='same', data_format=data_format, activation=activation, bn=True)
	# upsample
	P7_U = keras.layers.UpSampling2D(interpolation='bilinear', data_format=data_format)(P7_in)
	P6_td = keras.layers.Add()([P7_U, P6_in])
	P6_td = DepthwiseConvBlock(P6_td, kernel_size=3, strides=1, data_format=data_format, is_training=is_training,
							   name='BiFPN_U_P6')
	P6_U = keras.layers.UpSampling2D(interpolation='bilinear', data_format=data_format)(P6_td)
	P5_td = keras.layers.Add()([P6_U, P5_in])
	P5_td = DepthwiseConvBlock(P5_td, kernel_size=3, strides=1, data_format=data_format, is_training=is_training,
							   name='BiFPN_U_P5')
	P5_U = keras.layers.UpSampling2D(interpolation='bilinear', data_format=data_format)(P5_td)
	P4_td = keras.layers.Add()([P5_U, P4_in])
	P4_td = DepthwiseConvBlock(P4_td, kernel_size=3, strides=1, data_format=data_format, is_training=is_training,
							   name='BiFPN_U_P4')
	P4_U = keras.layers.UpSampling2D(interpolation='bilinear', data_format=data_format)(P4_td)
	P3_out = keras.layers.Add()([P4_U, P3_in])
	P3_out = DepthwiseConvBlock(P3_out, kernel_size=3, strides=1, data_format=data_format, is_training=is_training,
								name='BiFPN_U_P3')
	# downsample
	P3_D = keras.layers.MaxPooling2D(strides=(2, 2), data_format=data_format)(P3_out)
	P4_out = keras.layers.Add()([P3_D, P4_td, P4_in])
	P4_out = DepthwiseConvBlock(P4_out, kernel_size=3, strides=1, data_format=data_format, is_training=is_training,
								name='BiFPN_D_P4')
	P4_D = keras.layers.MaxPooling2D(strides=(2, 2), data_format=data_format)(P4_out)
	P5_out = keras.layers.Add()([P4_D, P5_td, P5_in])
	P5_out = DepthwiseConvBlock(P5_out, kernel_size=3, strides=1, data_format=data_format, is_training=is_training,
								name='BiFPN_D_P5')
	P5_D = keras.layers.MaxPooling2D(strides=(2, 2), data_format=data_format)(P5_out)
	P6_out = keras.layers.Add()([P5_D, P6_td, P6_in])
	P6_out = DepthwiseConvBlock(P6_out, kernel_size=3, strides=1, data_format=data_format, is_training=is_training,
								name='BiFPN_D_P6')
	P6_D = keras.layers.MaxPooling2D(strides=(2, 2), data_format=data_format)(P6_out)
	P7_out = keras.layers.Add()([P6_D, P7_in])
	P7_out = DepthwiseConvBlock(P7_out, kernel_size=3, strides=1, data_format=data_format, is_training=is_training,
								name='BiFPN_D_P7')

	return P3_out, P4_out, P5_out, P6_out, P7_out
