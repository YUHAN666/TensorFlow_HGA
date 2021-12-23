import tensorflow as tf
from model_components.decision_head import decision_head
from model_components.ghostnet import ghostnet_base
from model_components.seg_neck import bifpn_neck


class Model(object):

    def __init__(self, sess, mode, batch_size, batch_size_inference, input_height, input_width, input_channel, num_class, momentum, logger,
                 data_format="channels_last"):
        self.step = 0
        self.session = sess
        self.bn_momentum = momentum
        self.mode = mode
        self.logger = logger
        self.num_channel = input_channel
        self.height = input_height
        self.width = input_width
        self.num_class = num_class
        self.data_format = data_format
        if self.mode == "train_segmentation":
            self.keep_dropout_backbone = True
            self.keep_dropout_head = True
        elif self.mode == "train_decision":
            self.keep_dropout_backbone = False
            self.keep_dropout_head = True
        else:
            self.keep_dropout_backbone = False
            self.keep_dropout_head = False
        self.batch_size = batch_size
        self.batch_size_inference = batch_size_inference
        with self.session.as_default():
            # Build placeholder to receive data
            if self.mode == 'train_segmentation' or self.mode == 'train_decision':
                self.is_training_seg = tf.placeholder(tf.bool, name='is_training_seg')
                self.is_training_dec = tf.placeholder(tf.bool, name='is_training_dec')

                self.image_input = tf.placeholder(tf.float32, shape=(self.batch_size, self.height, self.width,
                                                                     self.num_channel), name='image_input')

                self.label = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_class), name='label_output')
                self.mask = tf.placeholder(tf.float32, shape=(self.batch_size, self.height, self.width, self.num_class),
                                           name='mask_output')

            elif self.mode == "savePb":
                self.is_training_seg = False
                self.is_training_dec = False

                self.image_input = tf.placeholder(tf.float32,shape=(self.batch_size_inference, self.height, self.width,
                                                                    self.num_channel), name='image_input')
            else:
                self.is_training_seg = False
                self.is_training_dec = False

                self.image_input = tf.placeholder(tf.float32,shape=(self.batch_size, self.height, self.width,
                                                                    self.num_channel), name='image_input')
                self.label = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_class), name='label_output')
                self.mask = tf.placeholder(tf.float32, shape=(self.batch_size, self.height, self.width, self.num_class),
                                           name='mask_output')

            # Building model graph
            self.segmentation_output, self.decision_output, self.mask_out, self.decision_out = self.build_model()

    def build_model(self):
        """
        Build model graph in session
        You can choose different backbone by setting "backbone" in param,
        which includes mixnet(unofficial version), mixnet_official(the official version), efficientnet, fast_scnn, sinet,
        ghostnet, sinet, lednet, cspnet
        :return: segmentation_output: nodes for calculating segmentation loss
                 decision_output: nodes for calculating decision loss
                 mask_out: nodes for visualization output mask of the model
        """
        # create backbone

        # Set depth_multiplier to change the depth of GhostNet
        backbone_output = ghostnet_base(self.image_input, mode=self.mode, data_format=self.data_format,
                                        scope='ghostnet_backbone',
                                        dw_code=None, ratio_code=None,
                                        se=1, min_depth=8, depth_multiplier=0.8, conv_defs=None,
                                        is_training=self.is_training_seg, momentum=self.bn_momentum)

        segmentation_output, decision_in = self.build_segmentation_head(backbone_output, self.is_training_seg)

        decision_output = self.build_decision_head(decision_in)
        decision_out = tf.nn.sigmoid(decision_output, name='decision_out')

        if self.data_format == 'channels_first':
            logits_pixel = tf.transpose(segmentation_output[0], [0, 2, 3, 1])
        else:
            logits_pixel = segmentation_output[0]
        logits_pixel = tf.image.resize_images(logits_pixel, (self.height, self.width), align_corners=True,
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask_out = tf.nn.sigmoid(logits_pixel, name='mask_out')
        if self.mode == "savePb":
            if self.batch_size_inference == 2:
                mask_out = [tf.nn.sigmoid(logits_pixel[0], name='mask_out1'),
                            tf.nn.sigmoid(logits_pixel[1], name='mask_out2')]
            else:
                mask_out = tf.nn.sigmoid(logits_pixel[0], name='mask_out1')

        return segmentation_output, decision_output, mask_out, decision_out

    def build_segmentation_head(self, backbone_output, is_training_seg):
        """
        Build segmentation head
        You can choose different segmentation head by setting "neck" in param, which includes bifpn, fpn, bfp, pan
        :return: segmentation_out: list of output nodes for calculate loss
                 decision_in: list of output nodes for decision head
        """
        with tf.variable_scope('segmentation_head'):

            P3_out, P4_out, P5_out, P6_out, P7_out = bifpn_neck(backbone_output, 64,
                                                                is_training=is_training_seg,
                                                                momentum=self.bn_momentum,
                                                                mode=self.mode, data_format=self.data_format)
            P3 = tf.layers.conv2d(P3_out, self.num_class, (1, 1), (1, 1), use_bias=False, name='P3',
                                  data_format=self.data_format)
            P4 = tf.layers.conv2d(P4_out, self.num_class, (1, 1), (1, 1), use_bias=False, name='P4',
                                  data_format=self.data_format)
            P5 = tf.layers.conv2d(P5_out, self.num_class, (1, 1), (1, 1), use_bias=False, name='P5',
                                  data_format=self.data_format)
            P6 = tf.layers.conv2d(P6_out, self.num_class, (1, 1), (1, 1), use_bias=False, name='P6',
                                  data_format=self.data_format)
            P7 = tf.layers.conv2d(P7_out, self.num_class, (1, 1), (1, 1), use_bias=False, name='P7',
                                  data_format=self.data_format)
            segmentation_out = [P3, P4, P5, P6, P7]
            decision_in = [P3_out, P3]

            return segmentation_out, decision_in

    def build_decision_head(self, decision_in, ):
        """
        Build decision head
        :return: output node of decision head
        """
        dec_out = decision_head(decision_in[0], decision_in[1], class_num=self.num_class, scope='decision_head',
                                keep_dropout_head=self.keep_dropout_head,
                                training=self.is_training_dec, data_format=self.data_format, momentum=self.bn_momentum,
                                mode=self.mode, activation='relu')

        return dec_out
