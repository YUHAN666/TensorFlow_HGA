import tensorflow as tf
from data_manager import DataManager
from model import Model
from image_augmentor import ImageAugmentor
from pb_tester import PbTester
from saver import Saver
from tensorboard_manager import TensorboardManager
from trainer import Trainer
from validator import Validator
from config import BasicParam as basic_param
from config import DataParam as data_param
from config import TrainParam as train_param
from config import SaverParam as saver_param
from config import PbTestParam as pb_param
from config import AugmentParam as aug_param

class Agent(object):

    def __init__(self, logger):

        self.logger = logger
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        self.mode = basic_param["mode"]
        self.num_class = basic_param["num_class"]
        self.momentum = basic_param["momentum"]
        self.new = basic_param["new"]

        self.image_height = data_param["image_height"]
        self.image_width = data_param["image_width"]
        self.input_channel = data_param["image_channel"]
        self.data_root_train = data_param["data_root_train"]
        self.data_root_valid = data_param["data_root_valid"]

        self.epochs = train_param["epochs"]
        self.batch_size = train_param["batch_size"]
        self.learning_rate = train_param["learning_rate"]
        self.tensorboard_dir = train_param["tensorboard_dir"]
        self.save_frequency = train_param["save_frequency"]
        self.valid_frequency = train_param["valid_frequency"]
        self.warm_up_step = train_param["warm_up_step"]
        self.decay_step = train_param["decay_step"]
        self.decay_rate = train_param["decay_rate"]
        self.tensorboard = train_param["tensorboard"]

        self.batch_size_inference = saver_param["batch_size_inference"]
        self.max_to_keep = saver_param["max_to_keep"]
        self.checkpoint_dir = saver_param["checkpoint_dir"]
        self.input_nodes_list = saver_param["input_nodes_list"]
        self.output_nodes_list = saver_param["output_nodes_list"]
        self.pb_save_path = saver_param["pb_save_path"]
        self.pb_name = saver_param["pb_name"]
        self.saving_mode = saver_param["saving_mode"]

        self.model = Model(self.session, self.mode, self.batch_size, self.batch_size_inference, self.image_height, self.image_width,
                           self.input_channel, self.num_class, self.momentum, self.logger)    # ???????????????graph?????????session???
        self.augmentor = ImageAugmentor(aug_param)

        if self.mode in ["train_decision", "visualization", "testing"]:
            self.data_manager_train = DataManager(self.data_root_train, self.batch_size, self.image_height, self.image_width)  # ????????????????????????
        else:
            self.data_manager_train = DataManager(self.data_root_train, self.batch_size, self.image_height,
                                                  self.image_width, augmentor=self.augmentor)  # ????????????????????????
        self.data_manager_valid = DataManager(self.data_root_valid, self.batch_size, self.image_height, self.image_width, shuffle=False, balance=False)    # ????????????????????????
        self.tensorboard_manager = TensorboardManager(self.session, self.tensorboard_dir)  # ??????TensorBoard???????????????
        self.trainer = Trainer(self.session, self.model, self.mode, self.learning_rate, self.epochs, self.save_frequency,
                               self.valid_frequency, self.image_height,self.image_width,self.logger,
                               self.tensorboard_manager,self.warm_up_step, self.decay_rate, self.decay_step,
                               data_format='channels_last', tensorboard=self.tensorboard)    # ?????????????????????????????????????????????

        self.saver = Saver(self.session, self.checkpoint_dir, self.max_to_keep, self.input_nodes_list,
                           self.output_nodes_list, self.pb_save_path, self.pb_name, self.saving_mode, self.logger, self.model)     # ?????????session?????????checkpoint???pb?????????
        self.validator = Validator(self.session, self.model, self.logger)        # ????????????????????????????????????

        logger.info("Successfully initialized")

    def run(self):

        if not self.new and self.mode != "testPb":
            self.saver.load_checkpoint()

        if self.mode == "train_segmentation":      # ????????????????????????
            self.trainer.train_segmentation(self.data_manager_train, self.data_manager_valid, self.saver)
        elif self.mode == "train_decision":        # ????????????????????????
            self.trainer.train_decision(self.data_manager_train, self.data_manager_valid, self.saver)
        elif self.mode == "visualization":         # ????????????????????????
            self.validator.valid_segmentation(self.data_manager_train)
        elif self.mode == "testing":               # ????????????????????????
            self.validator.valid_decision(self.data_manager_train)
            self.validator.valid_decision(self.data_manager_valid)
        elif self.mode == "savePb":                # ???????????????pb??????
            self.saver.save_pb()
        elif self.mode == "testPb":                # ??????pb????????????
            self.pb_tester = PbTester(pb_param, self.data_root_train, self.image_height, self.image_width, self.batch_size_inference, self.logger)
            self.pb_tester.test_segmentation()
            self.pb_tester.test_decision()
            # self.pb_tester.view_timeline()

        self.session.close()
        self.tensorboard_manager.close()






