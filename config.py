
MODE = {0: "train_segmentation",    # 训练模型分割部分
        1: "train_decision",        # 训练模型分类部分
        2: "visualization",         # 验证模型分割效果
        3: "testing",               # 验证模型分类效果
        4: "savePb",                # 将模型保存为pb文件
        5: "testPb",                # 测试pb模型效果
        }
BasicParam = {
	"mode": MODE[1],
	"new": False,
	"type": "PZT-DN",
	"num_class": 1,
	"momentum": 0.9
}
DataParam = {
	"image_height": 224,
	"image_width": 576, # PZT - 0.8
	# "image_width":1408,
	# "image_height": 256,  #PZT-SIDE
	# "image_height": 256,
	# "image_width": 512, # ST2-COIL - 0.8
	# "image_height": 192,
	# "image_width": 480,  # ST2-BASE-1 - 0.8
	# "image_height": 192,
	# "image_width": 352,  # ST2-BASE-2 - 0.8
	# "image_height": 384,
	# "image_width": 608, # ST3-COIL-1 - 0.8
	# "image_height": 224,
	# "image_width": 704, # ST3-COIL-2 - 0.8
	# "image_height": 608,
	# "image_width": 1056, # ST3-BASE - 0.8
	# "image_height": 160,
	# "image_width": 1120,  # ST8-COIL-1 - 0.8
	# "image_height": 448,
	# "image_width": 1248,  # ST8-COIL-2 - 0.8
	# "image_height": 736,
	# "image_width": 288,  # ST8-COIL-3 - 0.8
	# "image_height": 256,
	# "image_width": 960,  # ST12-COIL-1 - 0.5
	# "image_height": 288,
	# "image_width": 960, # ST12-COIL-2 - 0.5
	# "image_height": 288,
	# "image_width": 1056, # ST11-COIL - 0.5
	"image_channel": 1,
	"data_root_train": "./Datasets/{}/TRAIN/".format(BasicParam["type"]),
	"data_root_valid": "./Datasets/{}/VALID/".format(BasicParam["type"])
}
TrainParam = {
	"learning_rate": 3e-3,
	"batch_size": 16,
	"epochs": 200,
	"save_frequency": 1,
	"valid_frequency": 1,
	"warm_up_step": 200,
	"decay_step": 10000,
	"decay_rate": 0.1,
	"tensorboard": False,
	"tensorboard_dir": "./tensorboard/"
}
SaverParam = {
	"batch_size_inference": 1,
	"max_to_keep": 3,
	"checkpoint_dir": "./checkpoints/{}/".format(BasicParam["type"]),
	"input_nodes_list": ["image_input"],
	# "output_nodes_list": ["decision_out", "mask_out1", "mask_out2"],
	"output_nodes_list": ["decision_out", "mask_out1"],
	"pb_save_path": "./pbModel",
	"pb_name": "pzt_dn_model.pb",
	"saving_mode": "CBR"
}

AugmentParam = {
	# "crop": ((0,0.05), (0,0.05), (0,0.05), (0,0.05)),
	"crop": ((0,0.05), (0, 0.05), (0, 0.05), (0,0.05)),
	"rotate": (-5 , 5),
	# "rotate": False,
	"blur": False,
	# "motion": {"k": (3, 20),
	#            "angle": [-10, 10],
	#            "direction": 0},
	"motion":False,
	"gamma": (0.7, 1.3),
	"flip": True
}

PbTestParam = {
	"pb_path": "./pbModel/pzt_dn_model.pb",
	"pb_input_tensor": ["image_input:0"],  # 需测试的pb模型输入Tensor名
	"pb_output_mask_name": ["mask_out1:0"],  # 需测试的pb模型输出Mask Tensor名
	"pb_output_label_name": ["decision_out:0"],  # 需测试的pb模型输出Label Tensor名
	"timeline_dir": "./timeline/",  # timeline Jason文件输出路径

}

