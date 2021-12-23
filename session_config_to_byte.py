# 用于在C#中设置Session的显存使用比例
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2 #最大使用30%的显存
# config.allow_soft_placement = True

bytes = config.SerializeToString()
hexstring = ", ".join("%02d" % b for b in bytes)
print(hexstring)