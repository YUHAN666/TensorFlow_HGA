import tensorflow as tf
from tqdm import tqdm
from utiles.iou_eval import iouEval

class Trainer(object):

	def __init__(self, sess, model, mode, learning_rate, epochs, save_frequency, valid_frequency, image_height, image_width,
	             logger, tensorboard_manager, warm_up_step=0, decay_rate=0, decay_step=0, data_format="channels_last", tensorboard=False):
		self.session = sess
		self.model = model
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.save_frequency = save_frequency
		self.valid_frequency = valid_frequency
		self.mode = mode
		self.data_format = data_format
		self.warm_up_step = warm_up_step
		self.image_height = image_height
		self.image_width = image_width
		self.decay_rate = decay_rate
		self.decay_steps = decay_step
		self.logger = logger
		self.tensorboard = tensorboard
		self.tensorboard_manager = tensorboard_manager

		with self.session.as_default():
			self.global_step = tf.Variable(1, trainable=False)
			self.add_global = self.global_step.assign_add(1)
			if self.decay_rate and self.decay_steps:
				self.learning_rate = self.learning_rate_decay()
			if self.warm_up_step:
				self.learning_rate = self.learning_rate_warm_up()
			self.summary_learning_rate = tf.summary.scalar("learning_rate", self.learning_rate)

			if self.mode == "train_segmentation":

				train_segment_var_list = [v for v in tf.trainable_variables() if ('backbone' in v.name) or ('segmentation' in v.name)]

				# 关于tf.GraphKeys.UPDATE_OPS，这是一个tensorflow的计算图中内置的一个集合，
				# 其中会保存一些需要在训练操作之前完成的操作，并配合tf.control_dependencies函数使用。
				update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				update_ops_segment = [v for v in update_ops if ('backbone' in v.name) or ('segmentation' in v.name)]
				optimizer_segmentation = self.optimizer_func()
				segmentation_loss = self.segmentation_loss_func(self.model.segmentation_output, self.model.mask)

				"""
				 如果不在使用时添加tf.control_dependencies函数，即在训练时(training=True)每批次时只会计算当批次的mean和var，
				 并传递给tf.nn.batch_normalization进行归一化，由于mean_update和variance_update在计算图中并不在上述操作的依赖路径上，
				 因为并不会主动完成，也就是说，在训练时mean_update和variance_update并不会被使用到，
				 其值一直是初始值。因此在测试阶段(training=False)使用这两个作为mean和variance并进行归一化操作，
				 这样就会出现错误。而如果使用tf.control_dependencies函数，会在训练阶段每次训练操作执行前被动地去执行mean_update和variance_update，
				 因此moving_mean和moving_variance会被不断更新，在测试时使用该参数也就不会出现错误。
				"""
				with tf.control_dependencies(update_ops_segment):
					optimize_segment = optimizer_segmentation.minimize(segmentation_loss, var_list=train_segment_var_list)
				self.segmentation_loss = segmentation_loss
				self.optimize_segment = optimize_segment
				self.summary_segmentation_loss_train = tf.summary.scalar("segmentation_loss_train", self.segmentation_loss)
				self.summary_segmentation_loss_valid = tf.summary.scalar("segmentation_loss_valid", self.segmentation_loss)

			elif self.mode == "train_decision":
				train_decision_var_list = [v for v in tf.trainable_variables() if 'decision' in v.name]
				update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				update_ops_decision = [v for v in update_ops if 'decision' in v.name]
				optimizer_decision = self.optimizer_func()
				decision_loss = self.decision_loss_func(self.model.decision_output, self.model.label)
				with tf.control_dependencies(update_ops_decision):
					optimize_decision = optimizer_decision.minimize(decision_loss, var_list=train_decision_var_list)
				self.decision_loss = decision_loss
				self.optimize_decision = optimize_decision
				self.summary_decision_loss_train = tf.summary.scalar("decision_loss_train", self.decision_loss)
				self.summary_decision_loss_valid = tf.summary.scalar("decision_loss_valid", self.decision_loss)

	def optimizer_func(self):

		optimizer = tf.train.AdamOptimizer(self.learning_rate)

		return optimizer

	def learning_rate_decay(self):

		# if self.lr_decay == "exponential_decay":
			# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
		decayed_learning_rate = tf.train.exponential_decay(self.learning_rate, global_step=self.global_step,
														   decay_steps=self.decay_steps, decay_rate=self.decay_rate)
		# elif self.lr_decay == "inverse_time_decay":
		# 	# decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)
		# 	decayed_learning_rate = tf.train.inverse_time_decay(self.learning_rate, global_step=self.global_step,
		# 														decay_steps=self.decay_steps,
		# 														decay_rate=self.decay_rate)
		# elif self.lr_decay == "natural_exp_decay":
		# 	# decayed_learning_rate = learning_rate * exp(-decay_rate * global_step / decay_steps)
		# 	decayed_learning_rate = tf.train.natural_exp_decay(self.learning_rate, global_step=self.global_step,
		# 													   decay_steps=self.decay_steps, decay_rate=self.decay_rate)
		# elif self.lr_decay == "cosine_decay":
		# 	# cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
		# 	# decayed = (1 - alpha) * cosine_decay + alpha
		# 	# decayed_learning_rate = learning_rate * decayed
		# 	# alpha的作用可以看作是baseline，保证lr不会低于某个值。不同alpha的影响如下：
		# 	decayed_learning_rate = tf.train.cosine_decay(self.learning_rate, global_step=self.global_step,
		# 												  decay_steps=self.decay_steps, alpha=0.3)
		# else:
		# 	raise ValueError("Unknown decay strategy")
		return decayed_learning_rate

	def learning_rate_warm_up(self):

		warmup_learn_rate = self.learning_rate * tf.cast(self.global_step / self.warm_up_step, tf.float32)
		learning_rate = tf.cond(self.global_step <= self.warm_up_step, lambda: warmup_learn_rate, lambda: self.learning_rate)
		return learning_rate

	def segmentation_loss_func(self, segmentation_output, mask):
		""" Segmentation loss"""
		if self.data_format == 'channels_first':
			for nec_index in range(len(segmentation_output)):
				segmentation_output[nec_index] = tf.transpose(segmentation_output[nec_index], [0, 2, 3, 1])
		logits_pixel = tf.image.resize_images(segmentation_output[0], (self.image_height, self.image_width), align_corners=True,
											  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

		loss = tf.nn.sigmoid_cross_entropy_with_logits

		segmentation_loss = tf.reduce_mean(loss(logits=logits_pixel, labels=mask)) + \
							tf.reduce_mean(loss(logits=segmentation_output[1], labels=tf.image.resize_images(mask, (
							self.image_height // 4, self.image_width // 4)))) + \
							tf.reduce_mean(loss(logits=segmentation_output[2], labels=tf.image.resize_images(mask, (
							self.image_height // 8, self.image_width // 8)))) + \
							tf.reduce_mean(loss(logits=segmentation_output[3], labels=tf.image.resize_images(mask, (
							self.image_height // 16, self.image_width // 16)))) + \
							tf.reduce_mean(loss(logits=segmentation_output[4], labels=tf.image.resize_images(mask, (
							self.image_height // 32, self.image_width // 32))))

		return segmentation_loss

	def decision_loss_func(self, dec_out, label):
		""" Decision loss"""

		decision_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dec_out, labels=label)
		decision_loss = tf.reduce_mean(decision_loss)

		return decision_loss

	def train_segmentation(self, data_manager_train, data_manager_valid, saver):
		""" Train the segmentation part of the model """
		self.logger.info("Start training segmentation for {} epochs, {} steps per epochs, batch size is {}. Save to checkpoint every {} epochs "
						 .format(self.epochs, data_manager_train.num_batch, data_manager_train.batch_size, self.save_frequency))
		if self.decay_steps and self.decay_rate:
			lr = self.session.run([self.learning_rate])
			self.logger.info("Using decay strategy, initial learning rate:{}, decay_rate: {}， decay_steps: {}".format(lr,
				                                                                                         self.decay_rate,
				                                                                                         self.decay_steps))
		else:
			lr = self.learning_rate

		current_epoch = saver.step + 1
		with self.session.as_default():
			print('Start training segmentation for {} epochs, {} steps per epoch'.format(self.epochs, data_manager_train.num_batch))
			tensorboard_merged = tf.summary.merge([self.summary_learning_rate, self.summary_segmentation_loss_train])
			trainIoU = iouEval(data_manager_train.batch_size)
			train_loss = []
			train_iou = []
			val_loss = []
			val_iou = []
			train_iou_placeholder = tf.placeholder(tf.float32)  # placeholder for TensorBoard
			val_iou_placeholder = tf.placeholder(tf.float32)
			summary_train_iou = tf.summary.scalar("train set iou", train_iou_placeholder)
			summary_val_iou = tf.summary.scalar("val set iou", val_iou_placeholder)
			for i in range(current_epoch, self.epochs+current_epoch):
				trainIoU.reset()
				print('Epoch {}:'.format(i))
				pbar = tqdm(total=data_manager_train.num_batch, leave=True)
				iter_loss = []
				# epoch start
				for batch in range(data_manager_train.num_batch):
					img_batch, mask_batch, label_batch, _ = self.session.run(data_manager_train.next_batch)

					if self.tensorboard:
						mask_in, mask_out, _, loss_value_batch, tensorboard_result = self.session.run([self.model.mask,
									                                                                   self.model.mask_out,
									                                                                   self.optimize_segment,
									                                                                   self.segmentation_loss,
									                                                                   tensorboard_merged],
									                                                                  feed_dict={
									                                                                      self.model.image_input: img_batch,
									                                                                      self.model.mask: mask_batch,
									                                                                      self.model.label: label_batch,
									                                                                      self.model.is_training_seg: True,
									                                                                      self.model.is_training_dec: False})
						self.tensorboard_manager.add_summary(tensorboard_result,(i - 1) * data_manager_train.num_batch + batch)
					else:
						mask_in, mask_out, _, loss_value_batch = self.session.run([self.model.mask,
				                                                                   self.model.mask_out,
				                                                                   self.optimize_segment,
				                                                                   self.segmentation_loss],
				                                                                  feed_dict={
				                                                                      self.model.image_input: img_batch,
				                                                                      self.model.mask: mask_batch,
				                                                                      self.model.label: label_batch,
				                                                                      self.model.is_training_seg: True,
				                                                                      self.model.is_training_dec: False})



					trainIoU.addBatch(mask_in, mask_out)
					iter_loss.append(loss_value_batch)
					pbar.update(1)
					if self.decay_steps and self.decay_rate:
						_, lr = self.session.run([self.add_global, self.learning_rate])
				pbar.clear()
				pbar.close()
				# loss and iou check
				train_loss.append(sum(iter_loss)/len(iter_loss))
				train_iou.append(trainIoU.getIoU())
				if self.tensorboard:
					tensorboard_train_iou = self.session.run(summary_train_iou, feed_dict={train_iou_placeholder: trainIoU.getIoU()})
					self.tensorboard_manager.add_summary(tensorboard_train_iou, i)
				self.logger.info("Epoch{}  train_loss:{}, train_iou:{}, learning_rate:{}"
				                 .format(i, sum(iter_loss) / len(iter_loss), train_iou[i - current_epoch], lr))
				print('train_loss:{}, train_iou:{}, learning_rate:{}'
				      .format(sum(iter_loss) / len(iter_loss), train_iou[i - current_epoch], lr))

				if (i-current_epoch+1)%self.valid_frequency == 0:
					val_loss_epo, val_iou_epo = self.valid_segmentation(data_manager_valid, i)
					val_loss.append(val_loss_epo)
					val_iou.append(val_iou_epo)

					self.logger.info("Epoch{}, val_loss:{}, val_iou:{}".format(i, val_loss[i-current_epoch], val_iou[i-current_epoch]))
					print('val_loss:{}, val_iou:{}'.format(val_loss[i-current_epoch], val_iou[i-current_epoch]))
					if self.tensorboard:
						tensorboard_val_iou = self.session.run(summary_val_iou, feed_dict={val_iou_placeholder: val_iou_epo})
						self.tensorboard_manager.add_summary(tensorboard_val_iou, i)

				if (i-current_epoch+1) % self.save_frequency == 0 or i == self.epochs + current_epoch:
					saver.save_checkpoint(i)

		self.logger.info("Complete training segmentation, reduce train_loss from {} to {}, increase train_iou from {} to {} "
						 "reduce val_loss from {} to {}, increase val_iou from {} to {}"
						 .format(train_loss[0], train_loss[-1], train_iou[0], train_iou[-1], val_loss[0], val_loss[-1], val_iou[0], val_iou[-1]))

	def valid_segmentation(self, data_manager_valid, epoch):
		""" Evaluate the segmentation part during training process"""
		with self.session.as_default():
			print('start validating segmentation')
			total_loss = 0.0
			num_step = 0.0
			valIoU = iouEval(data_manager_valid.batch_size)

			for batch in range(data_manager_valid.num_batch):
				img_batch, mask_batch, label_batch, _ = self.session.run(data_manager_valid.next_batch)

				if self.tensorboard:
					mask_in, mask_out, total_loss_value_batch, tensorboard_result = self.session.run([self.model.mask,
																									  self.model.mask_out,
																									  self.segmentation_loss,
																									  self.summary_segmentation_loss_valid],
																 feed_dict={self.model.image_input: img_batch,
																			self.model.mask: mask_batch,
																			self.model.label: label_batch,
																			self.model.is_training_seg: False,
																			self.model.is_training_dec: False})
					self.tensorboard_manager.add_summary(tensorboard_result, epoch)
				else:
					mask_in, mask_out, total_loss_value_batch= self.session.run([self.model.mask,
																			     self.model.mask_out,
																		         self.segmentation_loss],
																 feed_dict={self.model.image_input: img_batch,
																			self.model.mask: mask_batch,
																			self.model.label: label_batch,
																			self.model.is_training_seg: False,
																			self.model.is_training_dec: False})

				valIoU.addBatch(mask_in, mask_out)
				num_step = num_step + 1
				total_loss += total_loss_value_batch

			total_loss = total_loss/num_step
			val_iou = valIoU.getIoU()
			return total_loss, val_iou

	def train_decision(self, data_manager_train, data_manager_valid, saver):
		"""Train the decision part of model"""
		with self.session.as_default():
			print('Start training decision for {} epochs, {} steps per epoch'.format(self.epochs, data_manager_train.num_batch))
			self.logger.info("Start training decision for {} epochs, {} steps per epochs, batch size is {}. Save to checkpoint every {} epochs "
				.format(self.epochs, data_manager_train.num_batch, data_manager_train.batch_size, self.save_frequency))
			if self.decay_steps and self.decay_rate:
				lr = self.session.run([self.learning_rate])
				self.logger.info(
					"Using decay strategy, initial learning rate:{}, decay_rate: {}， decay_steps: {}".format(lr,
					                                                                                         self.decay_rate,
					                                                                                         self.decay_steps))
			else:
				lr = self.learning_rate
			if self.decay_steps and self.decay_rate:
				self.logger.info("Using decay strategy, initial learning rate:{}, decay_rate: {}， decay_steps: {}"
								 .format(lr, self.decay_rate, self.decay_steps))
			current_epoch = saver.step + 1
			tensorboard_merged = tf.summary.merge([self.summary_learning_rate, self.summary_decision_loss_train])
			train_loss = []
			train_acc = []
			val_loss = []
			val_acc = []
			iter_loss = []
			for i in range(current_epoch, self.epochs+current_epoch):
				print('Epoch {}:'.format(i))
				pbar = tqdm(total=data_manager_train.num_batch, leave=True)
				# epoch start
				true_account = 0
				false_account = 0
				for batch in range(data_manager_train.num_batch):
					# batch start
					img_batch, mask_batch, label_batch, path = self.session.run(data_manager_train.next_batch)
					if self.tensorboard:
						decision_out, _, loss_value_batch, tensorboard_result = self.session.run([self.model.decision_out,
																								  self.optimize_decision,
																								  self.decision_loss,
																								  tensorboard_merged],
																				feed_dict={self.model.image_input: img_batch,
																						   self.model.mask: mask_batch,
																						   self.model.label: label_batch,
																						   self.model.is_training_seg: False,
																						   self.model.is_training_dec: True})
						self.tensorboard_manager.add_summary(tensorboard_result, (i - 1) * data_manager_train.num_batch + batch)
					else:
						decision_out, _, loss_value_batch = self.session.run([self.model.decision_out,
																			  self.optimize_decision,
																			  self.decision_loss],
																		feed_dict={self.model.image_input: img_batch,
																		           self.model.mask: mask_batch,
																		           self.model.label: label_batch,
																		           self.model.is_training_seg: False,
																		           self.model.is_training_dec: True})

					iter_loss.append(loss_value_batch)
					pbar.update(1)
					for b in range(data_manager_train.batch_size):
						if (decision_out[b] > 0.5 and label_batch[b] == 1) or (decision_out[b] < 0.5 and label_batch[b] == 0):
							true_account += 1
						else:
							false_account += 1

					if self.decay_steps and self.decay_rate:
						_, lr = self.session.run([self.add_global, self.learning_rate])
				train_acc.append(true_account/(true_account + false_account))
				pbar.clear()
				pbar.close()
				train_loss.append(sum(iter_loss)/len(iter_loss))
				val_loss_epo, val_acc_epo = self.valid_decision(data_manager_valid, i)
				val_loss.append(val_loss_epo)
				val_acc.append(val_acc_epo)
				print('train_loss:{}, train_accuracy:{}  val_loss:{}, val_accuracy:{}'.
					  format(train_loss[i-current_epoch], train_acc[i-current_epoch], val_loss[i-current_epoch], val_acc[i-current_epoch]))
				self.logger.info("Epoch {}  train_loss: {}, train_accuracy: {}, val_loss: {}, val_accuracy: {}"
								 .format(i+saver.step, train_loss[i-current_epoch], train_acc[i-current_epoch], val_loss[i-current_epoch], val_acc[i-current_epoch]))

				if i % self.save_frequency == 0 or i == self.epochs:
					# if val_loss < best_loss:
					# best_loss = val_loss
					# print('reduce loss to {}, saving model at epoch:{}'.format(val_loss, i))
					saver.save_checkpoint(i)
		self.logger.info("Complete training decision, reduce train_loss from {} to {}, increase train_accuracy from {} to {},  "
						 "reduce val_loss from {} to {}, increase val_accuracy from {} to {}"
						 .format(train_loss[0], train_loss[-1], train_acc[0], train_acc[-1], val_loss[0], val_loss[-1], val_acc[0], val_acc[-1]))

	def valid_decision(self, data_manager_valid, epoch):
		""" Evaluate the performance of decision part during training"""
		with self.session.as_default():
			# print('start validating decision')
			total_loss = 0.0
			num_step = 0.0
			true_account = 0
			false_account = 0
			for batch in range(data_manager_valid.num_batch):
				img_batch, mask_batch, label_batch, _ = self.session.run(data_manager_valid.next_batch)
				if self.tensorboard:
					decision_out, total_loss_value_batch, tensorboard_result = self.session.run([self.model.decision_out,
																								 self.decision_loss,
																								 self.summary_decision_loss_valid],
																 feed_dict={self.model.image_input: img_batch,
																			self.model.mask: mask_batch,
																			self.model.label: label_batch,
																			self.model.is_training_seg: False,
																			self.model.is_training_dec: False})
					self.tensorboard_manager.add_summary(tensorboard_result, epoch)
				else:
					decision_out, total_loss_value_batch = self.session.run([self.model.decision_out,
	                                                                         self.decision_loss],
																feed_dict={self.model.image_input: img_batch,
																           self.model.mask: mask_batch,
																           self.model.label: label_batch,
																           self.model.is_training_seg: False,
																           self.model.is_training_dec: False})

				for b in range(data_manager_valid.batch_size):
					if (decision_out[b] > 0.5 and label_batch[b] == 1) or (decision_out[b] < 0.5 and label_batch[b] == 0):
						true_account += 1
					else:
						false_account += 1
				num_step = num_step + 1
				total_loss += total_loss_value_batch
			accuracy = true_account/(true_account+false_account)
			total_loss /= num_step
			return total_loss, accuracy
