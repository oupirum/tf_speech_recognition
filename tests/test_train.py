import train
import dataset
import tensorflow as tf
import math

class TestTrain:
	def setup(self):
		print('')
		train.OPTIONS = train.argparser().parse_args([
				'--dataset_dir', './tests/dataset',
				'--batch_size', '10',
				'--words', 'go,stop,on,off',
				'--training_phases', '10:0.01'])
		self.config = train.Config()
		self.ds = dataset.Dataset(
				'./tests/dataset/',
				['go', 'stop', 'on', 'off'],
				self.config)
		sess = tf.InteractiveSession()
		self.tr = train.Train(self.ds, self.config, sess)

	def test_train(self):
		self.tr._sess.graph.get_operation_by_name('global_step')
		assert(self.tr._sess.graph.get_operation_by_name('final_fc_weights')\
				.outputs[0].shape.as_list() == [62720, len(self.ds.labels)])

	def test_do_step(self):
		self.tr.do_step(1, 0.01)
		acc, cr_entr = self.tr.do_step(2, 0.01)
		assert(not math.isnan(acc))
		assert(not math.isnan(cr_entr))
		assert(tf.train.global_step(self.tr._sess, self.tr._global_step) == 2)

	def test_do_validation(self):
		acc, size = self.tr.do_validation(1)
		assert(not math.isnan(acc))
		assert(size > 0)

	def test_do_final_test(self):
		acc, size = self.tr.do_final_test()
		assert(not math.isnan(acc))
		assert(size > 0)

	def test_training_cycle(self):
		acc_aver = train.run_training(self.tr, 1, self.config)
		assert(acc_aver > 0)
