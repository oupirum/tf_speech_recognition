import argparse
import tensorflow as tf
import numpy as np
import os
import dataset

OPTIONS = None

def main():
	config = Config()
	ds = dataset.Dataset(
			OPTIONS.dataset_dir,
			OPTIONS.words.split(','),
			config)
	sess = tf.InteractiveSession()
	train = Train(ds, config, sess)

	start_step = 1
	if OPTIONS.checkpoint:
		start_step = train.restore(OPTIONS.checkpoint)
	print('Start step: %d ' % start_step)

	tf.train.write_graph(sess.graph_def, OPTIONS.train_dir, 'graph.pbtxt')
	with open(
			os.path.join(OPTIONS.train_dir, 'labels.txt'), 'w') as f:
		f.write('\n'.join(ds.labels))

	run_training(train, start_step, config)

	accuracy, _ = train.do_final_test()
	print('Final test accuracy: %.1f%%'
			% (accuracy * 100,))

def run_training(train, start_step, config):
	prev_train_accuracies = []
	for step in range(start_step, config.num_steps + 1):
		learning_rate = get_curr_learning_rate(step, config.phases)
		accuracy, cross_entropy = train.do_step(step, learning_rate)

		prev_train_accuracies.append(accuracy)
		if len(prev_train_accuracies) > 30:
			prev_train_accuracies.pop(0)
		acc_average = np.sum(prev_train_accuracies) / len(prev_train_accuracies)

		print('#%d/%d: rate %f, acc %.1f%%, aver acc %.1f%%, cross entropy %f'
				% (step, config.num_steps, learning_rate,
				accuracy * 100, acc_average * 100, cross_entropy))

		is_last_step = step == config.num_steps
		if (step % OPTIONS.validation_interval == 0
				or is_last_step):
			accuracy, _ = train.do_validation(step)
			print('Validation accuracy: %.1f%%'
					% (accuracy * 100,))

		if (step % OPTIONS.save_interval == 0
				or is_last_step):
			train.save(step)
	return acc_average

def get_curr_learning_rate(step, phases):
	steps_sum = 0
	for phase in phases:
		steps_sum += phase['steps']
		if step <= steps_sum:
			learning_rate = phase['rate']
			break
	return learning_rate


class Train:
	def __init__(self, ds, config, sess):
		self._ds = ds
		self._config = config
		self._sess = sess

		self._fingerprint_pl = tf.placeholder(
				tf.float32, [None, self._config.fingerprint_size],
				name='fingerprint_pl')
		self._dropout_pl = tf.placeholder(
				tf.float32,
				name='dropout_pl')
		self._learning_rate_pl = tf.placeholder(
				tf.float32, [],
				name='learning_rate_pl')
		self._ground_truth_pl = tf.placeholder(
				tf.int64, [None],
				name='ground_truth_pl')

		logits = self._create_model()
		self._init_backprop(logits)

		self._global_step = tf.train.get_or_create_global_step()
		self._increment_global_step = tf.assign(
				self._global_step, self._global_step + 1)

		self._merged_summaries = tf.summary.merge_all()
		self._train_writer = tf.summary.FileWriter(
				OPTIONS.summaries_dir + '/train',
				self._sess.graph)
		self._validation_writer = tf.summary.FileWriter(
				OPTIONS.summaries_dir + '/validation')

		tf.global_variables_initializer().run()

	def _create_model(self):
		# taken from https://www.tensorflow.org/tutorials/audio_recognition
		'''
		(fingerprint_input)
		  v
		[Conv2D]<-(weights)
		  v
		[BiasAdd]<-(bias)
		  v
		[Relu]
		  v
		[MaxPool]
		  v
		[Conv2D]<-(weights)
		  v
		[BiasAdd]<-(bias)
		  v
		[Relu]
		  v
		[MaxPool]
		  v
		[MatMul]<-(weights)
		  v
		[BiasAdd]<-(bias)
		  v
		'''
		fingerprint = tf.reshape(
				self._fingerprint_pl,
				[-1, self._config.spectrogram_size,
						self._config.dct_coefficient_count, 1])
		filter_height = 20
		filter_width = 8
		filter_count = 64
		weights = tf.Variable(
				tf.truncated_normal(
						[filter_height, filter_width, 1, filter_count],
						stddev=0.01),
				name='weights_var')
		bias = tf.Variable(
				tf.zeros([filter_count]),
				name='bias_var')
		conv = tf.nn.conv2d(
				fingerprint,
				weights,
				[1, 1, 1, 1],
				'SAME') \
						+ bias

		relu = tf.nn.relu(conv)
		dropout = tf.nn.dropout(relu, self._dropout_pl)

		max_pool = tf.nn.max_pool(
				dropout,
				[1, 2, 2, 1], [1, 2, 2, 1],
				'SAME')

		filter2_height = 10
		filter2_width = 4
		filter2_count = 64
		weights2 = tf.Variable(
				tf.truncated_normal(
						[filter2_height, filter2_width,
								filter_count, filter2_count],
						stddev=0.01),
				name='weights2_var')
		bias2 = tf.Variable(
				tf.zeros([filter2_count]),
				name='bias2_var')
		conv2 = tf.nn.conv2d(
				max_pool,
				weights2,
				[1, 1, 1, 1],
				'SAME') \
						+ bias2

		relu2 = tf.nn.relu(conv2)
		dropout2 = tf.nn.dropout(relu2, self._dropout_pl)

		conv2_shape = dropout2.get_shape()
		conv2_output_height = conv2_shape[1]
		conv2_output_width = conv2_shape[2]
		conv2_element_count = int(
				conv2_output_width \
				* conv2_output_height \
				* filter2_count)
		conv2 = tf.reshape(
				dropout2,
				[-1, conv2_element_count])

		label_count = len(self._ds.labels)
		final_fc_weights = tf.Variable(
				tf.truncated_normal(
						[conv2_element_count, label_count],
						stddev=0.01),
				name='final_fc_weights')

		final_fc_bias = tf.Variable(
				tf.zeros([label_count]),
				name='final_fc_bias')
		final_fc = tf.matmul(conv2, final_fc_weights) + final_fc_bias

		return final_fc

	def _init_backprop(self, logits):
		self._cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
				labels=self._ground_truth_pl, logits=logits)
		tf.summary.scalar('cross_entropy', self._cross_entropy_mean)

		self._train_step = tf.train.GradientDescentOptimizer(
				self._learning_rate_pl).minimize(self._cross_entropy_mean)

		predicted_indices = tf.argmax(logits, 1)
		correct_prediction = tf.equal(
				predicted_indices, self._ground_truth_pl)
		self._evaluation_step = tf.reduce_mean(
				tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', self._evaluation_step)

	def do_step(self, step_n, learning_rate):
		fingerprints, ground_truth = self._ds.get_batch(
				'training',
				self._config.batch_size, 0,
				self._config.max_shift,
				self._config.noise_fraction,
				self._config.max_noise_volume)
		summary, accuracy, cross_entropy, _, _ = self._sess.run(
				[self._merged_summaries, self._evaluation_step,
						self._cross_entropy_mean,
						self._train_step, self._increment_global_step],
				feed_dict={
					self._fingerprint_pl: fingerprints,
					self._ground_truth_pl: ground_truth,
					self._learning_rate_pl: learning_rate,
					self._dropout_pl: 0.5
				})

		self._train_writer.add_summary(summary, step_n)

		return (accuracy, cross_entropy)

	def do_validation(self, step_n):
		total_accuracy = 0
		partition_size = self._ds.partition_size('validation')
		for i in range(0, partition_size, self._config.batch_size):
			fingerprints, ground_truth = self._ds.get_batch(
					'validation',
					self._config.batch_size, i,
					0, 0, 0)
			summary, accuracy = self._sess.run(
					[self._merged_summaries, self._evaluation_step],
					feed_dict={
						self._fingerprint_pl: fingerprints,
						self._ground_truth_pl: ground_truth,
						self._dropout_pl: 1.0
					})
			batch_size = min(self._config.batch_size, partition_size - i)
			total_accuracy += (accuracy * batch_size) / partition_size

			self._validation_writer.add_summary(summary, step_n)

		return (total_accuracy, partition_size)

	def do_final_test(self):
		total_accuracy = 0
		partition_size = self._ds.partition_size('testing')
		for i in range(0, partition_size, self._config.batch_size):
			fingerprints, ground_truth = self._ds.get_batch(
					'testing',
					self._config.batch_size, i,
					0, 0, 0)
			accuracy = self._sess.run(
					self._evaluation_step,
					feed_dict={
						self._fingerprint_pl: fingerprints,
						self._ground_truth_pl: ground_truth,
						self._dropout_pl: 1.0
					})
			batch_size = min(self._config.batch_size, partition_size - i)
			total_accuracy += (accuracy * batch_size) / partition_size

		return (total_accuracy, partition_size)

	def save(self, step_n):
		saver = tf.train.Saver(tf.global_variables())
		model_file = os.path.join(OPTIONS.train_dir, 'model.ckpt')
		saver.save(self._sess, model_file, global_step=step_n)

	def restore(self, model_file):
		saver = tf.train.Saver(tf.global_variables())
		saver.restore(self._sess, model_file)
		step = self._global_step.eval(session=self._sess)
		return step


class Config:
	def __init__(self):
		self.sample_rate = OPTIONS.sample_rate
		self.audio_len = int(
				OPTIONS.audio_len_ms * self.sample_rate / 1000)

		window_size_ms = 30
		self.window_size = int(
				window_size_ms * self.sample_rate / 1000)

		window_stride_ms = 10
		self.window_stride = int(
				window_stride_ms * self.sample_rate / 1000)

		self.spectrogram_size = int(
				(self.audio_len - self.window_size)
						/ self.window_stride) + 1
		self.dct_coefficient_count = 40
		self.fingerprint_size = self.spectrogram_size * self.dct_coefficient_count

		self.max_shift = int(
				OPTIONS.max_shift_ms * self.sample_rate / 1000)
		self.noise_fraction = OPTIONS.noise_fraction
		self.max_noise_volume = OPTIONS.max_noise_volume

		self.phases = [{
			'steps': int(p.split(':')[0]),
			'rate': float(p.split(':')[1])
		} for p in OPTIONS.training_phases.split(',')]
		self.num_steps = np.sum([p['steps'] for p in self.phases])

		self.batch_size = OPTIONS.batch_size


def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument(
			'--dataset_dir',
			type=str,
			default='./dataset/')
	parser.add_argument(
			'--words',
			type=str,
			required=True,
			help='Comma-separated labels')
	parser.add_argument(
			'--validation_interval',
			type=int,
			default=200)
	parser.add_argument(
			'--save_interval',
			type=int,
			default=100)
	parser.add_argument(
			'--train_dir',
			type=str,
			default='./train/')
	parser.add_argument(
			'--checkpoint',
			type=str,
			default='')
	parser.add_argument(
			'--summaries_dir',
			type=str,
			default='./train_logs/')

	parser.add_argument(
			'--audio_len_ms',
			type=int,
			default=1000)
	parser.add_argument(
			'--sample_rate',
			type=int,
			default=16000)
	parser.add_argument(
			'--max_shift_ms',
			type=int,
			default=100)
	parser.add_argument(
			'--noise_fraction',
			type=float,
			default=0.8)
	parser.add_argument(
			'--max_noise_volume',
			type=float,
			default=0.1)
	parser.add_argument(
			'--batch_size',
			type=int,
			default=100)
	parser.add_argument(
			'--training_phases',
			type=str,
			default='15000:0.001,3000:0.0001')

	return parser

if __name__ == '__main__':
	parser = argparser()
	OPTIONS = parser.parse_args()
	main()
