import argparse
import tensorflow as tf
import numpy as np
import audio

OPTIONS = None

def main():
	recognize(OPTIONS.model_file, OPTIONS.labels_file, OPTIONS.wav_file)

def recognize(model_file, labels_file, wav_file):
	sess = tf.InteractiveSession()
	load_model(sess, model_file)
	softmax = tf.nn.softmax(
			sess.graph.get_operation_by_name('MatMul').outputs[0],
			1)

	labels = read_labels(labels_file)
	scores_filter = ScoreFilter(labels)

	audio_data = audio.read_wav(wav_file)
	sample_count = len(audio_data)
	sample_rate = OPTIONS.sample_rate
	chunk_len = int((OPTIONS.chunk_len_ms * sample_rate) / 1000)
	chunk_stride = int((OPTIONS.chunk_stride_ms * sample_rate) / 1000)

	recognized = []

	offset = 0
	end = sample_count
	while True:
		if offset >= end:
			break
		chunk_start = offset
		chunk_end = chunk_start + chunk_len

		audio_chunk = np.array([audio_data[chunk_start:chunk_end]], np.float32)
		tail = 0
		if chunk_end >= end:
			tail = chunk_end - end
			audio_chunk = np.append(audio_chunk, np.zeros(tail))
		audio_chunk = audio_chunk.reshape([chunk_len, 1])

		result = sess.run(['Mfcc:0'], feed_dict={
			'Add:0': audio_chunk,
			'DecodeWav:1': sample_rate
		})
		fingerprint = result[0].flatten()
		fingerprint = fingerprint.reshape([1, len(fingerprint)])

		result = sess.run([softmax], feed_dict={
			'fingerprint_pl:0': fingerprint,
			'dropout_pl:0': 1.0
		})

		current_time_ms = int((offset * 1000) / sample_rate)
		is_recognized, label, time = scores_filter.process(
				result[0].flatten(), current_time_ms)
		if is_recognized:
			print(time, label)
			recognized.append(
					(time, label))

		offset += chunk_stride

	return recognized

def load_model(sess, ckpt_filename):
	saver = tf.train.import_meta_graph(ckpt_filename + '.meta')
	saver.restore(sess, ckpt_filename)

def read_labels(filename):
	labels = []
	with open(filename, 'r') as f:
		for line in f:
			labels.append(line.strip())
	return labels


class ScoreFilter:
	def __init__(self, labels):
		self._labels = labels

		self._prev_top_label = '_silence_'
		self._prev_top_time = -1
		self._queue = []

	def process(self, scores, current_time_ms):
		self._queue.append({
			'time': current_time_ms,
			'scores': scores
		})

		average_window_start = current_time_ms - OPTIONS.average_window_len_ms
		queue_len = len(self._queue)
		for i in range(queue_len - 1, -1, -1):
			if self._queue[i]['time'] < average_window_start:
				self._queue.pop(i)

		if len(self._queue):
			queue_time_delta = current_time_ms - self._queue[0]['time']
			if len(self._queue) < 3 \
					or (queue_time_delta < (OPTIONS.average_window_len_ms / 4)):
				return (False, None, None)

			average_scores = np.zeros(len(self._labels))
			for item in self._queue:
				scores = item['scores']
				for i, score in enumerate(scores):
					average_scores[i] += score / len(self._queue)

			current_top_index = np.argmax(average_scores)
			current_top_score = average_scores[current_top_index]
			current_top_label = self._labels[current_top_index]
			if current_top_label == '_silence_':
				return (False, None, None)

			time_since_last_top = 0
			if self._prev_top_label == '_silence_' \
					or self._prev_top_time == -1:
				time_since_last_top = float('Inf')
			else:
				time_since_last_top = current_time_ms - self._prev_top_time
			if current_top_score >= OPTIONS.score_threshold \
					and time_since_last_top > OPTIONS.suppression_period_ms:
				self._prev_top_label = current_top_label
				self._prev_top_time = current_time_ms
				return (True, current_top_label, current_time_ms)

		return (False, None, None)


def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument(
			'--model_file',
			type=str,
			required=True)
	parser.add_argument(
			'--labels_file',
			type=str,
			required=True)
	parser.add_argument(
			'--wav_file',
			type=str,
			required=True)

	parser.add_argument(
			'--sample_rate',
			type=int,
			default=16000)
	parser.add_argument(
			'--chunk_len_ms',
			type=int,
			default=1000)
	parser.add_argument(
			'--chunk_stride_ms',
			type=int,
			default=100)
	parser.add_argument(
			'--average_window_len_ms',
			type=int,
			default=500)
	parser.add_argument(
			'--suppression_period_ms',
			type=int,
			default=1500)
	parser.add_argument(
			'--score_threshold',
			type=float,
			default=0.35)

	return parser

if __name__ == '__main__':
	parser = argparser()
	OPTIONS = parser.parse_args()
	main()
