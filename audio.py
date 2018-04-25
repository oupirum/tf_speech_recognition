import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops
import random
import numpy as np

def read_wav(filename):
	with tf.Session(graph=tf.Graph()) as sess:
		wav_filename_ph = tf.placeholder(tf.string, [])
		wav_loader = io_ops.read_file(wav_filename_ph)
		wav_decoder = audio_ops.decode_wav(wav_loader, desired_channels=1)
		data = sess.run(wav_decoder, feed_dict={
			wav_filename_ph: filename
		})
		return data.audio.flatten()

class AudioProcessor:
	def __init__(self, config, noise_wav_files):
		self._config = config
		self._sess = tf.InteractiveSession(graph=tf.Graph())

		self._wav_file_pl = tf.placeholder(
				tf.string, [],
				name='wav_file_pl')
		wav_loader = io_ops.read_file(self._wav_file_pl)
		desired_samples = self._config.audio_len
		wav_decoder = audio_ops.decode_wav(
				wav_loader, desired_channels=1, desired_samples=desired_samples)

		self._volume_pl = tf.placeholder(
				tf.float32, [],
				name='volume_pl')
		self._shift_padding_pl = tf.placeholder(
				tf.int32, [2, 2],
				name='shift_padding_pl')
		self._shift_offset_pl = tf.placeholder(
				tf.int32, [2],
				name='shift_offset_pl')
		data = tf.multiply(
				wav_decoder.audio,
				self._volume_pl)
		data = tf.pad(
				data,
				self._shift_padding_pl,
				mode='CONSTANT')
		data = tf.slice(
				data,
				self._shift_offset_pl,
				[desired_samples, -1])

		self._noise_data_pl = tf.placeholder(
				tf.float32, [desired_samples, 1],
				name='noise_data_pl')
		self._noise_volume_pl = tf.placeholder(
				tf.float32,
				[],
				name='noise_volume_pl')
		noise_data = tf.multiply(
				self._noise_data_pl,
				self._noise_volume_pl)

		data = tf.add(noise_data, data)
		data = tf.clip_by_value(data, -1.0, 1.0)

		spectrogram = audio_ops.audio_spectrogram(
				data,
				window_size=self._config.window_size,
				stride=self._config.window_stride,
				magnitude_squared=True)
		self._mfcc = audio_ops.mfcc(
				spectrogram,
				wav_decoder.sample_rate,
				dct_coefficient_count=self._config.dct_coefficient_count)

		self._preload_noise_data(noise_wav_files)

	def produce_fingerprint(self, wav_file,
			volume,
			max_shift,
			max_noise_volume):
		shift = 0
		if max_shift > 0:
			shift = random.randrange(-max_shift, max_shift + 1)
		if shift > 0:
			shift_offset = [0, 0]
			shift_padding = [[shift, 0], [0, 0]]
		else:
			shift_offset = [-shift, 0]
			shift_padding = [[0, -shift], [0, 0]]

		noise_data, noise_volume = self._get_noise_data(max_noise_volume)

		data = self._sess.run(
				self._mfcc,
				feed_dict={
					self._wav_file_pl: wav_file,
					self._volume_pl: volume,
					self._shift_offset_pl: shift_offset,
					self._shift_padding_pl: shift_padding,
					self._noise_data_pl: noise_data,
					self._noise_volume_pl: noise_volume
				})
		return data.flatten()

	def _preload_noise_data(self, wav_files):
		self._noise_data = []
		with tf.Session(graph=tf.Graph()) as sess:
			wav_file_pl = tf.placeholder(tf.string, [])
			wav_loader = io_ops.read_file(wav_file_pl)
			wav_decoder = audio_ops.decode_wav(
					wav_loader,
					desired_channels=1)
			for wav_file in wav_files:
				wav_data = sess.run(
						wav_decoder,
						feed_dict={
							wav_file_pl: wav_file
						})
				self._noise_data.append(wav_data.audio.flatten())

	def _get_noise_data(self, max_noise_volume):
		audio_data_len = self._config.audio_len
		add_noise = self._noise_data and max_noise_volume > 0
		if add_noise:
			noise_data = self._noise_data[
					random.randrange(0, len(self._noise_data))]
			noise_offset = random.randrange(0, len(noise_data) - audio_data_len)
			noise_data = noise_data[
					noise_offset:(noise_offset + audio_data_len)]
			noise_data = noise_data.reshape([audio_data_len, 1])
			noise_volume = random.uniform(0, max_noise_volume)
		else:
			noise_data = np.zeros([audio_data_len, 1])
			noise_volume = 0

		return (noise_data, noise_volume)
