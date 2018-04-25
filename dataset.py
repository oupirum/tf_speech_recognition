from tensorflow.python.platform import gfile
import os
import random
import numpy as np
import audio

SILENCE_LABEL = '_silence_'
UNKNOWN_WORD_LABEL = '_unknown_'
NOISE_DIR_NAME = '_background_noise_'
VALIDATION_PART_PERC = 10
TESTING_PART_PERC = 10
SILENCE_PERCENT = 10
UNKNOWN_PERCENT = 10

class Dataset:
	def __init__(self, directory, words, config):
		self.labels = [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + words
		self._directory = directory
		self._config = config
		self._audio = audio.AudioProcessor(config,
				self._list_noise_files())
		self._index(words)

	def get_batch(self,
			partition,
			batch_size, offset,
			max_shift,
			noise_fraction, max_noise_volume):
		samples = self._index[partition]
		batch_size = max(0, min(batch_size, len(samples) - offset))
		batch = np.zeros((batch_size, self._config.fingerprint_size))
		goundtruth = np.zeros(batch_size)

		for i in range(offset, offset + batch_size):
			if partition == 'training':
				sample = random.choice(samples)
			else:
				sample = samples[i]

			volume = 1
			if sample['label'] == SILENCE_LABEL:
				volume = 0

			add_noise = random.uniform(0, 1) <= noise_fraction
			batch[i - offset, :] = self._audio.produce_fingerprint(
					sample['file'],
					volume,
					max_shift,
					max_noise_volume if add_noise else 0)

			label = sample['label']
			label_index = self._index_of_label(label)
			goundtruth[i - offset] = label_index

		return (batch, goundtruth.astype(int))

	def partition_size(self, partition):
		return len(self._index[partition])

	def _index(self, words):
		self._index = {
			'training': [],
			'validation': [],
			'testing': []
		}

		index_words = []
		index_unknown = []
		words_set = {}
		wav_files = gfile.Glob(os.path.join(self._directory, '*', '*.wav'))
		for wav_file in wav_files:
			word = os.path.basename(os.path.dirname(wav_file))
			if word == NOISE_DIR_NAME:
				continue
			words_set[word] = True

			if word in words:
				index_words.append({
					'label': word,
					'file': wav_file
				})
			else:
				index_unknown.append({
					'label': word,
					'file': wav_file
				})

		for word in words:
			if word not in words_set:
				raise Exception('word "%s" not found in dataset' % word)

		index_size = len(index_words)
		validation_part_size = round(VALIDATION_PART_PERC * index_size / 100)
		testing_part_size = round(TESTING_PART_PERC * index_size / 100)
		for i in range(0, index_size):
			sample = random.choice(index_words)
			if i < validation_part_size:
				self._index['validation'].append(sample)
			elif i < validation_part_size + testing_part_size:
				self._index['testing'].append(sample)
			else:
				self._index['training'].append(sample)

		random.shuffle(index_unknown)
		for partition in ['training', 'validation', 'testing']:
			partition_size = len(self._index[partition])

			silence_num = round(
					partition_size * SILENCE_PERCENT / (100 - SILENCE_PERCENT))
			for _ in range(0, silence_num):
				self._index[partition].append({
					'label': SILENCE_LABEL,
					'file': index_unknown[0]['file']
				})

			unknown_num = round(
					partition_size * UNKNOWN_PERCENT / (100 - UNKNOWN_PERCENT))
			self._index[partition].extend(index_unknown[:unknown_num])
			if len(index_unknown) < unknown_num:
				for _ in range(0, unknown_num - len(index_unknown)):
					sample = random.choice(index_unknown)
					self._index[partition].append(sample)

			random.shuffle(self._index[partition])

	def _list_noise_files(self):
		wav_files = gfile.Glob(os.path.join(
				self._directory, NOISE_DIR_NAME, '*.wav'))
		return wav_files

	def _index_of_label(self, label):
		if label in self.labels:
			return self.labels.index(label)
		else:
			return self.labels.index(UNKNOWN_WORD_LABEL)
