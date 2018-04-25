import dataset

class TestDataset:
	def setup(self):
		print('')

	def test_index(self):
		ds = dataset.Dataset(
				'./tests/dataset/',
				['go', 'stop', 'on', 'off'],
				ConfigFake())

		training_size = ds.partition_size('training')
		assert(training_size > 0)

		validation_size = ds.partition_size('validation')
		assert(validation_size > 0)
		val_perc = validation_size * 80 / training_size
		assert(val_perc >= 8 and val_perc <= 11)

		testing_size = ds.partition_size('testing')
		assert(testing_size > 0)
		test_perc = testing_size * 80 / training_size
		assert(test_perc >= 8 and test_perc <= 11)

	def test_get_batch(self):
		ds = dataset.Dataset(
				'./tests/dataset/',
				['go', 'stop', 'on', 'off'],
				ConfigFake())

		data, grtr = ds.get_batch(
				'training',
				10, 0,
				1600,
				0.8, 0.1)
		assert(data.shape == (10, 3920))
		assert(len(grtr) == 10)

	def test_get_batch_end(self):
		ds = dataset.Dataset(
				'./tests/dataset/',
				['go', 'stop', 'on', 'off'],
				ConfigFake())

		training_size = ds.partition_size('training')
		data, grtr = ds.get_batch(
				'training',
				10, training_size - 5,
				1600,
				0.8, 0.1)
		assert(data.shape == (5, 3920))
		assert(len(grtr) == 5)


class ConfigFake:
	def __init__(self):
		self.audio_len = 16000
		self.window_size = 480
		self.window_stride = 160
		self.spectrogram_size = 98
		self.dct_coefficient_count = 40
		self.fingerprint_size = 3920
