import recognize

model_file = './tests/train/checkpoint'
labels_file = './tests/train/labels.txt'

class TestRecognize:
	def setup(self):
		print('')
		recognize.OPTIONS = recognize.argparser().parse_args([
				'--model_file', '',
				'--labels_file', '',
				'--wav_file', ''])

	def test_recognize_ds_sample(self):
		recognized = recognize.recognize(
				model_file,
				labels_file,
				'./tests/dataset/stop/ffd2ba2f_nohash_3.wav')
		ground_truth = [('stop', 200)]
		self._assert_recognized(recognized, ground_truth, 100)

	def test_recognize_stream(self):
		recognized = recognize.recognize(
				model_file,
				labels_file,
				'./tests/stream.wav')
		ground_truth = self._read_ground_truth('./tests/stream_ground_truth.txt')
		self._assert_recognized(recognized, ground_truth, 70)

	def _assert_recognized(self, recognized, ground_truth, percent):
		recognized = {str(item[0]): item[1] for item in recognized}
		matched = []
		not_matched = []
		spreading_ms = 750
		for item in ground_truth:
			is_matched = False
			for i in range(item[1] - spreading_ms, item[1] + spreading_ms):
				key = str(round(i));
				if key in recognized and item[0] == recognized[key]:
					is_matched = True
					del recognized[key]
					break
			if is_matched:
				matched.append(item);
			else:
				not_matched.append(item)
		matched_percent = len(matched) * 100 / len(ground_truth)
		print('matched', len(matched), str(matched_percent) + '%')
		assert(matched_percent >= percent)

	def _read_ground_truth(self, filename):
		result = []
		with open(filename, 'r') as f:
			for line in f:
				pieces = line.split(',')
				if len(pieces) != 2:
					continue
				label = pieces[0].strip()
				time = round(float(pieces[1].strip()))
				result.append(
						(label, time))
		return result
