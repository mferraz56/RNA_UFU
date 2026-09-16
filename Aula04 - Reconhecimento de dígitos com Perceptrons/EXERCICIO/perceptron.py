from dataclasses import dataclass

import numpy as np


@dataclass
class Inspection:
	features: np.ndarray
	contributions: np.ndarray
	biases: np.ndarray
	scores: np.ndarray
	predicted: int


@dataclass
class TrainingStep:
	features: np.ndarray
	label: int
	predicted: int
	scores_before: np.ndarray
	weights_before: np.ndarray
	biases_before: np.ndarray
	updated: bool


class MultiClassPerceptron:
	"""Single linear layer: 64 pixels -> 10 scores -> argmax, without softmax."""

	def __init__(self, input_size=64, n_classes=10, learning_rate=0.1):
		if input_size < 1 or n_classes < 2:
			raise ValueError('Dimensoes invalidas.')
		if not np.isfinite(learning_rate) or learning_rate <= 0:
			raise ValueError('A taxa de aprendizado deve ser positiva e finita.')
		self.input_size = input_size
		self.n_classes = n_classes
		self.learning_rate = learning_rate
		self.weights = np.zeros((n_classes, input_size), dtype=float)
		self.biases = np.zeros(n_classes, dtype=float)
		self.updates = 0

	def _features(self, sample):
		features = np.asarray(sample, dtype=float)
		if features.shape != (self.input_size,) or not np.isfinite(features).all():
			raise ValueError('A entrada deve ser um vetor finito com o tamanho da rede.')
		return features

	def inspect(self, sample):
		features = self._features(sample)
		contributions = self.weights * features
		scores = contributions.sum(axis=1) + self.biases
		return Inspection(features.copy(), contributions.copy(), self.biases.copy(),
						  scores.copy(), int(np.argmax(scores)))

	def predict_one(self, sample):
		features = self._features(sample)
		return int(np.argmax(self.weights @ features + self.biases))

	def predict(self, samples):
		return np.array([self.predict_one(sample) for sample in samples], dtype=int)

	def train_one(self, sample, label):
		features = self._features(sample)
		if not isinstance(label, (int, np.integer)) or not 0 <= label < self.n_classes:
			raise ValueError('Rotulo fora das classes da rede.')
		scores = self.weights @ features + self.biases
		predicted = int(np.argmax(scores))
		step = TrainingStep(features.copy(), int(label), predicted, scores.copy(),
							self.weights.copy(), self.biases.copy(), predicted != label)
		if step.updated:
			self.weights[label] += self.learning_rate * features
			self.weights[predicted] -= self.learning_rate * features
			self.biases[label] += self.learning_rate
			self.biases[predicted] -= self.learning_rate
			self.updates += 1
		return step

	def fit(self, dataset, epochs=20, seed=42):
		if not len(dataset) or not isinstance(epochs, int) or epochs < 1:
			raise ValueError('Informe dados e um numero positivo de epocas.')
		random = np.random.default_rng(seed)
		history = []
		for epoch in range(1, epochs + 1):
			mistakes = 0
			for index in random.permutation(len(dataset)):
				features, label = dataset[index]
				mistakes += self.train_one(features, label).updated
			history.append({'epoch': epoch, 'mistakes': mistakes,
							'accuracy': self.evaluate(dataset)})
		return history

	def evaluate(self, dataset):
		if not len(dataset):
			return 0.0
		features = np.asarray([sample for sample, label in dataset])
		labels = np.asarray([label for sample, label in dataset])
		predicted = np.argmax(features @ self.weights.T + self.biases, axis=1)
		return float(np.mean(predicted == labels))
