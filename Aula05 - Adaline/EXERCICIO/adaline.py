import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(filename):
    """Read CSV fields with decimal commas, preserving continuous inputs."""
    features, targets = [], []
    with Path(filename).open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ['s1', 's2', 't']:
            raise ValueError('O CSV deve conter o cabecalho s1,s2,t.')
        for row in reader:
            try:
                sample = [float(row[name].replace(',', '.')) for name in ('s1', 's2')]
                target = int(row['t'])
                if None in row or target not in (-1, 1) or not np.isfinite(sample).all():
                    raise ValueError
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f'Dados invalidos na linha {reader.line_num}.') from error
            features.append(sample)
            targets.append(target)
    if not features:
        raise ValueError('A base de dados esta vazia.')
    return np.asarray(features, dtype=float), np.asarray(targets, dtype=int)


def split_indices(targets, test_size=0.3, seed=42):
    if not np.isfinite(test_size) or not 0 <= test_size < 1:
        raise ValueError('A fracao de teste deve estar entre 0 (inclusive) e 1 (exclusive).')
    indices = np.arange(len(targets))
    if test_size == 0:
        return indices, indices.copy()
    train, test = train_test_split(indices, test_size=test_size, random_state=seed, stratify=targets)
    return np.sort(train), np.sort(test)


class Adaline:
    """Online delta rule with linear activation; threshold is used only to classify."""

    def __init__(self, input_size=2, learning_rate=0.01, seed=42):
        if not isinstance(input_size, int) or input_size < 1:
            raise ValueError('Numero de entradas invalido.')
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('A taxa de aprendizado deve ser positiva e finita.')
        random = np.random.default_rng(seed)
        self.weights = random.uniform(-0.5, 0.5, input_size)
        self.bias = float(random.uniform(-0.5, 0.5))
        self.learning_rate = learning_rate

    def _features(self, features):
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.weights) or not np.isfinite(values).all():
            raise ValueError('Entradas devem ser uma matriz finita com uma coluna por peso.')
        return values

    def decision_function(self, features):
        return self._features(features) @ self.weights + self.bias

    def predict(self, features):
        return np.where(self.decision_function(features) >= 0, 1, -1)

    def fit(self, features, targets, epochs=200):
        return [step['epoch_result'] for step in self.iter_fit(features, targets, epochs)
                if step['epoch_result'] is not None]

    def iter_fit(self, features, targets, epochs=200):
        values = self._features(features)
        targets = np.asarray(targets, dtype=float)
        if not len(values) or targets.shape != (len(values),) or not np.isin(targets, [-1, 1]).all():
            raise ValueError('Informe amostras e um alvo -1 ou +1 por amostra.')
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError('O numero de epocas deve ser um inteiro positivo.')
        try:
            for epoch in range(1, epochs + 1):
                online_sse = 0.0
                for index, (sample, target) in enumerate(zip(values, targets)):
                    with np.errstate(over='raise', invalid='raise'):
                        weights_before, bias_before = self.weights.copy(), float(self.bias)
                        linear_output = float(sample @ self.weights + self.bias)
                        error = target - linear_output
                        online_sse += error ** 2
                        self.weights += self.learning_rate * error * sample
                        self.bias += self.learning_rate * error
                        epoch_result = None
                        if index == len(values) - 1:
                            residuals = targets - self.decision_function(values)
                            epoch_result = {'epoch': epoch, 'online_sse': float(online_sse),
                                            'sse': float(residuals @ residuals),
                                            'weights': self.weights.copy().tolist(), 'bias': float(self.bias)}
                    yield {'epoch': epoch, 'sample_index': index, 'features': sample.copy(),
                           'target': float(target), 'linear_output': linear_output, 'error': float(error),
                           'weights_before': weights_before, 'bias_before': bias_before,
                           'weights_after': self.weights.copy(), 'bias_after': float(self.bias),
                           'epoch_result': epoch_result}
        except (FloatingPointError, OverflowError) as error:
            raise ValueError('O treinamento divergiu. Diminua a taxa de aprendizado.') from error