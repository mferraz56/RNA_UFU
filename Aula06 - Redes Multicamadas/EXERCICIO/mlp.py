"""MLP de regressao com tanh, saida linear e retropropagacao em lote."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


INPUTS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
TARGETS = np.array([-0.9602, -0.5770, -0.0729, 0.3771, 0.6405, 0.6600,
                    0.4609, 0.1336, -0.2013, -0.4344, -0.5000])


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    mse: float


class MLPRegressor:
    """Aproxima uma funcao escalar; transforma x em 2*x-1 antes do forward."""

    def __init__(self, hidden_size: int = 10, learning_rate: float = 0.05, seed: int = 42):
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError('A camada oculta deve ter um numero inteiro positivo de neuronios.')
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('A taxa de aprendizado deve ser positiva e finita.')
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.seed = seed
        random = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (hidden_size + 1))
        self.hidden_weights = random.uniform(-limit, limit, (1, hidden_size))
        self.hidden_bias = random.uniform(-0.1, 0.1, hidden_size)
        self.output_weights = random.uniform(-limit, limit, hidden_size)
        self.output_bias = np.zeros(1)

    @property
    def parameters(self) -> tuple[NDArray[np.float64], ...]:
        """Pesos e biases na mesma ordem dos gradientes da retropropagacao."""
        return self.hidden_weights, self.hidden_bias, self.output_weights, self.output_bias

    def _inputs(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(inputs, dtype=float)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError('As entradas devem ser um vetor nao vazio com valores finitos.')
        return (2.0 * values - 1.0).reshape(-1, 1)

    def _forward(self, values: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        hidden = np.tanh(values @ self.hidden_weights + self.hidden_bias)
        return hidden, hidden @ self.output_weights + self.output_bias[0]

    def predict(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Retorna valores continuos, sem modificar os pesos do modelo."""
        return self._forward(self._inputs(inputs))[1]

    def loss_and_gradients(self, inputs: NDArray[np.float64], targets: NDArray[np.float64]
                           ) -> tuple[float, tuple[NDArray[np.float64], ...]]:
        """Calcula EQM e suas derivadas para todos os pesos e biases."""
        values = self._inputs(inputs)
        targets = np.asarray(targets, dtype=float)
        if targets.shape != (len(values),) or not np.isfinite(targets).all():
            raise ValueError('Informe um alvo finito para cada entrada.')
        hidden, predictions = self._forward(values)
        residuals = predictions - targets
        output_delta = 2.0 * residuals / len(values)
        hidden_delta = output_delta[:, None] * self.output_weights * (1.0 - hidden ** 2)
        gradients = (values.T @ hidden_delta, hidden_delta.sum(axis=0),
                     hidden.T @ output_delta, np.array([output_delta.sum()]))
        return float(np.mean(residuals ** 2)), gradients

    def fit(self, inputs: NDArray[np.float64], targets: NDArray[np.float64],
            epochs: int = 100000, tolerance: float = 1e-4) -> list[EpochResult]:
        """Treina por gradiente descendente; para na meta de EQM ou no limite de epocas."""
        if type(epochs) is not int or epochs < 1:
            raise ValueError('O numero de epocas deve ser um inteiro positivo.')
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('A tolerancia deve ser finita e nao negativa.')
        history = []
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                loss, gradients = self.loss_and_gradients(inputs, targets)
                history.append(EpochResult(0, loss))
                for epoch in range(1, epochs + 1):
                    if loss <= tolerance:
                        break
                    for parameter, gradient in zip(self.parameters, gradients):
                        parameter -= self.learning_rate * gradient
                    loss, gradients = self.loss_and_gradients(inputs, targets)
                    if not np.isfinite(loss):
                        raise FloatingPointError
                    history.append(EpochResult(epoch, loss))
        except FloatingPointError as error:
            raise ValueError('O treinamento divergiu. Reduza a taxa de aprendizado.') from error
        return history