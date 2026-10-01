"""Testes numericos da retropropagacao e do ajuste dos pontos da Aula 06."""

import unittest
import csv
import json
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from PIL import Image

from mlp import INPUTS, TARGETS, MLPRegressor


class MLPTests(unittest.TestCase):
    def test_backprop_matches_finite_differences(self):
        model = MLPRegressor(hidden_size=3)
        inputs, targets = INPUTS[:4], TARGETS[:4]
        _, gradients = model.loss_and_gradients(inputs, targets)
        epsilon = 1e-6
        for parameter, gradient in zip(model.parameters, gradients):
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                upper = model.loss_and_gradients(inputs, targets)[0]
                parameter[index] = original - epsilon
                lower = model.loss_and_gradients(inputs, targets)[0]
                parameter[index] = original
                self.assertAlmostEqual(gradient[index], (upper - lower) / (2 * epsilon), places=7)

    def test_one_step_is_gradient_descent(self):
        model = MLPRegressor()
        initial = [parameter.copy() for parameter in model.parameters]
        loss, gradients = model.loss_and_gradients(INPUTS, TARGETS)
        history = model.fit(INPUTS, TARGETS, epochs=1, tolerance=0)
        self.assertEqual(history[0].mse, loss)
        self.assertEqual(history[-1].epoch, 1)
        for before, gradient, after in zip(initial, gradients, model.parameters):
            np.testing.assert_allclose(after, before - model.learning_rate * gradient)

    def test_fit_sampled_function(self):
        model = MLPRegressor()
        history = model.fit(INPUTS, TARGETS)
        predictions = model.predict(INPUTS)
        self.assertLessEqual(history[-1].mse, 1e-4)
        self.assertLess(history[-1].mse, history[0].mse / 100)
        self.assertLess(float(np.max(np.abs(predictions - TARGETS))), 0.03)
        self.assertAlmostEqual(history[-1].mse, float(np.mean((predictions - TARGETS) ** 2)))
        saved = [parameter.copy() for parameter in model.parameters]
        self.assertEqual(model.predict(np.linspace(0, 1, 501)).shape, (501,))
        for before, after in zip(saved, model.parameters):
            np.testing.assert_array_equal(before, after)

    def test_seed_and_early_stop(self):
        first, second = MLPRegressor(), MLPRegressor()
        for first_parameter, second_parameter in zip(first.parameters, second.parameters):
            np.testing.assert_array_equal(first_parameter, second_parameter)
        targets = first.predict(INPUTS)
        self.assertEqual(len(first.fit(INPUTS, targets)), 1)

    def test_invalid_data_and_settings(self):
        for settings in ({'hidden_size': 0}, {'learning_rate': float('nan')}, {'learning_rate': -1}):
            with self.assertRaises(ValueError):
                MLPRegressor(**settings)
        model = MLPRegressor()
        for inputs, targets in (([], []), ([0, 1], [1]), ([np.nan], [1]), ([0], [np.inf])):
            with self.assertRaises(ValueError):
                model.fit(np.asarray(inputs), np.asarray(targets))
        for settings in ({'epochs': 0}, {'tolerance': -1}, {'tolerance': np.nan}):
            with self.assertRaises(ValueError):
                model.fit(INPUTS, TARGETS, **settings)


class ExportTests(unittest.TestCase):
    def test_exports_reproduce_predictions(self):
        import matplotlib

        matplotlib.use('Agg')
        from main import run_experiment

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output = Path(directory)
            summary = run_experiment(output, epochs=20, tolerance=0, show=False)
            saved = json.loads((output / 'modelo.json').read_text(encoding='utf-8'))
            self.assertEqual(summary, saved)
            self.assertFalse(saved['independent_test'])
            self.assertFalse(saved['target_reached'])
            self.assertEqual(saved['stop_reason'], 'max_epochs')
            parameters = saved['parameters']
            hidden = np.tanh((2 * INPUTS[:, None] - 1) @ np.array(parameters['hidden_weights'])
                             + np.array(parameters['hidden_bias']))
            reproduced = hidden @ np.array(parameters['output_weights']) + parameters['output_bias'][0]
            with (output / 'predicoes.csv').open(newline='', encoding='utf-8') as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 11)
            np.testing.assert_allclose(reproduced, [float(row['prediction']) for row in rows])
            self.assertAlmostEqual(saved['final_mse'], float(np.mean((reproduced - TARGETS) ** 2)))
            with (output / 'historico.csv').open(newline='', encoding='utf-8') as source:
                history = list(csv.DictReader(source))
            self.assertEqual(len(history), 21)
            self.assertEqual(float(history[-1]['mse']), saved['final_mse'])
            with (output / 'curva.csv').open(newline='', encoding='utf-8') as source:
                curve = list(csv.DictReader(source))
            self.assertEqual(len(curve), 501)
            self.assertEqual(float(curve[0]['x']), 0)
            self.assertEqual(float(curve[-1]['x']), 1)
            with Image.open(output / 'resultado.png') as image:
                self.assertGreater(image.width, 1000)
                self.assertGreater(float(np.asarray(image).std()), 0)


if __name__ == '__main__':
    unittest.main()