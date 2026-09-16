from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from adaline import Adaline, load_dataset, split_indices


class AdalineTests(unittest.TestCase):
    def test_incremental_training_matches_manual_delta_and_batch_api(self):
        features, targets = load_dataset(Path(__file__).with_name('Basedados_B2.csv'))
        train, unused = split_indices(targets)
        model = Adaline()
        reference = Adaline()
        expected = reference.fit(features[train], targets[train], epochs=3)
        initial = model.weights.copy()
        iterator = model.iter_fit(features[train], targets[train], epochs=3)
        np.testing.assert_array_equal(model.weights, initial)
        history = []
        count = 0
        for step in iterator:
            count += 1
            np.testing.assert_allclose(step['weights_after'], step['weights_before'] +
                                       model.learning_rate * step['error'] * step['features'])
            self.assertAlmostEqual(step['linear_output'],
                                   step['features'] @ step['weights_before'] + step['bias_before'])
            if step['epoch_result'] is not None:
                history.append(step['epoch_result'])
        self.assertEqual(count, 3 * len(train))
        self.assertEqual(history, expected)
        np.testing.assert_array_equal(model.weights, reference.weights)
        self.assertEqual(model.bias, reference.bias)

    def test_delta_rule_uses_linear_error_even_when_class_is_correct(self):
        model = Adaline(learning_rate=0.1)
        model.weights[:] = [0.2, 0.1]
        model.bias = 0.1
        features = np.array([[1.0, 2.0]])
        self.assertEqual(model.predict(features)[0], 1)
        history = model.fit(features, np.array([1]), epochs=1)
        np.testing.assert_allclose(model.weights, [0.25, 0.2])
        self.assertAlmostEqual(model.bias, 0.15)
        self.assertAlmostEqual(history[0]['online_sse'], 0.25)
        self.assertAlmostEqual(history[0]['sse'], 0.04)

    def test_b2_decimal_comma_and_disjoint_split(self):
        features, targets = load_dataset(Path(__file__).with_name('Basedados_B2.csv'))
        self.assertEqual(features.shape, (20, 2))
        np.testing.assert_allclose(features[0], [2.215, 2.063])
        self.assertEqual(set(targets), {-1, 1})
        train, test = split_indices(targets, 0.3, seed=42)
        self.assertEqual((len(train), len(test)), (14, 6))
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train) | set(test), set(range(20)))
        self.assertEqual(set(targets[test]), {-1, 1})
        np.testing.assert_array_equal(train, split_indices(targets, 0.3, seed=42)[0])

    def test_training_reduces_total_error_and_test_does_not_mutate_weights(self):
        features, targets = load_dataset(Path(__file__).with_name('Basedados_B2.csv'))
        train, test = split_indices(targets, 0.3, seed=42)
        model = Adaline()
        initial = np.sum((targets[train] - model.decision_function(features[train])) ** 2)
        history = model.fit(features[train], targets[train], epochs=200)
        self.assertEqual(len(history), 200)
        self.assertLess(history[-1]['sse'], initial)
        self.assertTrue(np.isfinite([entry['sse'] for entry in history]).all())
        self.assertAlmostEqual(history[-1]['sse'], np.sum(
            (targets[train] - model.decision_function(features[train])) ** 2))
        weights, bias = model.weights.copy(), model.bias
        predictions = model.predict(features[test])
        self.assertGreaterEqual(np.mean(predictions == targets[test]), 0.8)
        np.testing.assert_array_equal(weights, model.weights)
        self.assertEqual(bias, model.bias)

    def test_invalid_parameters_and_zero_threshold(self):
        for rate in (0, -0.1, float('nan')):
            with self.assertRaises(ValueError):
                Adaline(learning_rate=rate)
        model = Adaline()
        model.weights[:] = 0
        model.bias = 0
        np.testing.assert_array_equal(model.predict(np.zeros((1, 2))), [1])
        with self.assertRaises(ValueError):
            model.fit(np.zeros((1, 2)), np.array([0]))
        with self.assertRaises(ValueError):
            model.fit(np.zeros((0, 2)), np.array([]))

    def test_saved_outputs_reproduce_predictions_and_both_evaluation_modes(self):
        import matplotlib

        matplotlib.use('Agg')
        from matplotlib.image import imread
        from main import run_experiment

        base = Path(__file__).with_name('Basedados_B2.csv')
        for fraction in (0.3, 0.0):
            with self.subTest(test_fraction=fraction), tempfile.TemporaryDirectory() as folder:
                with redirect_stdout(io.StringIO()):
                    run_experiment(base, folder, test_size=fraction, show=False)
                output = Path(folder)
                summary = json.loads((output / 'modelo.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['independent_test'], fraction > 0)
                if fraction == 0:
                    self.assertEqual(summary['train_csv_lines'], summary['test_csv_lines'])
                    self.assertEqual(len(summary['train_csv_lines']), 20)
                else:
                    self.assertFalse(set(summary['train_csv_lines']) & set(summary['test_csv_lines']))
                with (output / 'predicoes.csv').open(newline='', encoding='utf-8') as source:
                    predictions = list(csv.DictReader(source))
                self.assertEqual(len(predictions), 6 if fraction else 20)
                for row in predictions:
                    score = np.dot(summary['weights'], [float(row['s1']), float(row['s2'])]) + summary['bias']
                    self.assertAlmostEqual(score, float(row['linear_output']))
                    self.assertEqual(1 if score >= 0 else -1, int(row['prediction']))
                with (output / 'historico.csv').open(newline='', encoding='utf-8') as source:
                    history = list(csv.DictReader(source))
                self.assertEqual(len(history), 200)
                self.assertAlmostEqual(float(history[-1]['sse']), summary['final_sse'])
                self.assertGreater(imread(output / 'resultado.png').std(), 0.05)


class TrainingWindowTests(unittest.TestCase):
    def test_live_controls_restart_and_completion_match_fit(self):
        import tkinter as tk
        from interface import TrainingWindow

        features, targets = load_dataset(Path(__file__).with_name('Basedados_B2.csv'))
        train, unused = split_indices(targets)
        root = tk.Tk()
        errors = []
        root.report_callback_exception = lambda *error: errors.append(error)
        try:
            model = Adaline()
            initial = model.weights.copy()
            app = TrainingWindow(root, model, features[train], targets[train], 2)
            root.geometry('1100x820')
            root.update()
            app.single_step()
            self.assertEqual(app.last_step['sample_index'], 0)
            self.assertEqual(app.history, [])
            self.assertEqual(len(app.network.find_withtag('input')), 3)
            self.assertEqual(len(app.network.find_withtag('neuron')), 1)
            app.delay.set(1000)
            app.toggle()
            self.assertTrue(app.running)
            self.assertIsNotNone(app.job)
            app.toggle()
            self.assertFalse(app.running)
            self.assertIsNone(app.job)
            paused_weights = model.weights.copy()
            root.after(30, root.quit)
            root.mainloop()
            np.testing.assert_array_equal(model.weights, paused_weights)
            app.reset()
            np.testing.assert_array_equal(model.weights, initial)
            self.assertIsNone(app.last_step)
            self.assertEqual(app.history, [])
            app.mode.set('Epoca')
            app.delay.set(20)
            app.toggle()
            root.after(400, root.quit)
            root.mainloop()
            self.assertTrue(app.completed)
            self.assertFalse(app.running)
            self.assertIsNone(app.job)
            self.assertEqual(str(app.finish_button['state']), 'normal')
            reference = Adaline()
            expected = reference.fit(features[train], targets[train], epochs=2)
            self.assertEqual(app.history, expected)
            np.testing.assert_array_equal(model.weights, reference.weights)
            self.assertEqual(model.bias, reference.bias)
            self.assertEqual(len(app.final_line.get_xdata()), 3)
            app.canvas.draw()
            self.assertGreater(np.asarray(app.canvas.buffer_rgba()).std(), 5)
            app.accept()
            self.assertTrue(app.accepted)
            app.reset()
            app.close()
            self.assertFalse(app.accepted)
            self.assertIsNone(app.job)
        finally:
            root.destroy()
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()