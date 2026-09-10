import csv
import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest

import numpy as np

from dataset import load_reference_dataset
from perceptron import MultiClassPerceptron


class PerceptronTests(unittest.TestCase):
    def test_wrong_prediction_updates_only_target_and_predicted(self):
        model = MultiClassPerceptron(2, 3, learning_rate=0.2)
        sample = np.array([1.0, 0.5])
        step = model.train_one(sample, 2)
        self.assertEqual(step.predicted, 0)
        self.assertTrue(step.updated)
        np.testing.assert_allclose(step.weights_before, 0)
        np.testing.assert_allclose(model.weights, [[-0.2, -0.1], [0, 0], [0.2, 0.1]])
        np.testing.assert_allclose(model.biases, [-0.2, 0, 0.2])
        self.assertEqual(model.predict_one(sample), 2)
        self.assertFalse(model.train_one(sample, 2).updated)
        np.testing.assert_allclose(step.weights_before, 0)

    def test_audit_reconstructs_every_score(self):
        model = MultiClassPerceptron(2, 3)
        model.train_one(np.array([1.0, 0.4]), 1)
        sample = np.array([0.7, 0.3])
        audit = model.inspect(sample)
        np.testing.assert_allclose(
            audit.contributions.sum(axis=1) + audit.biases, audit.scores
        )
        self.assertEqual(audit.predicted, model.predict_one(sample))
        self.assertEqual(model.updates, 1)

    def test_dataset_split_and_generalization(self):
        data = load_reference_dataset()
        self.assertEqual(data.train_images.shape[1], 64)
        self.assertEqual(len(data.train_labels) + len(data.test_labels), 1797)
        self.assertFalse(set(data.train_indices) & set(data.test_indices))
        self.assertEqual(set(data.test_labels), set(range(10)))
        self.assertGreaterEqual(data.train_images.min(), 0)
        self.assertLessEqual(data.train_images.max(), 1)
        model = MultiClassPerceptron(64, 10, learning_rate=0.1)
        history = model.fit(list(zip(data.train_images, data.train_labels)), epochs=20)
        self.assertEqual(len(history), 20)
        self.assertGreater(model.evaluate(list(zip(data.test_images, data.test_labels))), 0.85)

    def test_invalid_inputs_and_empty_dataset(self):
        with self.assertRaises(ValueError):
            MultiClassPerceptron(2, 3, learning_rate=0)
        model = MultiClassPerceptron(2, 3)
        for sample in ([1], [1, float('nan')]):
            with self.assertRaises(ValueError):
                model.predict_one(np.array(sample))
        with self.assertRaises(ValueError):
            model.train_one(np.ones(2), 3)
        with self.assertRaises(ValueError):
            model.fit([], epochs=1)


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        from interface import PerceptronApp

        self.root = tk.Tk()
        self.app = PerceptronApp(self.root)
        self.callback_errors = []
        self.root.report_callback_exception = lambda *error: self.callback_errors.append(error)
        self.root.update()

    def tearDown(self):
        self.app.close()
        self.assertEqual(self.callback_errors, [])

    def test_drawing_animation_audit_and_exports(self):
        app = self.app
        self.root.geometry('1100x760')
        self.root.update()
        for child in app.sidebar.winfo_children():
            self.assertLessEqual(child.winfo_y() + child.winfo_height(), app.sidebar.winfo_height())
        app.pixel_canvas.event_generate('<Button-1>', x=45, y=75)
        self.assertEqual(app.drawing[17], 1)
        app.pixel_canvas.event_generate('<Button-3>', x=45, y=75)
        self.assertEqual(app.drawing[17], 0)
        app.single_step()
        self.assertEqual(app.pending['phase'], 'entrada')
        app.single_step()
        self.assertEqual(app.pending['phase'], 'somas')
        np.testing.assert_array_equal(app.model.weights, 0)
        app.single_step()
        self.assertIsNone(app.pending)
        self.assertEqual(len(app.records), 1)
        app.capture()
        snapshot = app.snapshot_weights.copy()
        app.load_sample()
        weights = app.model.weights.copy()
        app.delay.set(20)
        app.classify()
        self.root.after(900, self.root.quit)
        self.root.mainloop()
        self.assertFalse(app.classifying)
        self.assertEqual(app.result.get(), f'Digito: {app.model.predict_one(app.drawing)}')
        np.testing.assert_array_equal(app.model.weights, weights)
        np.testing.assert_array_equal(app.snapshot_weights, snapshot)
        for tab in range(4):
            app.tabs.select(tab)
            self.root.update()
        self.assertEqual(len(app.table.get_children()), 65)
        self.assertEqual(len(app.network.find_withtag('input')), 64)
        self.assertEqual(len(app.network.find_withtag('output')), 10)
        self.assertEqual(len(app.confusion_axis.texts), 100)
        for canvas in (app.audit_canvas, app.metrics_canvas, app.gallery_canvas):
            canvas.draw()
            self.assertGreater(np.asarray(canvas.buffer_rgba()).std(), 5)
        with tempfile.TemporaryDirectory() as folder:
            json_path = Path(folder) / 'audit.json'
            csv_path = Path(folder) / 'audit.csv'
            app.export_audit(json_path)
            app.export_audit(csv_path)
            payload = json.loads(json_path.read_text(encoding='utf-8'))
            rebuilt = np.asarray(payload['weights']) @ payload['features'] + payload['biases']
            np.testing.assert_allclose(rebuilt, payload['scores'])
            self.assertEqual(int(np.argmax(rebuilt)), payload['predicted'])
            with csv_path.open(encoding='utf-8', newline='') as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 650)
            for digit in range(10):
                total = sum(float(row['contribution']) for row in rows if int(row['class']) == digit)
                self.assertAlmostEqual(total, payload['scores'][digit])
        app.reset()
        self.assertEqual(app.model.updates, 0)
        self.assertEqual(app.records, [])
        np.testing.assert_array_equal(app.snapshot_weights, 0)

    def test_fast_training_matches_core_and_is_replayable(self):
        app = self.app
        app.epochs.set('20')
        app.speed.set('Rapido (50 amostras/quadro)')
        app.single_step()
        while app.epoch < 20:
            app.running = True
            app.tick()
            app.pause()
        self.assertEqual(len(app.records), 20 * len(app.data.train_labels))
        self.assertIsNone(app.pending)
        self.assertIsNone(app.job)
        reference = MultiClassPerceptron()
        reference.fit(list(zip(app.data.train_images, app.data.train_labels)), epochs=20)
        np.testing.assert_array_equal(reference.weights, app.model.weights)
        np.testing.assert_array_equal(reference.biases, app.model.biases)
        replay = MultiClassPerceptron()
        lookup = dict(zip(app.data.train_indices, app.data.train_images))
        for record in app.records:
            step = replay.train_one(lookup[record['dataset_index']], record['label'])
            self.assertEqual(step.predicted, record['predicted_before'])
            np.testing.assert_allclose(step.scores_before, record['scores_before'])
        np.testing.assert_array_equal(replay.weights, app.model.weights)
        self.assertGreater(app.history[-1]['test'], 0.85)
        print(f'\n20 epocas: treino={app.history[-1]["train"]:.2%}, teste={app.history[-1]["test"]:.2%}')


if __name__ == '__main__':
    unittest.main()