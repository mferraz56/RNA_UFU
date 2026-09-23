import unittest
import json
import tempfile
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
from PIL import Image
from Projeto_Solar_RNA.dataset_solar import SolarDataset, CLASSES, get_class_group
from Projeto_Solar_RNA.neural_network import SingleLayerPerceptron, MLPSolarNetwork, evaluate_network
from Projeto_Solar_RNA.network_config import load_network_config


class TestSolarFeatures(unittest.TestCase):
    def test_pixels_then_row_and_column_means(self):
        image = np.arange(960, dtype=np.float32).reshape(40, 24) / 959
        features = SolarDataset.build_feature_vector(image)
        self.assertEqual(features.shape, (1039,))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_array_equal(features[:960], image.ravel())
        np.testing.assert_allclose(features[960:1000], image.mean(axis=1))
        np.testing.assert_allclose(features[1000:1024], image.mean(axis=0))
        expected_masks = [image[row:row + 8, column:column + 8].mean()
                          for row in range(0, 40, 8) for column in range(0, 24, 8)]
        np.testing.assert_allclose(features[1024:], expected_masks)

    def test_mask_order_and_coverage(self):
        block_values = np.arange(15, dtype=np.float32).reshape(5, 3) / 14
        image = np.repeat(np.repeat(block_values, 8, axis=0), 8, axis=1)
        features = SolarDataset.build_feature_vector(image)
        np.testing.assert_allclose(features[1024:], block_values.ravel(), atol=1e-7)
        config = load_network_config()
        self.assertEqual(config.mask_grid, (5, 3))
        self.assertEqual(config.num_inputs, 1039)
        self.assertEqual(config.hidden_grid, (64, 1))
        self.assertEqual(config.output_grid, (12, 1))

    def test_invalid_image_shape(self):
        with self.assertRaises(ValueError):
            SolarDataset.build_feature_vector(np.zeros((24, 40)))

    def test_statistics_include_all_images_and_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {}
            for index, (minimum, maximum) in enumerate(((0, 200), (20, 255), (0, 255))):
                pixels = np.full((40, 24), minimum, dtype=np.uint8)
                pixels[-1, -1] = maximum
                filename = f'{index}.png'
                Image.fromarray(pixels).save(root / filename)
                metadata[str(index)] = {'anomaly_class': CLASSES[index], 'image_filepath': filename}
            (root / 'module_metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            dataset = SolarDataset(root)
            output = dataset.export_intensity_statistics(root / 'results' / 'statistics.json')
            statistics = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(statistics['image_count'], 3)
            self.assertEqual(statistics['darkest']['value'], 0)
            self.assertEqual(statistics['lightest']['value'], 255)
            self.assertEqual(statistics['darkest']['images'], [
                {'id': '0', 'class': CLASSES[0]}, {'id': '2', 'class': CLASSES[2]}])
            self.assertEqual(statistics['lightest']['images'], [
                {'id': '1', 'class': CLASSES[1]}, {'id': '2', 'class': CLASSES[2]}])
            for index, (minimum, maximum) in enumerate(((0, 200), (20, 255), (0, 255))):
                self.assertEqual(statistics['images'][index], {
                    'id': str(index), 'class': CLASSES[index],
                    'min': minimum, 'max': maximum,
                    'min_normalized': minimum / 255, 'max_normalized': maximum / 255})
            self.assertEqual(dataset.get_feature_vector(0).shape, (1039,))
            self.assertEqual(len(dataset.cached_images), 1)


class TestSolarRNA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = SolarDataset()
        cls.train_idx, cls.val_idx, cls.test_idx = cls.dataset.split_dataset(seed=42)

    def test_dataset_loading(self):
        self.assertEqual(len(self.dataset.labels), 20000)
        self.assertEqual(len(CLASSES), 12)
        
        img = self.dataset.load_image(0)
        self.assertEqual(img.shape, (40, 24))
        self.assertTrue(0.0 <= img.min() and img.max() <= 1.0)
        
        # Test semantic grouping
        self.assertEqual(get_class_group('No-Anomaly'), 'Normal')
        self.assertEqual(get_class_group('Hot-Spot'), 'Célula')
        self.assertEqual(get_class_group('Diode'), 'Elétrica')
        self.assertEqual(get_class_group('Vegetation'), 'Externa')

    def test_synthetic_fault_injection(self):
        hs_img = SolarDataset.generate_synthetic_anomaly('Hot-Spot')
        self.assertEqual(hs_img.shape, (40, 24))
        self.assertGreater(hs_img.max(), 0.9)

        diode_img = SolarDataset.generate_synthetic_anomaly('Diode')
        self.assertEqual(diode_img.shape, (40, 24))
        self.assertAlmostEqual(float(diode_img[5, 5]), 0.85, places=2)

    def test_perceptron_forward_and_train(self):
        model = SingleLayerPerceptron()
        x = self.dataset.get_feature_vector(0)
        target = self.dataset.labels[0]
        
        pred, scores, probs = model.forward(x)
        self.assertEqual(len(scores), 12)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)
        
        pred, had_error, loss = model.train_step(x, target, lr=0.01)
        self.assertGreaterEqual(loss, 0.0)

    def test_mlp_spatial_grid_forward_and_train(self):
        model = MLPSolarNetwork()
        self.assertEqual(model.num_inputs, 1039)
        self.assertEqual(model.num_hidden, 64)
        self.assertEqual(model.W1.shape, (64, 1039))
        self.assertEqual(model.W2.shape, (12, 64))
        self.assertEqual(model.get_hidden_activation_grid().shape, (64, 1))

        x = self.dataset.get_feature_vector(0)
        target = self.dataset.labels[0]

        pred, scores, probs = model.forward(x)
        self.assertEqual(len(scores), 12)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)

        weights_before = model.W1.copy()
        pred, had_error, loss = model.train_step(x, target, lr=0.01)
        self.assertGreaterEqual(loss, 0.0)
        self.assertFalse(np.array_equal(weights_before, model.W1))
        self.assertTrue(np.isfinite(model.W1).all())

    def test_receptive_fields_include_mean_features(self):
        model = MLPSolarNetwork()
        image = self.dataset.load_image(0)
        features = self.dataset.get_feature_vector(0)
        field = model.get_hidden_neuron_receptive_field(0)
        self.assertEqual(field.shape, (40, 24))
        self.assertAlmostEqual(float(np.sum(field * image)), float(model.W1[0] @ features), places=5)
        output_field = model.get_receptive_field(0)
        self.assertEqual(output_field.shape, (40, 24))
        self.assertAlmostEqual(float(np.sum(output_field * image)),
                               float((model.W2[0] @ model.W1) @ features), places=5)
        perceptron = SingleLayerPerceptron()
        perceptron.weights[0] = model.W1[0]
        np.testing.assert_allclose(perceptron.get_receptive_field(0), field)

    def test_evaluation(self):
        model = SingleLayerPerceptron()
        acc, loss, preds, targets = evaluate_network(model, self.dataset, self.test_idx[:50])
        self.assertTrue(0.0 <= acc <= 1.0)
        self.assertEqual(len(preds), 50)
        self.assertEqual(len(targets), 50)
        acc, loss, preds, targets = evaluate_network(MLPSolarNetwork(), self.dataset, self.test_idx[:50])
        self.assertTrue(0.0 <= acc <= 1.0)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(len(preds), 50)


class TestNetworkConfig(unittest.TestCase):
    def test_custom_json_controls_model_and_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'network.json'
            values = {'image_shape': [40, 24], 'mask_shape': [4, 8],
                      'hidden_grid': [16, 2], 'output_grid': [12, 1],
                      'activation': 'sigmoid', 'seed': 7}
            path.write_text(json.dumps(values), encoding='utf-8')
            config = load_network_config(path)
            model = MLPSolarNetwork(config=config)
            image = np.arange(960, dtype=np.float32).reshape(40, 24) / 959
            features = SolarDataset.build_feature_vector(image, config=config)
            self.assertEqual(features.shape, (1054,))
            self.assertEqual(model.W1.shape, (32, 1054))
            self.assertEqual(model.get_hidden_activation_grid().shape, (16, 2))
            self.assertEqual(model.activation_name, 'sigmoid')
            self.assertEqual(model.forward(features)[2].shape, (12,))
            field = model.get_hidden_neuron_receptive_field(0)
            self.assertAlmostEqual(float(np.sum(field * image)), float(model.W1[0] @ features), places=5)
            np.testing.assert_array_equal(model.W1, MLPSolarNetwork(config=config).W1)
            self.assertEqual(SingleLayerPerceptron(config=config).weights.shape, (12, 1054))

    def test_invalid_configuration(self):
        config = load_network_config()
        for changes in ({'mask_shape': (7, 8)}, {'hidden_grid': (0, 1)},
                        {'output_grid': (13, 1)}, {'activation': 'unknown'}, {'seed': -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)


class TestSolarDashboard(unittest.TestCase):
    def setUp(self):
        from Projeto_Solar_RNA.visualization_solar import SolarNetworkView
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(str(error))
        self.addCleanup(self.root.destroy)
        self.root.geometry('600x400')
        self.view = SolarNetworkView(self.root, config=replace(load_network_config(), hidden_grid=(64, 64)))
        self.view.pack(fill='both', expand=True)
        self.root.update()
        self.view.update_view(np.zeros(1039), probs=np.full(12, 1 / 12), is_mlp=True,
                              hidden_activations=np.arange(4096, dtype=np.float32) / 4095)

    def click_hidden(self, index):
        center_x, center_y = self.view.hidden_nodes[index]
        event = SimpleNamespace(x=center_x - self.view.canvasx(0), y=center_y - self.view.canvasy(0))
        self.view.on_mouse_click(event)
        return event

    def test_dense_grid_selection_and_scrolling(self):
        self.assertEqual(len(self.view.hidden_nodes), 4096)
        self.assertEqual(len(self.view.find_withtag('hidden_heatmap')), 1)
        self.assertLess(len(self.view.find_all()), 1200)
        for index in (0, 63, 64, 2048, 4095):
            self.click_hidden(index)
            self.assertEqual(self.view.selected_hidden, index)
        self.view.xview_moveto(1)
        self.view.yview_moveto(1)
        event = self.click_hidden(4095)
        self.assertEqual(self.view.selected_hidden, 4095)
        self.view.on_mouse_move(event)
        self.assertEqual(self.view.hovered_hidden, 4095)
        detail = self.view.find_withtag('hidden_detail')[0]
        self.assertIn('H4095', self.view.itemcget(detail, 'text'))
        self.assertIn('1.00000', self.view.itemcget(detail, 'text'))

    def test_dense_grid_limits_connections_and_respects_toggle(self):
        self.click_hidden(42)
        self.view.selected_pixel = (0, 0)
        self.view.redraw()
        self.assertEqual(len(self.view.find_withtag('highlight_edge')), 2)
        self.view.state['show_edges'] = False
        self.view.redraw()
        self.assertFalse(self.view.find_withtag('highlight_edge'))

    def test_configured_column_and_mask_inputs(self):
        self.view.config = load_network_config()
        image = np.arange(960, dtype=np.float32).reshape(40, 24) / 959
        features = SolarDataset.build_feature_vector(image, config=self.view.config)
        self.view.update_view(features, probs=np.full(12, 1 / 12), is_mlp=True,
                              hidden_activations=np.arange(64, dtype=np.float32) / 63)
        self.assertEqual(len(self.view.hidden_nodes), 64)
        self.assertEqual(len(self.view.output_nodes), 12)
        self.assertEqual(len({position[0] for position in self.view.hidden_nodes.values()}), 1)
        self.assertEqual(len({position[0] for position in self.view.output_nodes.values()}), 1)
        self.assertGreater(self.view.hidden_nodes[1][1] - self.view.hidden_nodes[0][1], 22)
        self.assertEqual(len(self.view.find_withtag('mask_cell')), 15)
        self.assertEqual(len(self.view.find_withtag('mask_boundary')), 15)
        np.testing.assert_array_equal(self.view.state['mask_means'].ravel(), features[1024:])
        self.view.yview_moveto(1)
        event = self.click_hidden(63)
        self.view.on_mouse_move(event)
        self.assertEqual(self.view.selected_hidden, 63)
        detail = self.view.find_withtag('hidden_detail')[0]
        self.assertIn('H63: 1.00000', self.view.itemcget(detail, 'text'))


if __name__ == '__main__':
    unittest.main()
