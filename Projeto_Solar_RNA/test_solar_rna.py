import unittest
import numpy as np
from Projeto_Solar_RNA.dataset_solar import SolarDataset, CLASSES, get_class_group
from Projeto_Solar_RNA.neural_network import SingleLayerPerceptron, MLPSolarNetwork, evaluate_network


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
        model = SingleLayerPerceptron(num_inputs=960, num_classes=12)
        x = self.dataset.get_feature_vector(0)
        target = self.dataset.labels[0]
        
        pred, scores, probs = model.forward(x)
        self.assertEqual(len(scores), 12)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)
        
        pred, had_error, loss = model.train_step(x, target, lr=0.01)
        self.assertGreaterEqual(loss, 0.0)

    def test_mlp_spatial_grid_forward_and_train(self):
        model = MLPSolarNetwork(num_inputs=960, hidden_grid=(4, 6), num_classes=12)
        self.assertEqual(model.num_hidden, 24)
        self.assertEqual(model.get_hidden_activation_grid().shape, (4, 6))

        x = self.dataset.get_feature_vector(0)
        target = self.dataset.labels[0]

        pred, scores, probs = model.forward(x)
        self.assertEqual(len(scores), 12)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)

        pred, had_error, loss = model.train_step(x, target, lr=0.01)
        self.assertGreaterEqual(loss, 0.0)

    def test_evaluation(self):
        model = SingleLayerPerceptron(num_inputs=960, num_classes=12)
        acc, loss, preds, targets = evaluate_network(model, self.dataset, self.test_idx[:50])
        self.assertTrue(0.0 <= acc <= 1.0)
        self.assertEqual(len(preds), 50)
        self.assertEqual(len(targets), 50)


if __name__ == '__main__':
    unittest.main()
