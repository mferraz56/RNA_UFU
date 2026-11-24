import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data import CLASSES, ExperimentConfig, SolarDataModule
from model import SimpleSolarCNN
import torch


ROOT = Path(__file__).resolve().parent


class SyntheticSolarTests(unittest.TestCase):
    def test_config_and_model_shape(self):
        config = ExperimentConfig.load(ROOT / "experiment.json")
        self.assertEqual(config.image_shape, (40, 24))
        self.assertEqual(config.num_classes, 12)
        model = SimpleSolarCNN(12)
        output = model(torch.zeros(2, 1, 40, 24))
        self.assertEqual(output.shape, (2, 12))

    def test_split_is_stratified(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            images = base / "images"
            images.mkdir()
            metadata = {}
            for index in range(240):
                class_name = CLASSES[index % 12]
                filename = f"{index:03d}.jpg"
                image = np.full((40, 24), index % 255, dtype=np.uint8)
                Image.fromarray(image).save(images / filename)
                metadata[str(index)] = {
                    "image_filepath": f"images/{filename}",
                    "anomaly_class": class_name,
                }
            (base / "module_metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            module = SolarDataModule(base)
            module.split(0.70, 0.15, 0.15)
            self.assertEqual(len(module.train_indices) + len(module.val_indices) + len(module.test_indices), 240)
            self.assertEqual(len(set(np.unique(module.labels[module.train_indices]).tolist())), 12)

    def test_invalid_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            values = json.loads((ROOT / "experiment.json").read_text(encoding="utf-8"))
            values["dataset"]["train_ratio"] = 0.4
            path.write_text(json.dumps(values), encoding="utf-8")
            with self.assertRaises(ValueError):
                ExperimentConfig.load(path)


if __name__ == "__main__":
    unittest.main()
