from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


CLASSES = [
    "No-Anomaly",
    "Cell",
    "Cell-Multi",
    "Cracking",
    "Hot-Spot",
    "Hot-Spot-Multi",
    "Shadowing",
    "Diode",
    "Diode-Multi",
    "Vegetation",
    "Soiling",
    "Offline-Module",
]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    seed: int
    image_shape: Tuple[int, int]
    num_classes: int
    dataset: Dict[str, float]
    training: Dict[str, object]
    augmentation: Dict[str, object]

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if type(values.get("schema_version")) is not int or values["schema_version"] != 1:
            raise ValueError("schema_version deve ser o inteiro 1")
        if type(values.get("seed")) is not int or values["seed"] < 0:
            raise ValueError("seed deve ser inteiro nao negativo")
        shape = values.get("image_shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("image_shape deve ser [altura, largura]")
        if any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError("Dimensoes da imagem devem ser inteiros positivos")
        if tuple(shape) != (40, 24):
            raise ValueError("Este dataset requer image_shape [40, 24]")
        if values.get("num_classes") != 12:
            raise ValueError("Este dataset possui 12 classes")

        ratios = values.get("dataset")
        if not isinstance(ratios, dict):
            raise ValueError("dataset deve ser um objeto")
        ratio_sum = sum(ratios[name] for name in ("train_ratio", "val_ratio", "test_ratio"))
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio deve ser 1")

        training = values.get("training")
        augmentation = values.get("augmentation")
        if not isinstance(training, dict) or not isinstance(augmentation, dict):
            raise ValueError("training e augmentation devem ser objetos")
        return cls(
            values["schema_version"],
            values["seed"],
            tuple(shape),
            values["num_classes"],
            ratios,
            training,
            augmentation,
        )


class SolarDataModule:
    def __init__(self, base_path: Path, seed: int = 42, expected_shape: Tuple[int, int] = (40, 24)):
        self.base_path = Path(base_path)
        self.seed = seed
        self.expected_shape = expected_shape
        self.metadata_path = self.base_path / "module_metadata.json"
        self.images_dir = self.base_path / "images"
        self.paths = []
        self.labels = []
        self.keys = []
        self.train_indices = np.array([], dtype=np.int64)
        self.val_indices = np.array([], dtype=np.int64)
        self.test_indices = np.array([], dtype=np.int64)
        self._load_metadata()

    def _load_metadata(self) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadados nao encontrados: {self.metadata_path}")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        for key, record in metadata.items():
            class_name = record.get("anomaly_class")
            if class_name not in CLASS_TO_INDEX:
                continue
            image_path = self.base_path / record["image_filepath"]
            if image_path.exists():
                self.keys.append(key)
                self.paths.append(image_path)
                self.labels.append(CLASS_TO_INDEX[class_name])

        if not self.paths:
            raise ValueError("Nenhuma imagem valida foi localizada")
        self.labels = np.asarray(self.labels, dtype=np.int64)

    def split(self, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
        del test_ratio
        indices = np.arange(len(self.labels))
        train_indices, temp_indices, _, temp_labels = train_test_split(
            indices,
            self.labels,
            train_size=train_ratio,
            stratify=self.labels,
            random_state=self.seed,
        )
        val_fraction = val_ratio / (1.0 - train_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_fraction,
            stratify=temp_labels,
            random_state=self.seed,
        )
        self.train_indices = np.asarray(train_indices, dtype=np.int64)
        self.val_indices = np.asarray(val_indices, dtype=np.int64)
        self.test_indices = np.asarray(test_indices, dtype=np.int64)

    def load_image(self, index: int) -> np.ndarray:
        with Image.open(self.paths[index]) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        if array.shape != self.expected_shape:
            raise ValueError(
                f"Imagem {self.keys[index]} possui formato {array.shape}; esperado {self.expected_shape}"
            )
        return array

    def class_counts(self) -> Dict[str, int]:
        counts = np.bincount(self.labels, minlength=len(CLASSES))
        return {name: int(counts[index]) for index, name in enumerate(CLASSES)}


class SolarDataset(Dataset):
    def __init__(
        self,
        module: SolarDataModule,
        indices: Iterable[int],
        augmentation: Optional[Dict[str, object]] = None,
        seed: int = 42,
    ):
        self.module = module
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.augmentation = augmentation or {"enabled": False}
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[position])
        image = self.module.load_image(index)

        if self.augmentation.get("enabled"):
            if self.rng.random() < float(self.augmentation["horizontal_flip"]):
                image = np.flip(image, axis=1).copy()
            if self.rng.random() < float(self.augmentation["vertical_flip"]):
                image = np.flip(image, axis=0).copy()
            brightness = float(self.augmentation["brightness"])
            contrast = float(self.augmentation["contrast"])
            mean = float(image.mean())
            image = (image - mean) * (1.0 + self.rng.uniform(-contrast, contrast)) + mean
            image = image + self.rng.uniform(-brightness, brightness)
            image = np.clip(image, 0.0, 1.0)

        tensor = torch.from_numpy(image).unsqueeze(0).float()
        label = torch.tensor(int(self.module.labels[index]), dtype=torch.long)
        return tensor, label, index
