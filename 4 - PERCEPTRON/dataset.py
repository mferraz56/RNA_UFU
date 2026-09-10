from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass
class ReferenceDataset:
	train_images: np.ndarray
	test_images: np.ndarray
	train_labels: np.ndarray
	test_labels: np.ndarray
	train_indices: np.ndarray
	test_indices: np.ndarray
	description: str


def load_reference_dataset(seed: int = 42) -> ReferenceDataset:
	"""Load the bundled UCI digits data; keep the test partition out of training."""
	digits = load_digits()
	train_indices, test_indices = train_test_split(
		np.arange(len(digits.target)), test_size=0.25,
		random_state=seed, stratify=digits.target,
	)
	images = digits.data.astype(float) / 16.0
	return ReferenceDataset(
		images[train_indices], images[test_indices],
		digits.target[train_indices], digits.target[test_indices],
		train_indices, test_indices, digits.DESCR,
	)
