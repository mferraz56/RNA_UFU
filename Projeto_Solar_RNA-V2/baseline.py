import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from data import CLASSES, ExperimentConfig, SolarDataModule
from metrics import calculate_metrics, export_json


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = (
    REPOSITORY_ROOT
    / "Dataset"
    / "2020-02-14_InfraredSolarModules"
    / "InfraredSolarModules"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa baseline linear")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "experiment.json")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "resultados")
    parser.add_argument("--max-train-samples", type=int)
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    module = SolarDataModule(args.dataset, seed=config.seed, expected_shape=config.image_shape)
    module.split(
        float(config.dataset["train_ratio"]),
        float(config.dataset["val_ratio"]),
        float(config.dataset["test_ratio"]),
    )
    train_indices = module.train_indices
    if args.max_train_samples is not None:
        train_indices = train_indices[: args.max_train_samples]

    def features(indices):
        return np.asarray([module.load_image(int(index)).ravel() for index in indices], dtype=np.float32)

    model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        random_state=config.seed,
    )
    model.fit(features(train_indices), module.labels[train_indices])
    predictions = model.predict(features(module.test_indices))
    metrics = calculate_metrics(module.labels[module.test_indices], predictions, CLASSES)
    args.output.mkdir(parents=True, exist_ok=True)
    export_json(metrics, args.output / "baseline_metrics.json")
    print(
        f"Baseline - accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
