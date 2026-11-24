import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import CLASSES, ExperimentConfig, SolarDataModule, SolarDataset
from metrics import calculate_metrics, export_json
from model import SimpleSolarCNN


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = (
    REPOSITORY_ROOT
    / "Dataset"
    / "2020-02-14_InfraredSolarModules"
    / "InfraredSolarModules"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, device) -> tuple:
    model.eval()
    labels = []
    predictions = []
    probabilities = []
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    for inputs, targets, _ in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss_total += float(criterion(logits, targets).item()) * len(targets)
        probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
        predictions.extend(logits.argmax(dim=1).cpu().numpy())
        labels.extend(targets.cpu().numpy())
    loss = loss_total / max(1, len(loader.dataset))
    return loss, labels, predictions, probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina a CNN Solar V2")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "experiment.json")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "resultados")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train-samples", type=int)
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    seed_everything(config.seed)
    module = SolarDataModule(args.dataset, seed=config.seed, expected_shape=config.image_shape)
    module.split(
        float(config.dataset["train_ratio"]),
        float(config.dataset["val_ratio"]),
        float(config.dataset["test_ratio"]),
    )

    training = dict(config.training)
    epochs = args.epochs if args.epochs is not None else int(training["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(training["batch_size"])
    max_samples = (
        args.max_train_samples
        if args.max_train_samples is not None
        else training.get("max_train_samples")
    )
    train_indices = module.train_indices
    if max_samples is not None:
        train_indices = train_indices[: int(max_samples)]

    train_dataset = SolarDataset(module, train_indices, config.augmentation, seed=config.seed)
    val_dataset = SolarDataset(module, module.val_indices, {"enabled": False}, seed=config.seed)
    test_dataset = SolarDataset(module, module.test_indices, {"enabled": False}, seed=config.seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(training["num_workers"]))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=int(training["num_workers"]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=int(training["num_workers"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSolarCNN(config.num_classes).to(device)
    counts = np.asarray([module.class_counts()[name] for name in CLASSES], dtype=np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = torch.from_numpy(class_weights / class_weights.mean()).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    history = []
    best_val_loss = float("inf")
    best_state = None
    patience = int(training["early_stopping_patience"])
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(targets)
        train_loss /= max(1, len(train_loader.dataset))
        val_loss, val_labels, val_predictions, _ = evaluate(model, val_loader, device)
        val_metrics = calculate_metrics(val_labels, val_predictions, CLASSES)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })
        print(
            f"Epoca {epoch:03d}/{epochs} - "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping: paciencia {patience} atingida")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_labels, test_predictions, test_probabilities = evaluate(model, test_loader, device)
    test_metrics = calculate_metrics(test_labels, test_predictions, CLASSES)
    test_metrics["loss"] = test_loss

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "cnn_solar_v2.pt"
    torch.save({
        "model_state": model.state_dict(),
        "class_names": CLASSES,
        "seed": config.seed,
        "config": json.loads(args.config.read_text(encoding="utf-8")),
        "splits": {
            "train": len(train_indices),
            "val": len(module.val_indices),
            "test": len(module.test_indices),
        },
    }, checkpoint_path)
    export_json(history, args.output / "history.json")
    export_json(test_metrics, args.output / "test_metrics.json")
    export_json(module.class_counts(), args.output / "class_counts.json")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Teste: accuracy={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
