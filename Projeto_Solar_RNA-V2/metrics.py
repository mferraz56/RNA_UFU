import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def calculate_metrics(
    labels: Iterable[int],
    predictions: Iterable[int],
    class_names: Iterable[str],
) -> Dict[str, object]:
    y_true = np.asarray(list(labels), dtype=np.int64)
    y_pred = np.asarray(list(predictions), dtype=np.int64)
    names = list(class_names)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(names))),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(names)))).tolist(),
        "classification_report": report,
    }


def export_json(data: Dict[str, object], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
