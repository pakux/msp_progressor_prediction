#! /usr/bin/env python

import torch
import numpy as np
from os.path import join, abspath
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)


def run_test(
    column_name,
    data_dir,
    test_dataset,
    modality,
    modelname="sfcn",
    tensor_dir_dataset="mspaths",
):
    device = torch.device(
        conf["cuda"]["device"] if torch.cuda.is_available() else "cpu"
    )
    test_dataset = dataloader.BrainDataset(
        csv_file=abspath(join(data_dir, test_dataset, f"{column_name}.csv")),
        root_dir=abspath(
            join(tensor_dir_test, tensor_dir_dataset, f"{modality}96_affine")
        ),
        column_name=column_name,
        num_rows=None,
        num_classes=2,
        task="classification",
    )

    test_loader = DataLoader(
        test_dataset, batch_size=32, num_workers=8, drop_last=False
    )

    # Load the model and accordingly the saved state
    model = sfcn_cls.SFCN(output_dim=2).to(device)
    checkpoint = torch.load(
        join(
            models_dir,
            modelname,
            modality,
            f"{column_name}_e1000_b16_im96.pth",
        ),
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    test_outputs_binary = []
    test_labels = []
    test_eids = []

    with torch.no_grad():
        for eid, images, labels in mo.status.progress_bar(test_loader):
            test_eids.extend(eid)
            images = images.to(device)
            labels = labels.float().to(device)
            binary_labels = labels[:, 1]
            test_labels.extend(binary_labels.tolist())

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            binary_outputs = probs[:, 1]
            test_outputs_binary.extend(binary_outputs.tolist())
    eids = np.array(test_eids).astype(int)
    y_true = np.array(test_labels).astype(int)
    y_score = np.array(test_outputs_binary).astype(float)

    return eids, y_true, y_score


def performance_params(y_true, y_score, threshold="youden"):
    """
    Calculate performance parameters for given y_true /y_score table

    """
    if threshold == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        threshold_value = thresholds[np.argmax(tpr - fpr)]
    elif isinstance(threshold, (int, float)):
        threshold_value = threshold
    else:
        threshold_value = 0.5

    y_pred = (y_score >= threshold_value).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / cm.sum()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)

    return {
        "threshold": [threshold_value],
        "tn": [tn],
        "fp": [fp],
        "fn": [fn],
        "tp": [tp],
        "accuracy": [acc],
        "precision": [prec],
        "recall": [rec],
        "specifity": [spec],
        "f1": [f1],
        "balanced_accuracy": [bacc],
    }
