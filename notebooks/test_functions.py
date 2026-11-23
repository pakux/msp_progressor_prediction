#! /usr/bin/env python

import torch
import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join, abspath
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    auc,
    average_precision_score,
    precision_recall_curve,
)
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test


import matplotlib.pyplot as plt


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


def bootstrap_auc(y_true, y_score, curve="roc", n_bootstraps=1000, seed=42):
    """Calculate AUC with bootstrap confidence intervals"""
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue

        if curve == "roc":
            fpr, tpr, _ = roc_curve(y_true[indices], y_score[indices])
            score = auc(fpr, tpr)
        elif curve == "prc":
            precision, recall, _ = precision_recall_curve(
                y_true[indices], y_score[indices]
            )
            score = auc(recall, precision)

        bootstrapped_scores.append(score)

    lower = np.percentile(bootstrapped_scores, 2.5)
    upper = np.percentile(bootstrapped_scores, 97.5)
    return np.mean(bootstrapped_scores), lower, upper


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

    sens = tp / (tp + fp)

    roc_mean, roc_lower, roc_upper = bootstrap_auc(
        np.array(y_true), np.array(y_score), curve="roc"
    )
    prc_mean, prc_lower, prc_upper = bootstrap_auc(
        np.array(y_true), np.array(y_score), curve="prc"
    )

    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    return {
        "threshold": [threshold_value],
        "tn": [tn],
        "fp": [fp],
        "fn": [fn],
        "tp": [tp],
        "accuracy": [acc],
        "precision": [prec],
        "recall": [rec],
        "auroc": [f"{roc_mean:0.2f} [{roc_lower:0.2f} - {roc_upper:0.2f}]"],
        "auprc": [f"{prc_mean:0.2f} [{prc_lower:0.2f} - {prc_upper:0.2f}]"],
        "sensitivity": [f"{sens:0.3f}"],
        "specifity": [f"{spec:0.3f}"],
        "f1": [f1],
        "balanced_accuracy": [bacc],
        "ppv": [f"{ppv:0.3f}"],
        "npv": [f"{npv:0.3f}"],
    }


def youden_thresholds(y_true, y_score):
    """
    Find optimal thresholds using multiple methods

    Returns:
    --------
    dict with all threshold methods and their key metrics
    """
    # Method 1: Youden's Index (maximizes sensitivity + specificity - 1)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_index = tpr - fpr
    youden_idx = np.argmax(youden_index)
    youden_threshold = thresholds[youden_idx]
    youden_sensitivity = tpr[youden_idx]
    youden_specificity = 1 - fpr[youden_idx]

    # Method 2: Closest to Top-Left (minimizes distance to (0,1))
    distances = np.sqrt((1 - tpr) ** 2 + fpr**2)
    topleft_idx = np.argmin(distances)
    topleft_threshold = thresholds[topleft_idx]
    topleft_sensitivity = tpr[topleft_idx]
    topleft_specificity = 1 - fpr[topleft_idx]

    # Method 3: Balanced Accuracy (maximizes (sensitivity + specificity) / 2)
    balanced_acc = (tpr + (1 - fpr)) / 2
    balanced_idx = np.argmax(balanced_acc)
    balanced_threshold = thresholds[balanced_idx]
    balanced_sensitivity = tpr[balanced_idx]
    balanced_specificity = 1 - fpr[balanced_idx]

    # Method 4: F1 Score

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = np.zeros(len(precision))
    for i in range(len(precision)):
        if precision[i] + recall[i] > 0:
            f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    f1_idx = np.argmax(f1_scores)
    f1_threshold = pr_thresholds[f1_idx] if f1_idx < len(pr_thresholds) else 1.0
    f1_precision = precision[f1_idx]
    f1_recall = recall[f1_idx]

    return {
        "youden_threshold": youden_threshold,
        "youden_sensitivity": youden_sensitivity,
        "youden_specificity": youden_specificity,
        "youden_index": youden_index[youden_idx],
        "topleft_threshold": topleft_threshold,
        "topleft_sensitivity": topleft_sensitivity,
        "topleft_specificity": topleft_specificity,
        "balanced_threshold": balanced_threshold,
        "balanced_sensitivity": balanced_sensitivity,
        "balanced_specificity": balanced_specificity,
        "balanced_accuracy": balanced_acc[balanced_idx],
        "f1_threshold": f1_threshold,
        "f1_precision": f1_precision,
        "f1_recall": f1_recall,
        "f1_score": f1_scores[f1_idx],
    }


def plot_kaplan_meier2(
    time_to_event,
    event_observed,
    prediction_scores,
    test_cohort,
    threshold,
    save_path=None,
    ax=None,
):
    """
    Plot Kaplan-Meier curve stratified by DL model predictions with Hazard Ratio.

    Parameters:
    -----------
    time_to_event : array-like
        Time until event or censoring (in months)
    event_observed : array-like
        Binary labels (0: not progressing/censored, 1: progressing/event)
    prediction_scores : array-like
        DL model prediction scores (probabilities)
    test_cohort : str
        Name of test cohort for plot title
    threshold : float
        Threshold to stratify high-risk vs low-risk groups
    save_path : str
        Path to save the figure
    """

    # Create DataFrame
    df = pd.DataFrame(
        {
            "time": time_to_event,
            "event": event_observed,
            "risk_score": prediction_scores,
        }
    )

    # Stratify by model predictions
    df["risk_group"] = (df["risk_score"] >= threshold).astype(int)

    print(df)
    # Initialize Kaplan-Meier fitter
    kmf = KaplanMeierFitter()

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # Plot KM curves for each risk group
    colors = ["#1671bc", "#c11b0f"]  # Blue for low risk, red for high risk

    for idx, group in enumerate([0, 1]):
        mask = df["risk_group"] == group
        label = (
            f"Low Risk (n={mask.sum()})"
            if group == 0
            else f"High Risk (n={mask.sum()})"
        )

        kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"], label=label)

        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=colors[idx],
            linewidth=2.5,
            alpha=0.5,
        )

    # Perform log-rank test
    low_risk = df[df["risk_group"] == 0]
    high_risk = df[df["risk_group"] == 1]

    results = logrank_test(
        low_risk["time"],
        high_risk["time"],
        low_risk["event"],
        high_risk["event"],
    )

    # --- Add Hazard Ratio calculation using Cox Proportional Hazards Model ---
    from lifelines import CoxPHFitter

    df_cox = df[["time", "event", "risk_group"]].copy()
    df_cox.columns = ["T", "E", "risk_group"]

    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col="T", event_col="E", strata=None)

    hr = cph.hazard_ratios_.iloc[0]
    hr_ci = cph.confidence_intervals_.iloc[0].to_list()

    # Prepare text for the plot
    textstr = (
        f"Log-rank test:\n"
        f"p = {results.p_value:.4f}\n"
        f"χ² = {results.test_statistic:.2f}\n\n"
        f"Hazard Ratio (HR):\n"
        f"HR = {hr:.2f} "
        f"[{hr_ci[0]:.2f} – {hr_ci[1]:.2f}]"
    )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.02,
        0.02,
        textstr,
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment="bottom",
        bbox=props,
    )

    # Add labels and title
    ax.set_xlabel("Time (days)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Progression-Free Survival", fontsize=14, fontweight="bold")
    ax.set_title(
        f"{test_cohort}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        if len(dirname(save_path)) > 0:
            makedirs(dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Kaplan-Meier curve saved to {save_path}")

    # Return metrics including HR
    km_metrics = {
        "threshold": threshold,
        "n_low_risk": int((df["risk_group"] == 0).sum()),
        "n_high_risk": int((df["risk_group"] == 1).sum()),
        "events_low_risk": int(low_risk["event"].sum()),
        "events_high_risk": int(high_risk["event"].sum()),
        "logrank_p_value": results.p_value,
        "logrank_chi2": results.test_statistic,
        "hazard_ratio": hr,
        "hazard_ratio_95ci_lower": hr_ci[0],
        "hazard_ratio_95ci_upper": hr_ci[1],
    }

    return km_metrics, ax


def find_optimal_thresholds(y_true, y_score):
    """
    Find optimal thresholds using multiple methods

    Returns:
    --------
    dict with all threshold methods and their key metrics
    """
    # Method 1: Youden's Index (maximizes sensitivity + specificity - 1)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_index = tpr - fpr
    youden_idx = np.argmax(youden_index)
    youden_threshold = thresholds[youden_idx]
    youden_sensitivity = tpr[youden_idx]
    youden_specificity = 1 - fpr[youden_idx]

    # Method 2: Closest to Top-Left (minimizes distance to (0,1))
    distances = np.sqrt((1 - tpr) ** 2 + fpr**2)
    topleft_idx = np.argmin(distances)
    topleft_threshold = thresholds[topleft_idx]
    topleft_sensitivity = tpr[topleft_idx]
    topleft_specificity = 1 - fpr[topleft_idx]

    # Method 3: Balanced Accuracy (maximizes (sensitivity + specificity) / 2)
    balanced_acc = (tpr + (1 - fpr)) / 2
    balanced_idx = np.argmax(balanced_acc)
    balanced_threshold = thresholds[balanced_idx]
    balanced_sensitivity = tpr[balanced_idx]
    balanced_specificity = 1 - fpr[balanced_idx]

    # Method 4: F1 Score
    from sklearn.metrics import precision_recall_curve

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = np.zeros(len(precision))
    for i in range(len(precision)):
        if precision[i] + recall[i] > 0:
            f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    f1_idx = np.argmax(f1_scores)
    f1_threshold = pr_thresholds[f1_idx] if f1_idx < len(pr_thresholds) else 1.0
    f1_precision = precision[f1_idx]
    f1_recall = recall[f1_idx]

    return {
        "youden_threshold": youden_threshold,
        "youden_sensitivity": youden_sensitivity,
        "youden_specificity": youden_specificity,
        "youden_index": youden_index[youden_idx],
        "topleft_threshold": topleft_threshold,
        "topleft_sensitivity": topleft_sensitivity,
        "topleft_specificity": topleft_specificity,
        "balanced_threshold": balanced_threshold,
        "balanced_sensitivity": balanced_sensitivity,
        "balanced_specificity": balanced_specificity,
        "balanced_accuracy": balanced_acc[balanced_idx],
        "f1_threshold": f1_threshold,
        "f1_precision": f1_precision,
        "f1_recall": f1_recall,
        "f1_score": f1_scores[f1_idx],
    }


def kmplotter_multimodal(
    dataframes: dict,
    event_dfs: pd.DataFrame,
    functional_test=None,
    testcol="name",
    ax=None,
    modcolors=None,
):
    """
    Plot Kaplan Meiers from combined dataframes:

    Three groups will be defined:
        High risk: high risk in all dataframes
        Low risk: low risk in all dataframes
        intermediate: low and high in different dataframes

        event_dfs containst dict of event_Dataframest for each of the functional tests

    Parameters
    ----------
    dataframes: dict of dataframes
        key: name of modality
    event_dfs: dict of pd.DataFrame
        dataframe that contains columns of time to event, for each testname
    functional_test: str or list of str
        name the test PST, WST,... or give a list

    ax=None

    """
    # If no functional test is given: use all available frem first dataframe
    if functional_test is None:
        functional_test = (
            dataframes[list(dataframes.keys())[0]][testcol].unique().tolist()
        )

    if modcolors is None:
        modcolors = {
            key: color for key, color in zip(dataframes.keys(), sns.color_palette())
        }

    # in case of only a single functional_test is given as string: put it into an list
    if type(functional_test) is str:
        functional_test = [functional_test]

    if ax is None:
        fig, ax = plt.subplots(len(functional_test), 2, figsize=(18, 24))

    for axrow, testname in enumerate(functional_test):
        print(testname)

        ### Calculate thresholds in all dataframes seperately
        multimodal_df = pd.DataFrame()
        for modality, modality_df in dataframes.items():
            df = modality_df.query(f'{testcol} == "{testname}"')
            threshold = find_optimal_thresholds(
                df["y_test"].values, df["y_score"].values
            )["youden_threshold"]
            df["risk_group"] = (df["y_score"] >= threshold).astype(int)
            multimodal_df = pd.concat((multimodal_df, df), ignore_index=True)

        multimodal_df = pd.DataFrame(
            multimodal_df.groupby("eid").agg({"risk_group": "sum"}).reset_index()
        )

        intermediate_ids = multimodal_df.query("risk_group == 1").eid.tolist()

        # return multimodal_df
        event_df = event_dfs[testname][
            [
                "eid",
                f"time_{testname}".lower(),
                f"worst_progressor_2ycutoff_{testname}_2z".lower(),
            ]
        ]
        event_df.columns = ["eid", "time", "event"]
        df = pd.merge(
            left=event_df,
            right=multimodal_df,
            left_on="eid",
            right_on="eid",
        )

        print(f"current test: {testname}")
        df.dropna(inplace=True)

        colors = ["#1671bc", "#D4AF37", "#c11b0f"]
        kmf = KaplanMeierFitter()

        for idx, group in enumerate([0, 1, 2]):
            mask = df["risk_group"] == group
            if group == 0:
                label = f"Low Risk (n={mask.sum()})"
            elif group == 1:
                label = f"Intermediate Risk (n={mask.sum()})"
            elif group == 2:
                label = f"High Risk (n={mask.sum()})"
            else:
                label = f"n = {mask.sum()}"

            print(f"fitting for group {label}")

            kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"], label=label)
            kmf.plot_survival_function(
                ax=ax[axrow, 0],
                ci_show=True,
                color=colors[idx],
                linewidth=2.5,
                alpha=0.5,
            )

            # Add labels and title
            ax[axrow, 0].set_ylabel(f"{testname}", fontsize=18, fontweight="bold")
            ax[axrow, 0].set_xlabel("")
            ax[axrow, 0].set_ylim(ymin=0)

        ### Add irtermediate  plots seperately i.e. flair and t1w

        multimodal_df = pd.DataFrame()
        for modality, modality_df in dataframes.items():
            df = modality_df.query(f'{testcol} == "{testname}"')
            threshold = find_optimal_thresholds(
                df["y_test"].values, df["y_score"].values
            )["youden_threshold"]

            # restrict to those eids that are in intermediate risk group
            df = df.query("eid in @intermediate_ids")
            df["risk_group"] = pd.NA
            # df["risk_group"] = (df["y_score"] >= threshold).astype(int)
            df.loc[df["y_score"] >= threshold, "risk_group"] = modality
            multimodal_df = pd.concat((multimodal_df, df))

        df = pd.merge(
            left=event_df.query("eid in @intermediate_ids"),
            right=multimodal_df,
            left_on="eid",
            right_on="eid",
        )
        df.dropna(inplace=True)
        kmf = KaplanMeierFitter()

        print(df["risk_group"])

        for group, dataframe in dataframes.items():
            mask = df["risk_group"] == group
            kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"], label=group)
            kmf.plot_survival_function(
                ax=ax[axrow, 1],
                ci_show=True,
                color=modcolors[group],
                linewidth=2.5,
                alpha=0.5,
            )
            ax[axrow, 1].set_xlabel("")
            ax[axrow, 1].set_ylim(ymin=0)

    ax[axrow, 1].set_xlabel("Time (days)", fontsize=14, fontweight="bold")
    ax[axrow, 0].set_xlabel("Time (days)", fontsize=14, fontweight="bold")
    return fig, ax
