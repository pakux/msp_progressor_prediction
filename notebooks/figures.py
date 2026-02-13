import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full", app_title="Figures and plots")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup

    First do all the administrative stuff as import libs, set paths, etc.

    You might need to adapt some path- /filenames to your own setup.

    `braindraindir` is used to locate the BrainTrain repository. Some files from this repo will be imported. So please make sure its correct.
    """)
    return


@app.cell
def _(mo):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import sys
    from os.path import join, abspath, dirname
    from os import makedirs
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from pathlib import Path
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    from lifelines.plotting import add_at_risk_counts

    # Define Paths and Filenames for further work / from previous work with BrainTrain
    # braindraindir = "../../../RadBrainDL_msp/code/BrainTrain/"  # source path f BrainTrain 🧠🚆
    #                                                             # will be used to load modules
    braindraindir = "/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/code/BrainTrain/"

    # data_dir = "../../../RadBrainDL_msp/data/"
    data_dir = "/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/data/"
    models_dir = "models"
    # tensor_dir_test = "../../../RadBrainDL_msp/images/"
    tensor_dir_test = "/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/images"

    sys.path.append(braindraindir)
    try:
        from dataloaders import dataloader
    except ModuleNotFoundError:
        mo.md("Could not load Braintrain! This might break things").callout(
            kind="danger"
        )

    try:
        from architectures import sfcn_cls
    except ModuleNotFoundError:
        mo.md("Could not load SFCN module! This might break things.").callout(
            kind="danger"
        )


    columns = [
        "worst_progression_pst_2z",
        "worst_progression_mdt_2z",
        "worst_progression_cst_2z",
        "worst_progression_wst_2z",
    ]

    sns.set_style("whitegrid")
    sns.set_context("talk")
    return (
        DataLoader,
        F,
        KaplanMeierFitter,
        Path,
        abspath,
        auc,
        columns,
        data_dir,
        dataloader,
        dirname,
        join,
        logrank_test,
        makedirs,
        models_dir,
        np,
        pd,
        plt,
        precision_recall_curve,
        roc_curve,
        sfcn_cls,
        sns,
        tensor_dir_test,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Tests and Performance
    Perform tests on the dataset. This is only needed once. Remove `outputs.csv` in case you want to re-test. Beware in this case you need more RAM.
    """)
    return


@app.cell
def _(
    DataLoader,
    F,
    abspath,
    auc,
    dataloader,
    join,
    mo,
    models_dir,
    np,
    pd,
    plt,
    precision_recall_curve,
    roc_curve,
    sfcn_cls,
    sns,
    tensor_dir_test,
    torch,
):
    def bootstrap_auc(y_true, y_score, curve="roc", n_bootstraps=1000, seed=42):
        """Calculate AUC with bootstrap confidence intervals"""
        rng = np.random.RandomState(seed)
        bootstrapped_scores = []

        for _ in range(n_bootstraps):
            indices = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true)) < 2:
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


    def plot_roc_curve(df, y_true="y_test", y_score="y_score", dataset="name"):
        """
        Plot auroc curve for a dataframe
        """
        data_names = df[
            dataset
        ].unique()  # retrieve different dataset-names from df (dataset-column defaults to "name")
        f = plt.figure(figsize=(10, 8))

        for data_name in data_names:
            subset = df[df[dataset] == data_name]
            y_true_array = np.array(subset[y_true].to_list())
            y_score_array = np.array(subset[y_score].to_list())

            fpr, tpr, _ = roc_curve(subset[y_true], subset[y_score])
            roc_auc = auc(fpr, tpr)
            roc_mean, roc_lower, roc_upper = bootstrap_auc(
                y_true_array, y_score_array, curve="roc"
            )
            ax = sns.lineplot(
                x=fpr, y=tpr, label=f"{data_name} (AUC = {roc_auc:.2f})"
            )

        sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--")
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1.05))
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operationg Characteristic (ROC) Curves")

        # plt.show()

        return ax


    def plot_prc_curve(df, y_true="y_test", y_score="y_score", dataset="name"):
        """Plot Precision-Recall curve with confidence intervals"""
        data_names = df[
            dataset
        ].unique()  # retrieve different dataset-names from df (dataset-column defaults to "name")
        f = plt.figure(figsize=(10, 8))
        for data_name in data_names:
            subset = df[df[dataset] == data_name]
            y_true_array = np.array(subset[y_true].to_list())
            y_score_array = np.array(subset[y_score].to_list())
            precision, recall, _ = precision_recall_curve(
                y_true_array, y_score_array
            )
            prc_auc = auc(recall, precision)
            prc_mean, prc_lower, prc_upper = bootstrap_auc(
                y_true_array, y_score_array, curve="prc"
            )
            pos_rate = y_true_array.mean()

            ax = sns.lineplot(
                x=recall,
                y=precision,
                lw=2,
                label=f"{data_name} (AUC = {prc_auc:.2f} [{prc_lower:.2f}–{prc_upper:.2f}])",
            )

        plt.hlines(
            pos_rate,
            0,
            1,
            colors="gray",
            linestyles="--",
            label=f"Baseline = {pos_rate:.3f}",
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PRC Curve ")

        return ax


    def run_test(column_name, data_dir, test_dataset, modality):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test_dataset = dataloader.BrainDataset(
            csv_file=abspath(
                join(data_dir, test_dataset, "test", f"{column_name}.csv")
            ),
            root_dir=abspath(join(tensor_dir_test, "mspaths2", modality)),
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
            join(models_dir, modality, f"{column_name}_e1000_b32_im96.pth"),
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


    df = pd.DataFrame()
    return plot_prc_curve, plot_roc_curve, run_test


@app.cell
def _(pd):
    df_dist = pd.read_csv("../../../data/neuro_progressors_distance.csv")
    df_dist.groupby("mpi").agg({"progressor_pst": "sum"}) > 0
    return (df_dist,)


@app.cell
def _(df_dist):
    df_dist.groupby("mpi").agg({"progressor_pst": "sum"}) > 0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Demographics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 1: Study Design
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data distribution (Histograms for PST, WST, MDT, CST)

    Here we could include Barplots (maybe stacked) for each of the Progressors/NonProgressors
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Image Processing

    Preprocessing Pipeline
    """)
    return


@app.cell(hide_code=True)
def preprocessing_pipeline(mo):
    pipeline = mo.mermaid("""
    graph LR

      A[T1w Image] --> C
      A --> N4Bias(N4BiasfieldCorrection)
      N4Bias --> BET(brain extraction)
      B[FLAIR Image; below=A] -->|register to T1w| C[Registered FLAIR]
      A --> MASK(skullstripped T1w) 
      BET -->|brainmask| MASK
      BET-->|brainmask| BETFLAIR[skullstripped FLAIR]
      C --> BETFLAIR
      MASK-->|register| D[MNI 152 standard space]
      D -->|coregister| F>FLAIR in MNI 152 space]
      D --> E>T1w brain in MNI 152 space]
      BETFLAIR --> F
      F -->|crop and resize to 96×96×96 voxel| FLAIR
      E -->|crop and resize to 96×96×96 voxel| T1w

    """)
    pipeline
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Modelling and Evaluation
    """)
    return


@app.cell
def _(mo):
    diagram = """
    flowchart TD
      A["3D T1-weighted MRI] --> B[Preprocessing"]
      B --> B1{"Steps"}
      B1 --> B1a["N4 bias field correction"]
      B1 --> B1b["Skull-stripping"]
      B1 --> B1c["Affine + nonlinear registration to template"]
      B1 --> B1d["Intensity normalization (z-score)"]
      B1 --> B1e["Resample to fixed voxel spacing & crop/pad to ROI"]
      B --> C["Data augmentation (train only)"]
      C --> C1{Augmentations}
      C1 --> C1a[Random affine/elastic]
      C1 --> C1b[Random intensity scaling]
      C1 --> C1c[Random flips/crops]
      C1 --> C1d[Gaussian noise]

      C --> D[Backbone: Foundation 3D Encoder]
      D --> D1[Pretrained on large 3D brain MRI corpus]
      D --> D2[Architecture: 3D ViT / 3D Swin Transformer or 3D CNN]
      D --> D3[Output: Global feature vector]

      D --> E["Clinical embedding (optional)"]
      E --> E1[Age, sex, disease duration, baseline PST/Dex scores]
      E --> F[Concatenate features]
      F --> G[Task heads]
      G --> G1["Progression classifier (binary): Worsened >=2 z-scores"]
      G --> G2["Regression head: predicted ΔPST z-score"]
      G --> G3["Regression head: predicted ΔDex z-score"]
      G --> G4["Uncertainty head: aleatoric + epistemic"]

      G1 --> H[Losses]
      G2 --> H
      G3 --> H
      G4 --> H
      H --> H1{Combined loss}
      H1 --> H1a["Binary cross-entropy (classifier)"]
      H1 --> H1b["MSE or Huber (regressions)"]
      H1 --> H1c["KL / MC-dropout loss (uncertainty)""]
      H1 --> H1d["Class-balancing / focal loss if needed"]

      H --> I[Training loop]
      I --> I1[Fine-tune foundation encoder + heads]
      I1 --> I2[Validation: AUROC, AUPRC, sensitivity at fixed specificity]
      I1 --> I3[Calibration: reliability plots, expected calibration error]

      I --> J["Explainability & QC"]
      J --> J1["Saliency / Grad-CAM (3D)"]
      J --> J2["SHAP on clinical + global features"]
      J --> J3["Overlay predicted risk on MRI slices"]

      J --> K[Deployment]
      K --> K1["Input: single 3D T1 -> Preproc -> Model"]
      K --> K2["Output: risk probability, predicted Δz-scores, uncertainty"]
      K --> K3["Integration: clinical dashboard / decision support"]
    """

    mo.mermaid(diagram=diagram)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 2: Classification Performance (SFCN)
    """)
    return


@app.cell(hide_code=True)
def _(Path, columns, data_dir, pd, run_test):
    _outfile = Path(f"output_t1w.csv")
    if _outfile.exists():
        df_t1w = pd.read_csv(_outfile)
    else:
        df_t1w = pd.DataFrame()

        for _column_name in columns:
            # run_test should create and return y_test, y_score or write output.csv
            _eids, _y_test, _y_score = run_test(
                _column_name, data_dir, "mspaths2/t1w", "t1w"
            )
            # Save to CSV (using pandas for header and robust types)
            _df_current = pd.DataFrame(
                {
                    "eid": _eids,
                    "y_test": _y_test,
                    "y_score": _y_score,
                    "name": _column_name,
                }
            )

            df_t1w = pd.concat((df_t1w, _df_current), ignore_index=True)
            df_t1w.to_csv(_outfile, index=False)


    # Rename Entries to human readable format
    df_t1w.loc[df_t1w.name.str.contains("_pst"), "name"] = "PST"
    df_t1w.loc[df_t1w.name.str.contains("_cst"), "name"] = "CST"
    df_t1w.loc[df_t1w.name.str.contains("_wst"), "name"] = "WST"
    df_t1w.loc[df_t1w.name.str.contains("_mdt"), "name"] = "MDT"
    return (df_t1w,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## T1w - AUROCs (all 4 tasks)
    """)
    return


@app.cell(hide_code=True)
def _(df_t1w, plot_roc_curve, plt):
    # Create Auroc Curves
    plot_roc_curve(df_t1w)
    plt.savefig(f"auroc_t1w_worst_progression_2z.svg")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## T1w - AUPRC (all 4 tasks)
    """)
    return


@app.cell(hide_code=True)
def _(df_t1w, plot_prc_curve, plt):
    # Create PRC Curves
    plot_prc_curve(df_t1w)
    plt.savefig(f"prc_t1w_worst_progression_2z.svg")
    plt.show()
    return


@app.cell(hide_code=True)
def _(Path, columns, data_dir, pd, run_test):
    _outfile = Path(f"output_flair.csv")
    if _outfile.exists():
        df_flair = pd.read_csv(_outfile)
    else:
        df_flair = pd.DataFrame()

        for _column_name in columns:
            # run_test should create and return y_test, y_score or write output.csv
            _eids, _y_test, _y_score = run_test(
                _column_name, data_dir, "mspaths2/t1w", "flair"
            )
            # Save to CSV (using pandas for header and robust types)
            _df_current = pd.DataFrame(
                {
                    "eid": _eids,
                    "y_test": _y_test,
                    "y_score": _y_score,
                    "name": _column_name,
                }
            )

            df_flair = pd.concat((df_flair, _df_current), ignore_index=True)
            df_flair.to_csv(_outfile, index=False)


    # Rename Entries to human readable format
    df_flair.loc[df_flair.name.str.contains("_pst"), "name"] = "PST"
    df_flair.loc[df_flair.name.str.contains("_cst"), "name"] = "CST"
    df_flair.loc[df_flair.name.str.contains("_wst"), "name"] = "WST"
    df_flair.loc[df_flair.name.str.contains("_mdt"), "name"] = "MDT"
    return (df_flair,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FLAIR -  AUROCs (all 4 tasks)
    """)
    return


@app.cell(hide_code=True)
def _(df_flair, plot_roc_curve, plt):
    # Create Auroc Curves
    plot_roc_curve(df_flair)
    plt.savefig(f"auroc_flair_worst_progression_2z.svg")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FLAIR - AUPRC (all 4 task
    """)
    return


@app.cell(hide_code=True)
def _(df_flair, plot_prc_curve, plt):
    # Create PRC Curves
    plot_prc_curve(df_flair)
    plt.savefig(f"prc_flair_worst_progression_2z.svg")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 3: Progression Curves

    ## Kaplan Meier Curves for T1w
    """)
    return


@app.cell(hide_code=True)
def _(
    KaplanMeierFitter,
    columns,
    data_dir,
    dirname,
    join,
    logrank_test,
    makedirs,
    np,
    pd,
    plt,
    roc_curve,
):
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




    def plot_kaplan_meier(
        time_to_event,
        event_observed,
        prediction_scores,
        test_cohort,
        threshold,
        save_path=None,
    ):
        """
        Plot Kaplan-Meier curve stratified by DL model predictions

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

        # Initialize Kaplan-Meier fitter
        kmf = KaplanMeierFitter()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot KM curves for each risk group
        colors = ["#2ecc71", "#F39C12"]  # Green for low risk, red for high risk

        for idx, group in enumerate([0, 1]):
            mask = df["risk_group"] == group
            label = (
                f"Low Risk (n={mask.sum()})"
                if group == 0
                else f"High Risk (n={mask.sum()})"
            )

            kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"], label=label)

            kmf.plot_survival_function(
                ax=ax, ci_show=True, color=colors[idx], linewidth=2.5, alpha=0.5
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

        # Add labels and title
        ax.set_xlabel("Time (days)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Progression-Free survival", fontsize=14, fontweight="bold")
        ax.set_title(
            f"Kaplan-Meier Curve  on {test_cohort}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Add log-rank test results
        p_value = results.p_value
        test_stat = results.test_statistic

        textstr = f"Log-rank test:\np = {p_value:.4f}\nχ² = {test_stat:.2f}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.02,
            0.02,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            bbox=props,
        )

        ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            if len(dirname(save_path)) > 0:
                makedirs(dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            print(f"Kaplan-Meier curve saved to {save_path}")
    

        # Return metrics
        km_metrics = {
            "threshold": threshold,
            "n_low_risk": int((df["risk_group"] == 0).sum()),
            "n_high_risk": int((df["risk_group"] == 1).sum()),
            "events_low_risk": int(low_risk["event"].sum()),
            "events_high_risk": int(high_risk["event"].sum()),
            "logrank_p_value": p_value,
            "logrank_chi2": test_stat,
        }

        return km_metrics

    def kmplots(df, name):
        col_mapping = {"_pst": "PST", "_cst": "CST", "_wst": "WST", "_mdt": "MDT"}
        for _column in columns:
    
            _data_df = pd.read_csv(join(data_dir, "mspaths2", "t1w", "test", f"{_column}.csv"))
    
            shortname  = next((v for k, v in col_mapping.items() if k in _column), None)
    
            km_data = _data_df.merge(df.query(f'name == "{shortname}"'))
            km_data.time.fillna(0, inplace=True)
            thresholds_dict = find_optimal_thresholds(km_data['y_test'].values, km_data['y_score'].values)
            time_to_event = km_data["time"].values
            event_observed = km_data["y_test"].values
            prediction_scores = km_data["y_score"].values
            km_threshold = thresholds_dict["youden_threshold"]
            km_path = join(f"{shortname}_{name}.svg")
    
            km_metrics = plot_kaplan_meier(
                        time_to_event,
                        event_observed,
                        prediction_scores,
                        f"{shortname} - {name}",
                        threshold=km_threshold,
                        save_path=km_path,
                    )
    
            plt.show()


    return (kmplots,)


@app.cell
def _():
    return


@app.cell
def _(df_t1w, kmplots):
    kmplots(df_t1w, 'T1w')

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Kaplan Meier Curves for FLAIR
    """)
    return


@app.cell
def _(df_flair, kmplots):
    kmplots(df_flair, 'FLAIR')

    return


@app.cell
def _(df_t1w):
    df_t1w
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 4: Heatmaps
    """)
    return


if __name__ == "__main__":
    app.run()
