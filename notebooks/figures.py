import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="Figures and plots")


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
def setup_1(mo):
    import re
    import sys
    from glob import glob
    from math import pi
    from os import makedirs
    from os.path import abspath, basename, dirname, join
    from pathlib import Path

    import cmcrameri
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import spiderplot as spidy
    import torch
    import torch.nn.functional as F
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts
    from lifelines.statistics import logrank_test
    from matplotlib.colors import ListedColormap
    from matplotlib.gridspec import GridSpec
    from nilearn import image
    from nilearn.plotting import plot_anat, plot_img, plot_roi, plot_stat_map
    from scipy.stats import ks_2samp
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
    from torch.utils.data import DataLoader

    # Define Paths and Filenames for further work / from previous work with BrainTrain
    # braindraindir = "../../../RadBrainDL_msp/code/BrainTrain/"  # source path f BrainTrain 🧠🚆
    #                                                             # will be used to load modules
    braindraindir = (
        "/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/code/BrainTrain_Katherlab/"
    )
    # braindraindir = "mnt/radbrain_dl/code/BrainTrain"
    patientstable = "baseline_characteristics.csv"
    # patientstable = "../../../RadBrainDL_msp/baseline_characteristics.csv"
    # data_dir = "../../../RadBrainDL_msp/data/"
    data_dir = "/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/data/"
    # data_dir = "/mnt/radbrain_dl/data/"
    models_dir = "models"
    models_dir = "/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/models"
    scores_dir = "/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/scores/"
    explainability_dir = "/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/explainability"
    # tensor_dir_test = "../../../RadBrainDL_msp/images/"
    tensor_dir_test = "/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/images"
    # tensor_dir_test = "/mnt/radbrain_dl/images/"
    evaluations_dir = "/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/evaluations/"
    sys.path.append(braindraindir)
    try:
        from utils.architectures import sfcn_cls
        from utils.dataloaders import dataloader
    except ModuleNotFoundError:
        mo.md("Could not load Braintrain! This might break things").callout(
            kind="danger"
        )

    # try:
    #    from architectures import sfcn_cls
    # except ModuleNotFoundError:
    #    mo.md("Could not load SFCN module! This might break things.").callout(
    #        kind="danger"
    #    )

    columns = [
        "worst_progressor_2ycutoff_pst_2z",
        "worst_progressor_2ycutoff_wst_2z",
        "worst_progressor_2ycutoff_cst_2z",
        "worst_progressor_2ycutoff_mdt_2z",
    ]

    test_order = ["PST", "MDT", "CST", "WST"]

    palette = sns.color_palette("Set2", 3)
    cmap = ListedColormap(palette)

    sns.set_palette("tab10")
    sns.set_style("whitegrid")
    sns.set_context("notebook")

    dataset_order = ["training", "validation", "test"]



    color_female = "#008080"
    color_male = "#FFA500"
    return (
        DataLoader,
        F,
        KaplanMeierFitter,
        Path,
        abspath,
        auc,
        cm,
        cmap,
        color_female,
        color_male,
        columns,
        data_dir,
        dataloader,
        dataset_order,
        dirname,
        evaluations_dir,
        explainability_dir,
        glob,
        join,
        ks_2samp,
        logrank_test,
        makedirs,
        mcolors,
        models_dir,
        np,
        patientstable,
        pd,
        plot_stat_map,
        plt,
        precision_recall_curve,
        roc_curve,
        scores_dir,
        sfcn_cls,
        sns,
        spidy,
        tensor_dir_test,
        test_order,
        torch,
    )


@app.cell
def _():
    return


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


    def plot_roc_curve(df, y_true="y_test", y_score="y_score", dataset="name", figure=None, ax=None):
        """
        Plot auroc curve for a dataframe
        """
        data_names = df[
            dataset
        ].unique()  # retrieve different dataset-names from df (dataset-column defaults to "name")
        if figure is None:
            f = plt.figure(figsize=(10, 8))
        else:
            f = figure

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
                x=fpr, 
                y=tpr,
                label=f"{data_name} (AUC = {roc_auc:.2f})",
                ax=ax
            )

        sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", ax=ax)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1.05))
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curves")

        # plt.show()

        return ax


    def plot_prc_curve(df, y_true="y_test", y_score="y_score", dataset="name", figure=None, ax=None):
        """Plot Precision-Recall curve with confidence intervals"""
        data_names = df[
            dataset
        ].unique()  # retrieve different dataset-names from df (dataset-column defaults to "name")

        f = plt.figure(figsize=(10, 8)) if figure is None else figure

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
                ax=ax
            )

        # plt.hlines(
            # pos_rate,
            # 0,
            # 1,
            # colors="gray",
            # linestyles="--",
            # label=f"Baseline = {pos_rate:.3f}",
            # ax=ax
        # )
        sns.lineplot(x=[0,1], y=[pos_rate, pos_rate], linestyle="--", ax=ax)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PRC Curve ")

        return ax


    def run_test(column_name, data_dir, test_dataset, modality, modelname="sfcn"):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        test_dataset = dataloader.BrainDataset(
            csv_file=abspath(
                join(data_dir, test_dataset, "test", f"{column_name}.csv")
            ),
            root_dir=abspath(
                join(tensor_dir_test, "mspaths2", f"{modality}96_affine")
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


    df = pd.DataFrame()
    return plot_prc_curve, plot_roc_curve, run_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Demographics
    """)
    return


@app.cell
def _(patientstable, pd):
    pat_df2 = pd.read_csv(patientstable, dtype={"site": str})
    pat_df2
    return


@app.cell
def _(columns, data_dir, dataset_order, join, patientstable, pd):
    # read patients characteristics table:
    pat_df = pd.read_csv(patientstable, dtype={"site": str})
    pat_df.site = pat_df.site.str.replace(".0", "")
    # get list of patients in test-dataset
    test_ids = pd.read_csv(
        join(data_dir, "mspaths2", "t1w", "test", f"{columns[0]}.csv")
    ).eid.to_list()

    # get list of patients in training-dataset
    training_ids = pd.read_csv(
        join(data_dir, "mspaths", "t1w", "train", f"{columns[0]}.csv")
    ).eid.to_list()

    # get list of patients in validation-dataset
    validation_ids = pd.read_csv(
        join(data_dir, "mspaths", "t1w", "val", f"{columns[0]}.csv")
    ).eid.to_list()

    pat_df.loc[pat_df.mpi.isin(training_ids), "dataset"] = "training"
    pat_df.loc[pat_df.mpi.isin(validation_ids), "dataset"] = "validation"
    pat_df.loc[pat_df.mpi.isin(test_ids), "dataset"] = "test"

    pat_df["dataset"] = pd.Categorical(
        pat_df["dataset"], categories=dataset_order, ordered=True
    )

    len(pat_df.mpi.unique())
    return (pat_df,)


@app.cell
def _(pat_df):
    pat_df
    return


@app.cell
def _(pat_df, pd):
    _demographics = {}
    for _ds in ["training", "validation", "test"]:
        _demographics[_ds] = [len(pat_df.query(f'dataset=="{_ds}"'))]

    _df = pd.DataFrame(_demographics).transpose()
    _df.columns = ["count"]
    print(_df["count"].sum())
    print(_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Patient distribution
    """)
    return


@app.cell
def _(pat_df):
    pat_df.mpi.unique()
    return


@app.cell
def _(pat_df, pd):
    _demographics = {}
    for _ds in pat_df.dataset.unique():
        _demographics[_ds] = [len(pat_df.query(f'dataset=="{_ds}"'))]

    _df = pd.DataFrame(_demographics).transpose()
    _df.columns = ["count"]

    percentage_df = _df["count"].apply(
        lambda x: (x / len(pat_df.mpi.unique())) * 100
    )
    percentage_df
    return


@app.cell
def _(pat_df, pd, plt, sns):
    # Create a barplot showing the train/test/validation split across centers
    sns.set_palette("Accent")
    _ax = pd.crosstab(pat_df["site"], pat_df["dataset"]).plot(
        kind="bar",
        stacked=True,
        figsize=(10, 4),
        colormap="Set2",
    )

    _ax.set_xlabel("Center ID")
    _ax.set_ylabel("Number of Patients")
    _ax.set_title(
        "Distribution of Patients Across Training, Validation, and Test Sets by Center"
    )
    _ax.legend(title="Dataset", labels=["Training", "Validation", "Test"])

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig("center_distribution_split.svg")
    plt.show()
    return


@app.cell
def _(cmap, pat_df, pd, plt):
    _ax = pd.crosstab(pat_df["site"], pat_df["dataset"]).plot(
        kind="barh", stacked=True, cmap=cmap
    )

    _ax.set_xlabel("number of patients")
    _ax.set_ylabel("Center ID")

    plt.savefig("test_train_center_split.svg")
    plt.show()
    return


@app.cell(hide_code=True)
def _(color_female, color_male, pat_df, pd, plt):
    _colors = [color_female, color_male]

    _ax = pd.crosstab(pat_df["dataset"], pat_df["sex"]).plot(
        kind="barh", stacked=True, legend=False, color=_colors
    )

    _ax.set_xlabel("Number of Subjects")
    _ax.set_ylabel("")
    # _ax.set_yticklabels(['Val', 'Train', 'Test'])
    plt.legend(
        title="",
        loc="upper right",
        labels=["female", "male"],
        bbox_to_anchor=(1.2, 1),
    )

    plt.tight_layout()
    plt.savefig("test_train_sex_split.svg")
    plt.show()
    return


@app.cell
def _(color_female, color_male, pat_df, pd, plt, sns):
    # Set up the plotting style
    sns.set_style(
        "whitegrid",
    )
    sns.set_palette("Set2")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 1) Bar plot showing count of males and females across datasets
    colors = [color_male, color_female]

    # Create crosstab for gender distribution by dataset
    gender_dist = pd.crosstab(pat_df["dataset"], pat_df["sex"])

    # Plot bar chart
    gender_dist.plot(
        kind="bar", ax=ax1, color=colors, edgecolor="black", linewidth=0.5
    )
    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("Count")
    ax1.set_title("Gender Distribution Across Datasets")
    ax1.legend(title="Sex", labels=["Male", "Female"])
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for i, (dataset, row) in enumerate(gender_dist.iterrows()):
        for j, (gender, count) in enumerate(row.items()):
            ax1.text(
                j + i * 0.1 - 0.15,
                count + 0.5,
                str(count),
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # 2) Pie chart showing overall gender distribution
    gender_counts = pat_df["sex"].value_counts()
    ax2.pie(
        gender_counts.values,
        labels=gender_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    ax2.set_title("Overall Gender Distribution")

    plt.tight_layout()
    plt.savefig("gender_distribution.svg")

    # 3) Create additional pie charts for each dataset
    datasets = pat_df["dataset"].unique()
    n_datasets = len(datasets)

    # Create a figure with subplots for each dataset
    fig_pies, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 6))

    if n_datasets == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_data = pat_df[pat_df["dataset"] == dataset]
        gender_counts = dataset_data["sex"].value_counts()

        # Plot pie chart for this dataset
        ax.pie(
            gender_counts.values,
            labels=gender_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax.set_title(f"Gender Distribution - {str(dataset).capitalize()}")

    plt.tight_layout()
    plt.savefig("gender_distribution_by_dataset.svg")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(color_female, color_male, ks_2samp, pat_df, plt, sns):
    ages_train = pat_df.loc[pat_df["dataset"] == "training", "age"]
    ages_test = pat_df.loc[pat_df["dataset"] == "test", "age"]
    ks_stat, p = ks_2samp(ages_train, ages_test)

    print("KS p-value:", p)

    plt.figure(figsize=(8, 5))
    sns.violinplot(
        data=pat_df.query('sex in ["female", "male"]'),
        x="dataset",
        y="age",
        inner="quart",
        hue="sex",
        cut=0,
        hue_order=["female", "male"],
        palette=[color_female, color_male],
        split=True,
    )
    ### We dont need stripplot - have enough data
    # sns.stripplot(
    #    data=pat_df.query('sex in ["female", "male"]'),
    #    x="dataset",
    #    y="age",
    #    hue="sex",
    #    hue_order=["female", "male"],
    #    # palette=[color_female, color_male],
    #    dodge = True,
    #    palette=[color_female, color_male],
    #    linewidth=0.1,
    #    alpha=0.1
    # )

    plt.xlabel("Dataset")
    plt.ylabel("Age")
    plt.title("")

    plt.legend(title="", loc="upper right", bbox_to_anchor=(1.5, 1))

    plt.tight_layout()

    plt.savefig("dataset_age_distribution.svg")
    plt.show()
    return


@app.cell
def _(pat_df, plt, sns):
    import scikit_posthocs as sp  # für paarweise Dunn-Tests; optional
    from scipy import stats

    # Annahme: pat_df bereits geladen mit Spalten 'age','sex','dataset'
    # Sicherstellen, dass dataset-Kategorien in gewünschter Reihenfolge sind:
    order = ["training", "validation", "test"]
    # pat_df['dataset'] = pd.Categorical(pat_df['dataset'], categories=order, ordered=True)

    # 1) Violinplot mit innerem Boxplot
    sns.violinplot(
        data=pat_df,
        x="dataset",
        y="age",
        order=order,
        inner="box",
        hue="dataset",
    )
    plt.xlabel("Dataset")
    plt.ylabel("Age")
    plt.title("")

    # 2) Kruskal-Wallis H-Test (global)
    groups = [
        pat_df.loc[pat_df["dataset"] == g, "age"].dropna().values for g in order
    ]
    kw_stat, kw_p = stats.kruskal(*groups)

    # Anzeige des Testergebnisses im Plot
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    text = f"Kruskal-Wallis H={kw_stat:.3f}\np={kw_p:.3e}"
    plt.gca().text(
        0.02,
        0.98,
        text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    plt.legend(title="", loc="upper right", bbox_to_anchor=(1.5, 1))

    plt.tight_layout()
    plt.show()

    # 3) Optional: paarweise Tests (Dunn) mit Bonferroni-Korrektur, nur anzeigen falls global signifikant
    if kw_p < 0.05:
        # Benötigt scikit-posthocs: pip install scikit-posthocs
        data_for_posthoc = pat_df[["age", "dataset"]].dropna()
        dunn = sp.posthoc_dunn(
            data_for_posthoc,
            val_col="age",
            group_col="dataset",
            p_adjust="bonferroni",
        )
        print("Dunn post-hoc (Bonferroni-korrigierte p-Werte):\n", dunn)
    else:
        print(
            "Kruskal-Wallis nicht signifikant (p >= 0.05); keine paarweisen Tests durchgeführt."
        )
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
        A["3D T1-weighted MRI"] --> B["Preprocessing"]
        B --> B1["Steps"]
        B1 --> B1a["N4 bias field correction"]
        B1 --> B1b["Skull-stripping"]
        B1 --> B1c["Affine + nonlinear registration to template"]
        B1 --> B1d["Intensity normalization (z-score)"]
        B1 --> B1e["Resample to fixed voxel spacing & crop/pad to ROI"]
        B --> C["Data augmentation (train only)"]
        C --> C1["Augmentations"]
        C1 --> C1a["Random affine/elastic"]
        C1 --> C1b["Random intensity scaling"]
        C1 --> C1c["Random flips/crops"]
        C1 --> C1d["Gaussian noise"]
        C --> D["Backbone: Foundation 3D Encoder"]
        D --> D1["Pretrained on large 3D brain MRI corpus"]
        D --> D2["Architecture: 3D ViT / 3D Swin Transformer or 3D CNN"]
        D --> D3["Output: Global feature vector"]
        D --> E["Clinical embedding (optional)"]
        E --> E1["Age, sex, disease duration, baseline PST/Dex scores"]
        E --> F["Concatenate features"]
        F --> G["Task heads"]
        G --> G1["Progression classifier (binary): Worsened >= 2 z-scores"]
        G --> G2["Regression head: predicted ΔPST z-score"]
        G --> G3["Regression head: predicted ΔDex z-score"]
        G --> G4["Uncertainty head: aleatoric + epistemic"]
        G1 --> H["Losses"]
        G2 --> H
        G3 --> H
        G4 --> H
        H --> H1["Combined loss"]
        H1 --> H1a["Binary cross-entropy (classifier)"]
        H1 --> H1b["MSE or Huber (regressions)"]
        H1 --> H1c["KL / MC-dropout loss (uncertainty)"]
        H1 --> H1d["Class-balancing / focal loss if needed"]
        H --> I["Training loop"]
        I --> I1["Fine-tune foundation encoder + heads"]
        I1 --> I2["Validation: AUROC, AUPRC, sensitivity at fixed specificity"]
        I1 --> I3["Calibration: reliability plots, expected calibration error"]
        I --> J["Explainability & QC"]
        J --> J1["Saliency / Grad-CAM (3D)"]
        J --> J2["SHAP on clinical + global features"]
        J --> J3["Overlay predicted risk on MRI slices"]
        J --> K["Deployment"]
        K --> K1["Input: single 3D T1 - Preproc - Model"]
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


@app.cell
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


@app.cell
def _():
    return


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
    data: {"type": "error", "errorText": "'OpenAIModelProfile' object has no attribute 'supports_thinking'"}
    """)
    return


@app.cell
def _(df_flair, df_t1w, modalities, plot_prc_curve, plot_roc_curve, plt):
    _fig, _axs = plt.subplots(3,3,figsize=(15, 15))

    _width_ratios = [0.3, 5, 5]
    _height_ratios = [0.3, 5, 5]

    _gs = plt.GridSpec(
        figure=_fig,
        ncols= 3,
        nrows= 3,
        width_ratios=_width_ratios,
        height_ratios=_height_ratios,
        wspace=0.15,  # Width space between plots (decrease this to reduce space)
        hspace=0,  # Height space between plots
    )

    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.9, hspace=0)


    _axs[0, 0].axis("off")  # empty corner cell




    # Tablehead

    for _col, _mod in enumerate(modalities, start=1):
        _axs[0, _col].axis("off")
        _axs[0, _col].text(
            0.5,
            0.5,
            _mod,
            ha="center",
            va="center",
            color="black",
            fontsize=30,
        )

    # Curve names
    for _row, _name in enumerate(["ROC", "PRC"], start=1):
        _axs[_row, 0].axis("off")
        _axs[_row, 0].text(
            0.5,
            0.5,
            _name.upper(),
            ha="right",
            va="center",
            color="black",
            fontsize=30,
            rotation=90,
        )

    _fig.add_subplot(plot_roc_curve(df_t1w,   ax=_axs[1,1], figure=_fig))
    _fig.add_subplot(plot_prc_curve(df_t1w,   ax=_axs[2,1], figure=_fig))
    _fig.add_subplot(plot_roc_curve(df_flair, ax=_axs[1,2], figure=_fig))
    _fig.add_subplot(plot_prc_curve(df_flair, ax=_axs[2,2], figure=_fig))

    for _x in [1,2]:
        for _y in [1,2]:
            _axs[_x,_y].set_title('')

    plt.tight_layout()
    plt.savefig("auroc_prc.svg")

    plt.show()
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 3: Progression Curves

    ## Kaplan Meier Curves for T1w
    """)
    return


@app.cell
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
                f1_scores[i] = (
                    2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
                )
        f1_idx = np.argmax(f1_scores)
        f1_threshold = (
            pr_thresholds[f1_idx] if f1_idx < len(pr_thresholds) else 1.0
        )
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

        # Initialize Kaplan-Meier fitter
        kmf = KaplanMeierFitter()

        # Create figure
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

        # --- Add Hazard Ratio calculation using Cox Proportional Hazards Model ---
        from lifelines import CoxPHFitter

        df_cox = df[["time", "event", "risk_group"]].copy()
        df_cox.columns = ["T", "E", "risk_group"]

        cph = CoxPHFitter()
        cph.fit(df_cox, duration_col="T", event_col="E", strata=None)

        hr = cph.hazard_ratios_.iloc[0]
        hr_ci = cph.confidence_intervals_.iloc[0]

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
            fontsize=12,
            verticalalignment="bottom",
            bbox=props,
        )

        # Add labels and title
        ax.set_xlabel("Time (days)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Progression-Free Survival", fontsize=14, fontweight="bold")
        ax.set_title(
            f"Kaplan-Meier Curve on {test_cohort}",
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

        return km_metrics


    def kmplots(df, name):
        col_mapping = {"_pst": "PST", "_cst": "CST", "_wst": "WST", "_mdt": "MDT"}
        for _column in columns:
            _data_df = pd.read_csv(
                join(data_dir, "mspaths2", "t1w", "test", f"{_column}.csv")
            )
            _test_name = _column.replace("worst_progressor_2ycutoff_", "").replace(
                "_2z", ""
            )
            _data_df = _data_df.query(f"not(time_{_test_name} <= 0)")

            shortname = next(
                (v for k, v in col_mapping.items() if k in _column), None
            )
            km_data = _data_df.merge(df.query(f'name == "{shortname}"'))
            # km_data.time.fillna(0, inplace=True)
            km_data.dropna(subset=f"time_{_test_name}", inplace=True)
            km_data.to_csv(join("data", f"kmdata_{_column}.csv"), index=False)
            thresholds_dict = find_optimal_thresholds(
                km_data["y_test"].values, km_data["y_score"].values
            )
            time_to_event = km_data[f"time_{_test_name}"].values
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
def _(columns, data_dir, join, pd):
    for _column in columns:
        _data_df = pd.read_csv(
            join(data_dir, "mspaths2", "t1w", "test", f"{_column}.csv")
        )

        print(_data_df)
    return


@app.cell
def _(df_t1w):
    df_t1w
    return


@app.cell
def _(df_t1w, kmplots):
    kmplots(df_t1w, "T1w")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kaplan Meier Curves for FLAIR
    """)
    return


@app.cell
def _(df_flair, kmplots):
    _ax = kmplots(df_flair, "FLAIR")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 4: Heatmaps
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 5: Regional attention
    """)
    return


@app.cell
def _(pd):
    attention_maps_df = pd.DataFrame()

    test_names = ["pst", "mdt", "cst", "wst"]
    modalities = ["T1w", "FLAIR"]
    for test_name in test_names:
        for modality in modalities:
            _df = pd.read_csv(
                f"regional_attention/regional_scores_{test_name}_{modality}.csv",
                index_col="eid",
            )

            _df["modality"] = modality
            _df["test"] = test_name

            attention_maps_df = pd.concat(
                (attention_maps_df, _df), ignore_index=True
            )

    attention_maps_df.columns
    return attention_maps_df, modalities, test_names


@app.cell
def _():
    # attention_maps_df.pivot(index=['eid', 'modality', 'test'], columns=['Precentral_L' , 'Precentral_R'])
    return


@app.cell
def _(attention_maps_df):
    def get_top_100_per_column(
        df, region_columns, groupby_cols=["modality", "test"]
    ):
        """Get top 100 rows per region column, grouped by modality and test."""
        results = {}

        for region in region_columns:
            # Sort by region value descending and get top 100 per group
            top_100 = (
                df.sort_values(region, ascending=False)
                .groupby(groupby_cols, group_keys=False)
                .head(100)
                .copy()
            )
            results[region] = top_100

        return results


    # Get all region columns (excluding 'modality' and 'test')
    region_columns = [
        col for col in attention_maps_df.columns if col not in ["modality", "test"]
    ]

    # Get top 100 per region
    top_100_results = get_top_100_per_column(attention_maps_df, region_columns)

    print(f"Processed {len(top_100_results)} regions")
    print(f"Sample region: {list(top_100_results.keys())[0]}")
    return (top_100_results,)


@app.cell
def _(top_100_results):
    # Example: Access top 100 results for a specific region
    sample_region = "Precentral_L"
    top_100_sample = top_100_results[sample_region]

    print(f"Top 100 results for {sample_region}:")
    print(f"Total rows: {len(top_100_sample)}")
    print(f"\nBreakdown by modality and test:")
    print(top_100_sample.groupby(["modality", "test"]).size())

    print(f"\nFirst few rows of {sample_region} sorted by value:")
    print(top_100_sample.head(10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sailiency maps
    """)
    return


@app.cell
def _(
    cm,
    columns,
    explainability_dir,
    glob,
    join,
    mcolors,
    modalities,
    np,
    pd,
    plot_stat_map,
    plt,
    scores_dir,
    test_names,
):
    model = "sfcn"
    selected_slices = [31, 38, 50]
    _vmin = 0
    _vmax = 0.6

    # -----------------
    # Extract eids of top FLAIR results
    # ------------------

    best_ids = {}
    for _col in columns:
        _df = pd.read_csv(
            join(
                scores_dir,
                model,
                "test",
                "mspaths2",
                "flair",
                f"{_col}_e1000_b16_im96.csv",
            )
        )
        _true_df = pd.DataFrame(_df.query("label == pred_class"))
        _true_df.sort_values(by="prob_class_1", ascending=False, inplace=True)

        # extract eid with highest prob for true positive progressors
        best_ids[_col] = _true_df.eid.to_list()[0]


    plt.rcParams.update(
        {
            # Figure / saved‑file background
            "figure.facecolor": "black",
            "savefig.facecolor": "black",
            # Axes background (the part inside each subplot)
            "axes.facecolor": "black",
            # Axes edge / spines – we want them black as well
            "axes.edgecolor": "black",  # keep the spines visible (optional)
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            # Grid lines (if you ever turn a grid on)
            "grid.color": "black",
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
        }
    )


    _figure_cols = modalities
    _figure_rows = test_names
    _figure = plt.figure(figsize=(20, 24))
    _figure.patch.set_facecolor("black")

    _width_ratios = [1, 5, 5]
    _height_ratios = [0.3, 2, 2, 2, 2, 0.1]
    print(len(_figure_cols) + 1)
    _gs = plt.GridSpec(
        figure=_figure,
        ncols=len(_figure_cols) + 1,
        nrows=len(_figure_rows) + 2,
        width_ratios=_width_ratios,
        height_ratios=_height_ratios,
        wspace=0.15,  # Width space between plots (decrease this to reduce space)
        hspace=0,  # Height space between plots
    )
    _ax = (
        _figure.subplots()
    )  # This creates a 2D array of axes matching the GridSpec
    _ax = np.array(
        [
            [_figure.add_subplot(_gs[i, j]) for j in range(_gs.ncols)]
            for i in range(_gs.nrows)
        ]
    )
    _figure.patch.set_facecolor("black")

    # Set all axes to have black background
    for _i in range(_ax.shape[0]):
        for _j in range(_ax.shape[1]):
            _ax[_i, _j].set_facecolor("black")
            # Optional: Make axis spines invisible for cleaner look
            for spine in _ax[_i, _j].spines.values():
                spine.set_visible(False)
            _ax[_i, _j].tick_params(
                axis="both", which="both", length=0
            )  # Remove ticks

    _figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)


    imagelist = {}
    for _col in columns:
        imagelist[_col] = {}

    # Fill in the actual plots

    for _modality in modalities:
        for _col in columns:
            _test = _col.split("_")[-2].upper()

            _eid = best_ids[_col]
            # Get filename of heatmap
            _heatmap = glob(
                join(
                    explainability_dir,
                    f"{_col}",
                    "mspaths2",
                    f"{_modality.lower()}",
                    f"{model}",
                    "saliency",
                    "magnitude",
                    f"{_col}_e1000_b16_im96",
                    f"single_{_eid}_*_heatmap.nii.gz",
                )
            )[0]

            # get filename of brainimg
            _brain = glob(
                join(
                    explainability_dir,
                    f"{_col}",
                    "mspaths2",
                    f"{_modality.lower()}",
                    f"{model}",
                    "saliency",
                    "magnitude",
                    f"{_col}_e1000_b16_im96",
                    f"single_{_eid}_*_brain.nii.gz",
                )
            )[0]
            imagelist[_col][_modality] = [_brain, _heatmap]
            print(_heatmap)

            _f = plot_stat_map(
                stat_map_img=_heatmap,
                bg_img=_brain,
                cut_coords=selected_slices,
                display_mode="z",
                radiological=True,
                cmap="jet",
                colorbar=False,
                transparency=_heatmap,
                resampling_interpolation="continuous",
                transparency_range=[0, 0.1],
                cbar_tick_format="%.2f",
                black_bg=True,
                title=None,
                vmin=_vmin,
                vmax=_vmax,
                annotate=False,
                dim=1.5,
                axes=_ax[columns.index(_col) + 1, modalities.index(_modality) + 1],
            )


    _ax[0, 0].axis("off")  # empty corner cell

    # Tablehead

    for _col, _mod in enumerate(modalities, start=1):
        _ax[0, _col].axis("off")
        _ax[0, _col].text(
            0.5,
            0.5,
            _mod,
            ha="center",
            va="center",
            color="white",
            fontsize=40,
        )

    # Show Test names
    for _row, _name in enumerate(test_names, start=1):
        _ax[_row, 0].axis("off")
        _ax[_row, 0].text(
            0.5,
            0.5,
            _name.upper(),
            ha="right",
            va="center",
            color="white",
            fontsize=40,
            rotation=90,
        )

    # ------------------------------------------------------------------
    #  Create & style the single colorbar
    # ------------------------------------------------------------------
    cax = _figure.add_subplot(_gs[-1, 1:3])
    cax.set_facecolor("black")
    cax.tick_params(colors="white", labelsize=24)
    cax.set_box_aspect(0.1)
    # cax = _ax[-1, 1:]

    # Create a ScalarMappable that matches nilearn's display
    norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)
    sm = cm.ScalarMappable(
        cmap="jet",
        norm=norm,
    )
    sm.set_array([])

    _cbar = _figure.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        label="Intensity",
        format="%.2f",
        aspect=0.01,
    )

    _cbar.set_ticks([0, 0.2, 0.4, 0.6], labels=[0.1, 0.2, 0.4, 0.6])
    plt.savefig("heatmaps_sliced_t1w_flair.svg")
    plt.show()
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mean attention over top 20 results for each test.
    """)
    return


@app.cell
def _(pd, plt, sns, spidy, test_order):
    ##### FLAIR

    sns.set_style("whitegrid")
    sns.set_theme(style="whitegrid", rc=None)

    sns.set_context("notebook", font_scale=1.75)
    plt.rcParams.update(
        {
            # Figure / saved‑file background
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            # Axes background (the part inside each subplot)
            "axes.facecolor": "white",
            # Axes edge / spines – we want them black as well
            "axes.edgecolor": "0.7",  # keep the spines visible (optional)
            "axes.labelcolor": "0.5",
            "xtick.color": "0.5",
            "ytick.color": "0.5",
            # Grid lines (if you ever turn a grid on)
            "grid.color": "black",
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
        }
    )

    region_df = pd.read_csv("im96_flair.csv")
    region_df["test"] = region_df.test.str.upper()
    regions_long_df = region_df.melt(
        id_vars=["test"], var_name="region", value_name="value"
    )

    regions_long_df = regions_long_df.rename(columns={"test": "dataset"})

    _ax = spidy.spiderplot(
        x="region",
        y="value",
        hue="dataset",
        legend=True,
        data=regions_long_df,
        hue_order=["PST", "MDT", "CST", "WST"],
        # palette=cmap,
    )

    _ax.set_title("Regional Attention - FLAIR")
    _ax.set_rlim([0, 0.055])

    plt.legend(bbox_to_anchor=(1.7, 1.2))
    plt.savefig("regional_major_attention_flair.svg")
    plt.savefig("regional_major_attention_flair.png")
    plt.show()

    grid = sns.FacetGrid(
        regions_long_df,
        col="dataset",
        col_wrap=4,
        hue="dataset",
        sharey=True,
        sharex=True,
        height=4,
        aspect=1.5,
    )
    grid.map_dataframe(sns.barplot, "value", "region")
    grid.fig.tight_layout(w_pad=1)
    grid.set_titles(col_template="{col_name}")
    plt.show()
    ##### T1w

    region_df = pd.read_csv("im96_t1w.csv")
    region_df["test"] = region_df.test.str.upper()
    regions_long_df = region_df.melt(
        id_vars=["test"], var_name="region", value_name="value"
    )




    regions_long_df = regions_long_df.rename(columns={"test": "dataset"})

    _ax = spidy.spiderplot(
        x="region",
        y="value",
        hue="dataset",
        legend=True,
        data=regions_long_df,
        hue_order=["PST", "MDT", "CST", "WST"],
        # palette=cmap,
    )
    _ax.set_title("Regional Attention - T1w")
    _ax.set_rlim([0, 0.055])

    plt.legend(bbox_to_anchor=(1.7, 1.2))
    plt.savefig("regional_major_attention_t1w.svg")
    plt.savefig("regional_major_attention_t1w.png")
    plt.show()

    regions_long_df = pd.DataFrame()
    for _modality in ["t1w", "flair"]:
        _mod_region_df = pd.read_csv(f"im96_{_modality}.csv")
        _mod_region_df["test"] = region_df.test.str.upper()
        _mod_regions_long_df = _mod_region_df.melt(id_vars=["test"], var_name="region", value_name="value")
        _mod_regions_long_df["modality"] = "T1w" if _modality == "t1w" else "FLAIR"


        regions_long_df = pd.concat((regions_long_df, _mod_regions_long_df))

    regions_long_df = regions_long_df.rename(columns={"test": "dataset"})

    print(regions_long_df)


    grid = sns.FacetGrid(
        regions_long_df,
        #x='region',
        row="dataset",
        col="modality",
        hue="dataset",
        row_order=test_order,
        sharey=True,
        sharex=True,
        height=4,
        aspect=1.5,
    )
    grid.map_dataframe(sns.barplot, "region", "value")
    grid.set_titles(row_template="", col_template="{col_name}")
    grid.set_axis_labels("")

    # Rotate x-axis tick labels
    for _ax in grid.axes.flat:
        _ax.tick_params(axis='x', rotation=60, labelrotation_mode="xtick")
        _ax.set_xlabel("")


    for _i, _ax in enumerate(grid.axes):
        _bbox = _ax[0].get_position()  # Get subplot position in figure coordinates
        grid.fig.text(
            _bbox.x0 - 0.1,         # 2.5% of figure width to the left
            _bbox.y0 + _bbox.height / 2,  # Vertically centered
            grid.row_names[_i],       # Row title
            ha='right', va='center', rotation=90, fontsize=20, weight=700
        )
        for _j, _axcol in enumerate(_ax):
            if _i == 0:

                _axcol.set_title(grid.col_names[_j], fontsize=20, weight=700)

            else:
                _axcol.set_title("")

    plt.savefig('regional_attention.svg')
    plt.savefig('regional_attention.png')

    plt.show()
    return region_df, regions_long_df


@app.cell
def _(region_df):
    region_df
    return


@app.cell
def _(regions_long_df):
    regions_long_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Top regions with Freesurfer results
    """)
    return


@app.cell
def _(pd, plt, sns, test_order):
    # read region means
    fsregions_df = pd.read_csv("region_means.tsv", 
                                  sep="\t", 
                                  dtype={'subject': str})
    fsregions_df['neurotes'] = fsregions_df.neurotes.str.upper() 


    # We don't need to aggregate the rows it's easier if we put together stuff from the original table

    fsregions_molten_df = fsregions_df.groupby(by=["neurotes", "modality", "region"]).aggregate({"mean_intensity": ["mean","std"] })
    # fsregions_molten_df.reset_index(inplace=True)
    print(fsregions_molten_df)

    fsgrid = sns.FacetGrid(
        fsregions_df,
        #x='region',
        row="neurotes",
        col="modality",
        hue="neurotes",
        row_order=test_order,
        sharey=True,
        sharex=True,
        height=4,
        aspect=1.5,
    )
    fsgrid.map_dataframe(sns.barplot, "region", "mean_intensity")
    fsgrid.set_titles(row_template="", col_template="{col_name}")
    fsgrid.set_axis_labels("")




    # Rotate x-axis tick labels
    for _ax in fsgrid.axes.flat:
        _ax.tick_params(axis='x', rotation=60, labelrotation_mode="xtick")
        _ax.set_xlabel("")
        _ax.set_ylabel("")


    for _i, _ax in enumerate(fsgrid.axes):
        _bbox = _ax[0].get_position()  # Get subplot position in figure coordinates
        fsgrid.fig.text(
            _bbox.x0 - 0.1,         # 2.5% of figure width to the left
            _bbox.y0 + _bbox.height / 2,  # Vertically centered
            fsgrid.row_names[_i],       # Row title
            ha='right', va='center', rotation=90, fontsize=20, weight=700
        )
        for _j, _axcol in enumerate(_ax):
            if _i == 0:
                fsgrid.col_names[_j] = "MPRAGE" if fsgrid.col_names[_j].lower() == "t1w" else fsgrid.col_names[_j]
                fsgrid.col_names[_j] = "FLAIR" if fsgrid.col_names[_j].lower() == "flair" else fsgrid.col_names[_j]

                _axcol.set_title(fsgrid.col_names[_j], fontsize=20, weight=700)

            else:
                _axcol.set_title("")

    plt.savefig('regional_attention.svg')
    plt.savefig('regional_attention.png')

    plt.show()
    return (fsregions_df,)


@app.cell
def _(fsregions_df):
    fsregions_df.query('region=="CorpusCallossum"')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Confusion Matrices
    """)
    return


@app.cell
def _(columns, evaluations_dir, join, pd, plt, sns, test_order):
    cms_df = pd.DataFrame()


    for _modality in ["t1w", "flair"]:
        for _colname in columns:
            _df = pd.read_csv(join(evaluations_dir, "metrics", "sfcn", "test", "mspaths2", _modality, f"{_colname}_e1000_b16_im96.csv"))

            _cm = _df[['tn', 'fp', 'fn', 'tp']]
            _cm['modality'] = _modality
            _cm['neurotest'] = _colname.split('_')[3].upper()
            cms_df = pd.concat((cms_df, _cm), ignore_index=True)

    rows = []
    for _, r in cms_df.iterrows():
        rows.append({
            "Predicted": "Positive", "True Label": "Positive", "Count": r["tp"],
            "modality": r["modality"], "neurotest": r["neurotest"],
        })
        rows.append({
            "Predicted": "Positive", "True Label": "Negative", "Count": r["fp"],
            "modality": r["modality"], "neurotest": r["neurotest"],
        })
        rows.append({
            "Predicted": "Negative", "True Label": "Positive", "Count": r["fn"],
            "modality": r["modality"], "neurotest": r["neurotest"],
        })
        rows.append({
            "Predicted": "Negative", "True Label": "Negative", "Count": r["tn"],
            "modality": r["modality"], "neurotest": r["neurotest"],
        })

    cm_long = pd.DataFrame(rows)

    # Reihenfolge der Klassen fixieren
    cat_order = ["Negative", "Positive"]


    g = sns.FacetGrid(
        cm_long,
        col="modality",          # Spalten = Modalities
        row="neurotest",         # Zeilen = Neurotests
        row_order=test_order,
        margin_titles=True,
        despine=True,
        sharex=True,
        sharey=True,
        height=4,
        aspect=1.5,
    
    )

    # ── 3. Heatmap auf jedes Facet mappen ──
    def draw_heatmap(data, **kwargs):
        pivot = data.pivot_table(
            index="True Label",
            columns="Predicted",
            values="Count",
            fill_value=0,
        )
        # Sicherstellen, dass beide Achsen beide Klassen enthalten
        pivot = pivot.reindex(index=cat_order, columns=cat_order, fill_value=0)
        sns.heatmap(
            pivot,
            annot=True,
            fmt="3g",
            cmap="Blues",
            cbar=False,
            square=True,
            linewidths=0.5,
            linecolor="gray",
            **kwargs
        )
    g.map_dataframe(draw_heatmap)
    # Achsen-Tick-Labels auf allen Facets setzen
    g.set_titles(col_template="", row_template="")
    g.set_axis_labels("")



    g.set_axis_labels("Predicted", "True Label")
    for _ax in g.axes.flat:
        _ax.set_xticks([0.5, 1.5])
        _ax.set_xticklabels(cat_order, rotation=0)
        _ax.set_yticks([0.5, 1.5])
        _ax.set_yticklabels(cat_order, rotation=0)

    for _i, _ax in enumerate(g.axes):
        _bbox = _ax[0].get_position()  # Get subplot position in figure coordinates
        g.fig.text(
            _bbox.x0 - 0.15,         # 2.5% of figure width to the left
            _bbox.y0 + _bbox.height / 2,  # Vertically centered
            g.row_names[_i].upper(),       # Row title
            ha='right', va='center', rotation=90, fontsize=20, weight=700
        )
        for _j, _axcol in enumerate(_ax):
            if _i == 0:
                g.col_names[_j] = "MPRAGE" if g.col_names[_j].lower() == "t1w" else g.col_names[_j]
                g.col_names[_j] = "FLAIR" if g.col_names[_j].lower() == "flair" else g.col_names[_j]

                _axcol.set_title(g.col_names[_j], fontsize=20, weight=700)

            else:
                _axcol.set_title("")


    g.fig.subplots_adjust(left=0.15, top=0.95)  
    plt.tight_layout()
    plt.savefig("confusion_matrices.svg")
    plt.savefig("confusion_matrices.png",     
                dpi=300,
                bbox_inches="tight",    # schneidet nicht ab, erfasst fig.text
                pad_inches=0.2,         # etwas Padding rundherum)
               )


    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
