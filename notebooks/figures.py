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
    from os.path import join, abspath
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from pathlib import Path
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    import matplotlib.pyplot as plt


    # Define Paths and Filenames for further work / from previous work with BrainTrain
    # braindraindir = ('../../../RadBrainDL_msp/code/BrainTrain/') # source path of BrainTrain 🧠🚆 
    #                                                             # will be used to load modules 
    braindraindir = ('/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/code/BrainTrain/')

    # data_dir = ('../../../RadBrainDL_msp/data/')
    data_dir = ('/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/data/')
    models_dir = ('/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/models/')
    # tensor_dir_test = '../../../RadBrainDL_msp/images/'
    tensor_dir_test = '/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/images'

    modality = 'flair'
    sys.path.append(braindraindir)
    try:
        from dataloaders import dataloader
    except ModuleNotFoundError:
        mo.md("Could not load Braintrain! This might break things").callout(kind="danger")

    try:
        from architectures import sfcn_cls
    except ModuleNotFoundError:
        mo.md("Could not load SFCN module! This might break things.").callout(kind="danger")


    # Define some other variables
    test_dataset = f'mspaths2/{modality}'  


    columns = [
        'worst_progression_pst_2z',
        'worst_progression_mdt_2z',
        'worst_progression_cst_2z',
        'worst_progression_wst_2z',
              ]

    sns.set_style("whitegrid")
    sns.set_context("talk")


    return (
        DataLoader,
        F,
        Path,
        abspath,
        auc,
        columns,
        data_dir,
        dataloader,
        join,
        modality,
        models_dir,
        np,
        pd,
        plt,
        precision_recall_curve,
        roc_curve,
        sfcn_cls,
        sns,
        tensor_dir_test,
        test_dataset,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Perform tests on the dataset. This is only needed once. Remove
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(
    DataLoader,
    F,
    Path,
    abspath,
    auc,
    columns,
    data_dir,
    dataloader,
    join,
    mo,
    modality,
    models_dir,
    np,
    pd,
    plt,
    precision_recall_curve,
    roc_curve,
    sfcn_cls,
    sns,
    tensor_dir_test,
    test_dataset,
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
                precision, recall, _ = precision_recall_curve(y_true[indices], y_score[indices])
                score = auc(recall, precision)

            bootstrapped_scores.append(score)

        lower = np.percentile(bootstrapped_scores, 2.5)
        upper = np.percentile(bootstrapped_scores, 97.5)
        return np.mean(bootstrapped_scores), lower, upper

    def plot_roc_curve(df, y_true ="y_test", y_score="y_score", dataset="name" ):
        """
        Plot auroc curve for a dataframe
        """
        data_names = df[dataset].unique() # retrieve different dataset-names from df (dataset-column defaults to "name")
        f = plt.figure(figsize=(10,8))

        for data_name in data_names:
            subset = df[df[dataset] == data_name]
            y_true_array = np.array(subset[y_true].to_list())
            y_score_array = np.array( subset[y_score].to_list())

            fpr, tpr, _ = roc_curve(subset[y_true], subset[y_score])
            roc_auc = auc(fpr, tpr)
            roc_mean, roc_lower, roc_upper = bootstrap_auc(y_true_array, y_score_array, curve="roc")
            ax = sns.lineplot(x=fpr, y=tpr, label=f"{data_name} (AUC = {roc_auc:.2f})")

        sns.lineplot(x=[0,1], y=[0,1], linestyle='--')
        ax.set_xlim((0,1))
        ax.set_ylim((0,1.05))
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operationg Characteristic (ROC) Curves')

        # plt.show()

        return ax

    def plot_prc_curve(df, y_true ="y_test", y_score="y_score", dataset="name" ):
        """Plot Precision-Recall curve with confidence intervals"""
        data_names = df[dataset].unique() # retrieve different dataset-names from df (dataset-column defaults to "name")
        f = plt.figure(figsize=(10, 8))
        for data_name in data_names:
            subset = df[df[dataset] == data_name]
            y_true_array = np.array(subset[y_true].to_list())
            y_score_array = np.array( subset[y_score].to_list())
            precision, recall, _ = precision_recall_curve(y_true_array, y_score_array)
            prc_auc = auc(recall, precision)
            prc_mean, prc_lower, prc_upper = bootstrap_auc(y_true_array, y_score_array, curve="prc")
            pos_rate = y_true_array.mean()
    
            ax = sns.lineplot(x=recall, y=precision, lw=2, label=f"{data_name} (AUC = {prc_auc:.2f} [{prc_lower:.2f}–{prc_upper:.2f}])")
    
        plt.hlines(pos_rate, 0, 1, colors="gray", linestyles="--",
                   label=f"Baseline = {pos_rate:.3f}")
    
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PRC Curve ")
    
        return ax
    



    def run_test(column_name, data_dir, test_dataset):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test_dataset = dataloader.BrainDataset(
                csv_file= abspath(join(data_dir, test_dataset, 'test', f'{column_name}.csv')),
                root_dir=abspath(join(tensor_dir_test, 'mspaths2', modality)),
                column_name=column_name,
                num_rows=None,
                num_classes=2,
                task='classification'
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            num_workers=8,
            drop_last=False)

        # Load the model and accordingly the saved state
        model = sfcn_cls.SFCN(output_dim=2).to(device)
        checkpoint = torch.load(join(models_dir, 'sfcn', f'{column_name}_e1000_b32_im96.pth' ), map_location=device, weights_only=False)


        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)


        model.eval()



        test_outputs_binary = []
        test_labels = []
        test_eids = []


        with torch.no_grad():

            for eid, images, labels  in mo.status.progress_bar(test_loader):
                test_eids.extend(eid)
                images = images.to(device)
                labels = labels.float().to(device)
                binary_labels = labels[:, 1]
                test_labels.extend(binary_labels.tolist())

                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                binary_outputs = probs[:, 1]
                test_outputs_binary.extend(binary_outputs.tolist())

        y_true = np.array(test_labels).astype(int)
        y_score = np.array(test_outputs_binary).astype(float)            

        return y_true, y_score

    outfile = Path("output.csv")
    if outfile.exists():
        df = pd.read_csv(outfile)
    else:
        df = pd.DataFrame()

        for column_name in columns:  

            # run_test should create and return y_test, y_score or write output.csv
            y_test, y_score = run_test(column_name, data_dir, test_dataset)
            # Save to CSV (using pandas for header and robust types)
            df_current = pd.DataFrame({"y_test": y_test, "y_score": y_score, "name": column_name})

            df = pd.concat((df, df_current), ignore_index=True)
            df.to_csv(outfile, index=False)


    # Rename Entries to human readable format
    df.loc[df.name.str.contains('_pst'),'name'] = 'PST'
    df.loc[df.name.str.contains('_cst'),'name'] = 'CST'
    df.loc[df.name.str.contains('_wst'),'name'] = 'WST'
    df.loc[df.name.str.contains('_mdt'),'name'] = 'MDT'

    
    # Create Auroc Curves
    plot_roc_curve(df)
    plt.savefig("auroc_worst_progression_2z.svg")
    plt.show()

    # Create PRC Curves
    plot_prc_curve(df)
    plt.savefig("prc_worst_progression_2z.svg")
    plt.show()




    return (df,)


@app.cell
def _(pd):
    df_dist = pd.read_csv('../../../data/neuro_progressors_distance.csv')
    df_dist.groupby('mpi').agg({'progressor_pst': 'sum'}) > 0
    return (df_dist,)


@app.cell
def _(df_dist):
    df_dist.groupby('mpi').agg({'progressor_pst': 'sum'}) > 0
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
def _(df):
    df.loc[df.name.str.contains('_pst'),'name'] = 'PST'
    df.loc[df.name.str.contains('_cst'),'name'] = 'CST'
    df.loc[df.name.str.contains('_wst'),'name'] = 'WST'
    df.loc[df.name.str.contains('_mdt'),'name'] = 'MDT'
    df
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
    pipeline = mo.mermaid('''
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

    ''')
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
    diagram = '''
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
    '''

    mo.mermaid(diagram=diagram)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 2: Classification Performance (SFCN)
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## T1w - AUROCs (all 4 tasks)
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## T1w - AUPRC (all 4 tasks)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FLAIR -  AUROCs (all 4 tasks)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FLAIR - AUPRC (all 4 task
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 3: Progression Curves
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 4: Heatmaps
    """)
    return


if __name__ == "__main__":
    app.run()
