import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["ipynb", "html"])


@app.cell
def _(DataLoader, F, models_dir, tensor_dir_test):
    import marimo as mo

    import sys
    import torch
    import config
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import test_functions
    from pathlib import Path
    import matplotlib.pyplot as plt
    from os.path import join, abspath

    # from torch.utils.data import DataLoader


    conf = config.Configuration()
    conf.set_outputdir(abspath(join( '..', 'figures', 'extEU_KM3groups')))

    braindraindir = join(conf.root_dir, "code", "BrainTrain_Katherlab")
    print(braindraindir)
    sys.path.append(braindraindir)
    try:
        from utils.architectures import sfcn_cls
        from utils.dataloaders import dataloader
    except:
        print("Coul not load BrainTrain")
        mo.md("Could not load Braintrain! This might break things").callout(
            kind="danger"
        )




    data_dir = join(conf.root_dir, "data_extEU", "data")

    columns = [
        "worst_progressor_2ycutoff_pst_2z",
        "worst_progressor_2ycutoff_wst_2z",    
        "worst_progressor_2ycutoff_cst_2z",    
        "worst_progressor_2ycutoff_mdt_2z",
    ]


    def run_test(
        column_name,
        data_dir,
        test_dataset,
        modality,
        modelname="sfcn",
        tensor_dir_dataset="mspaths",
    ):
        device = torch.device(
            conf.device if torch.cuda.is_available() else "cpu"
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

    df = pd.DataFrame()

    sns.set_style('whitegrid')
    sns.set_theme(style='whitegrid', rc=None)
    sns.set_context('notebook', font_scale=1.75)
    return (
        Path,
        columns,
        data_dir,
        df,
        join,
        mo,
        pd,
        plt,
        run_test,
        test_functions,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read data
    """)
    return


@app.cell
def _(Path, columns, data_dir, pd, run_test):
    _outfile = Path(f"output_exteu_t1w.csv")
    if _outfile.exists():
        df_t1w = pd.read_csv(_outfile)
    else:
        df_t1w = pd.DataFrame()

        for _column_name in columns:
            # run_test should create and return y_test, y_score or write output.csv
            _eids, _y_test, _y_score = run_test(
                _column_name, data_dir, "mspaths/t1w/external_test", "t1w"
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

    _outfile = Path(f"output_exeu_flair.csv")
    if _outfile.exists():
        df_flair = pd.read_csv(_outfile)
    else:
        df_flair = pd.DataFrame()

        for _column_name in columns:
            # run_test should create and return y_test, y_score or write output.csv
            _eids, _y_test, _y_score = run_test(
                _column_name, data_dir, "mspaths/flair/external_test", "flair"
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
    return df_flair, df_t1w


@app.cell
def _(
    columns,
    data_dir,
    df,
    find_optimal_thresholds,
    join,
    km_data,
    output_dir,
    pd,
    plot_kaplan_meier,
    thresholds_dict,
):
    def kmplots(df_t1w, df_flair, name):

        col_mapping = {
            "_pst": "PST",
            "_cst": "CST",
            "_wst": "WST",
            "_mdt": "MDT",
        }
        figlist = []
        for _column in columns:
            shortname = next((v for k, v in col_mapping.items() if k in _column), None)
            _test_name = _column.replace("worst_progressor_2ycutoff_", "").replace("_2z", "")

            _data_df_t1w = pd.read_csv(
                join(
                    data_dir,
                    "mspaths",
                    "t1w",
                    "external_test",
                    f"{_column}.csv",
                )
            )
            _data_df_t1w = _data_df_t1w.query(f"not(time_{_test_name} <= 0)")
            km_data_t1w = _data_df_t1w.merge(df.query(f'name == "{shortname}"'))
            # km_data.time.fillna(0, inplace=True)
            km_data_t1w.dropna(subset=f"time_{_test_name}", inplace=True)
            km_data_t1w.to_csv(join("data", f"kmdata_{_column}.csv"), index=False)
            thresholds_dict_t1w = find_optimal_thresholds(
                km_data_t1w["y_test"].values, km_data_t1w["y_score"].values
            )

            _data_df_flair = pd.read_csv(
                join(
                    data_dir,
                    "mspaths",
                    "flair",
                    "external_test",
                    f"{_column}.csv",
                )
            )
            _data_df_flair = _data_df_flair.query(f"not(time_{_test_name} <= 0)")
            km_data_flair = _data_df_flair.merge(df.query(f'name == "{shortname}"'))
            # km_data.time.fillna(0, inplace=True)
            km_data_flair.dropna(subset=f"time_{_test_name}", inplace=True)
            km_data_flair.to_csv(join("data", f"kmdata_{_column}.csv"), index=False)
            thresholds_dict_flair = find_optimal_thresholds(
                km_data_flair["y_test"].values, km_data_flair["y_score"].values
            )

            time_to_event = km_data[f"time_{_test_name}"].values
            event_observed = km_data["y_test"].values
            prediction_scores = km_data["y_score"].values
            km_threshold = thresholds_dict["youden_threshold"]
            km_path = join(output_dir, f"{shortname}_{name}.svg")
            km_metrics, ax = plot_kaplan_meier(
                time_to_event,
                event_observed,
                prediction_scores,
                f"{shortname}",
                threshold=km_threshold,
                save_path=km_path,
            )
            figlist.append(ax)

            # plt.show()
        return figlist

    return


@app.cell
def _(df_t1w):
    df_t1w["name"].unique().tolist()
    return


@app.cell
def _(columns, data_dir, join, pd):
    events_dfs = {}

    for column in columns:
        event_df = pd.read_csv(
                join(
                    data_dir,
                    "mspaths",
                    "t1w",
                    "external_test",
                    f"{column}.csv",
                )
        )

        colname = column.replace("worst_progressor_2ycutoff_","").replace("_2z","").upper()
        events_dfs[colname] = event_df

    events_dfs["PST"].eid
    return (events_dfs,)


@app.cell
def _(df_flair, df_t1w, events_dfs, plt, test_functions):
    ax,fig = test_functions.kmplotter_multimodal({"MPRAGE": df_t1w, "FLAIR": df_flair}, events_dfs)

    plt.savefig("kmplots_intermediate.svg")
    plt.show()
    return (fig,)


@app.cell
def _(fig):
    fig
    return


if __name__ == "__main__":
    app.run()
