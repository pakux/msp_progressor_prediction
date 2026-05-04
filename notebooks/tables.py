import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd


    baseline_characteristics_df = pd.read_csv("baseline_characteristics.csv")
    len(baseline_characteristics_df.query('sex == "female"'))

    dataset_df = pd.read_csv("eid_distribution.csv")


    baseline_characteristics_df = baseline_characteristics_df.merge(
        right=dataset_df, left_on="eid", right_on="eid"
    )
    return baseline_characteristics_df, mo, pd


@app.cell
def _(baseline_characteristics_df):
    baseline_characteristics_df.sex.unique()
    return


@app.cell
def _(baseline_characteristics_df, pd):
    demographics = {}
    demographics["dataset"] = ["all"]

    _sex_ds = len(baseline_characteristics_df.query('sex.notnull()'))
    _sex_f = len(baseline_characteristics_df.query('sex == "female"'))
    demographics["sex"] = [ _sex_ds, _sex_f, _sex_f * 100.0 / _sex_ds]



    demographics["age"] = [
        len(baseline_characteristics_df.query("age.notnull()")),
        baseline_characteristics_df.age.mean(),
        baseline_characteristics_df.age.std(),
    ]
    demographics["mstype"] = [
        len(baseline_characteristics_df.query("mstype.notnull()"))
    ]


    demographics["pdds"] = [
        len(baseline_characteristics_df.query("pdds_scr.notnull()")),
        baseline_characteristics_df.pdds_scr.mean(),
        baseline_characteristics_df.pdds_scr.std(),
    ]


    demographics["mstype_cis"] = [
        pd.NA,
        len(
            baseline_characteristics_df.query(
                'mstype == "Clinically Isolated Syndrome"'
            )
        ),
    ]
    demographics["mstype_rrms"] = [
        pd.NA,
        len(
            baseline_characteristics_df.query('mstype == "Relapsing Remitting MS"')
        ),
    ]
    demographics["mstype_spms"] = [
        pd.NA,
        len(
            baseline_characteristics_df.query(
                'mstype == "Secondary Progressive MS"'
            )
        ),
    ]
    demographics["mstype_ppms"] = [
        pd.NA,
        len(
            baseline_characteristics_df.query('mstype == "Primary Progressive MS"')
        ),
    ]


    for ds in baseline_characteristics_df.dataset.unique():

        demographics["dataset"].append(ds)
        _sex_ds = len(baseline_characteristics_df.query('dataset == @ds and sex.notnull()'))
        _sex_f = len(baseline_characteristics_df.query('dataset == @ds and sex == "female"'))
        demographics["sex"].append(_sex_ds)
        demographics["sex"].append(_sex_f)
        demographics["sex"].append(_sex_f * 100.0 / _sex_ds)

        demographics["age"].append( len(
                baseline_characteristics_df.query(
                    "dataset == @ds and age.notnull()"
                )
            )
        )
        demographics["age"].append(
            baseline_characteristics_df.query("dataset == @ds").age.mean()
        )
        demographics["age"].append(
            baseline_characteristics_df.query("dataset == @ds").age.std()
        )

        demographics["mstype"].append(
            len(
                baseline_characteristics_df.query(
                    "dataset == @ds and mstype.notnull()"
                )
            )
        )

        demographics["pdds"].append(
            len(
                baseline_characteristics_df.query(
                    "dataset == @ds and pdds_scr.notnull()"
                )
            )
        )
        demographics["pdds"].append(
            baseline_characteristics_df.query("dataset == @ds").pdds_scr.mean()
        )
        demographics["pdds"].append(
            baseline_characteristics_df.query("dataset == @ds").pdds_scr.std()
        )

        demographics["mstype_cis"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Clinically Isolated Syndrome"'
                )
            )
        )

        demographics["mstype_rrms"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Relapsing Remitting MS"'
                )
            )
        )
        demographics["mstype_spms"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Secondary Progressive MS"'
                )
            )
        )
        demographics["mstype_ppms"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Primary Progressive MS"'
                )
            )
        )


    # pd.DataFrame(demographics)
    demographics
    return


@app.cell
def _():
    # pd.DataFrame(demographics)
    return


@app.cell
def _(baseline_characteristics_df):
    mstype_counts = baseline_characteristics_df["mstype"].value_counts()
    mstype_counts_percentage = (
        mstype_counts / len(baseline_characteristics_df.query("mstype.notnull()"))
    ) * 100

    print(mstype_counts)
    print(mstype_counts_percentage)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Demographics

    | Category | Count | Percentage |
    | --- | --- | --- |
    | Clinically Isolated Syndrome |  |  |
    | Relapsing Remitting MS |  |  |
    | Secondary Progressive MS |  |  |
    | Primary Progressive MS |  |  |
    | Other |  |  |
    """)
    return


if __name__ == "__main__":
    app.run()
