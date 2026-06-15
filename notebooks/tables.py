import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def setup_1():
    import marimo as mo
    import pandas as pd
    from os.path import join, abspath

    tests = ['pst', 'mdt', 'wst', 'cst']

    baseline_characteristics_df = pd.read_csv("baseline_characteristics.csv")
    baseline_characteristics_df.mstype

    baseline_characteristics_df['mstype'] = baseline_characteristics_df['mstype'].replace('rr', 'Relapsing Remitting MS')
    baseline_characteristics_df['mstype'] = baseline_characteristics_df['mstype'].replace('sp', 'Secondary Progressive MS')
    baseline_characteristics_df['mstype'] = baseline_characteristics_df['mstype'].replace('pp', 'Primary Progressive MS')
    baseline_characteristics_df['mstype'] = baseline_characteristics_df['mstype'].replace('Secondary Progressing MS', 'Secondary Progressive MS')

    # dataset_df = pd.read_csv("eid_distribution.csv") # TODO: delete the file when this works
    dataset_df = pd.read_csv('data/dataset.csv')
    dataset_df = dataset_df.drop(columns=['site', 'sex'])
    baseline_characteristics_df = baseline_characteristics_df.merge(
        right=dataset_df, left_on="mpi", right_on="eid"
    )

    baseline_characteristics_df = baseline_characteristics_df.query('dataset.notnull()')

    def progression_col(testname):
        return f"worst_progressor_2ycutoff_{testname}_2z"

    return (
        baseline_characteristics_df,
        dataset_df,
        mo,
        pd,
        progression_col,
        tests,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read eid distribution from  `training`, `validation` and `external test` datasets.
    We need to keep in mind, that some ID's are not available for all tests.

    Maybe I should

    |                | Total | PST | MDT | WST | CST |
    | -------------- | ---   | --- | --- | --- | --- |
    | Task Completed |  nt   | nt  | nt  | nt. | nt  |
    | Age  mean (SD) |
    | sex female (%) |
    """)
    return


@app.cell
def _(baseline_characteristics_df, pd, progression_col, tests):
    demographics = {}
    demographics["dataset"] = ["all n", "all value"]

    _sex_ds = len(baseline_characteristics_df.query('sex.notnull()'))
    _sex_f = len(baseline_characteristics_df.query('sex == "female"'))
    demographics["sex"] = [_sex_ds, f"{_sex_f} ({_sex_f / _sex_ds:0.2%})"]
    demographics["age"] = [
        len(baseline_characteristics_df.query("age.notnull()")),
    f"{baseline_characteristics_df.age.mean():0.1f} ± {   baseline_characteristics_df.age.std():0.2f}"
    ]
    demographics["mstype"] = [
        len(baseline_characteristics_df.query("mstype.notnull()")), ""
    ]


    demographics["pdds"] = [
        len(baseline_characteristics_df.query("pdds_scr.notnull()")),
    f"{baseline_characteristics_df.pdds_scr.mean():0.1f} ± {
        baseline_characteristics_df.pdds_scr.std():0.2f}"
    ]

    demographics["mstype_cis"] = [
        pd.NA,
        len(
            baseline_characteristics_df.query(
                'mstype == "Clinically Isolated Syndrome"'
            )
        ),
    ]
    demographics["mstype_prms"] = [
       pd.NA,
        len(
            baseline_characteristics_df.query('mstype == "Progressive Relapsing MS"')
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

    for _t in tests:
        demographics[f"progressors_{_t}"] = [pd.NA, pd.NA]


    for ds in["training", "validation", "external test"]:

        demographics["dataset"].append(f"{ds} n")
        demographics["dataset"].append(f"{ds} value")

        _sex_ds = len(baseline_characteristics_df.query('dataset == @ds and sex.notnull()'))
        _sex_f = len(baseline_characteristics_df.query('dataset == @ds and sex == "female"'))
        demographics["sex"].append(_sex_ds)
        demographics["sex"].append( f"{_sex_f} ({_sex_f / _sex_ds:0.2%})")

        demographics["age"].append( len(
                baseline_characteristics_df.query(
                    "dataset == @ds and age.notnull()"
                )
            )
        )
        demographics["age"].append(
            f'{baseline_characteristics_df.query("dataset== @ds").age.mean():0.1f} ± {   baseline_characteristics_df.query("dataset== @ds").age.std():0.2f}'

        )

        demographics["mstype"].append(
            len(
                baseline_characteristics_df.query(
                    "dataset == @ds and mstype.notnull()"
                )
            )
        )
        demographics["mstype"].append(pd.NA)

        demographics["pdds"].append(
            len(
                baseline_characteristics_df.query(
                    "dataset == @ds and pdds_scr.notnull()"
                )
            )
        )
        demographics["pdds"].append(
    f"{baseline_characteristics_df.query("dataset == @ds").pdds_scr.mean():0.1f} ± {
        baseline_characteristics_df.query("dataset == @ds").pdds_scr.std():0.2f}" )

        demographics["mstype_cis"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Clinically Isolated Syndrome"'
                )
            )
        )
        demographics["mstype_cis"].append(pd.NA)

        demographics["mstype_prms"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset== @ds and mstype == "Progressive Relapsing MS"'
                )
            )
        )
        demographics["mstype_prms"].append(pd.NA)

        demographics["mstype_rrms"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Relapsing Remitting MS"'
                )
            )
        )    
        demographics["mstype_rrms"].append(pd.NA)


        demographics["mstype_spms"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Secondary Progressive MS"'
                )
            )
        )
        demographics["mstype_spms"].append(pd.NA)


        demographics["mstype_ppms"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Primary Progressive MS"'
                )
            )
        )
        demographics["mstype_ppms"].append(pd.NA)

        for _t in tests:
            _n = len(baseline_characteristics_df.query(f"dataset == @ds and ~{progression_col(_t)}.isna()"))
            _n_progressors = len(baseline_characteristics_df.query(f"dataset == @ds and {progression_col(_t)} == 1"))

            demographics[f"progressors_{_t}"].append(_n)
            demographics[f"progressors_{_t}"].append(
                f'{_n_progressors:0.0f} ({_n_progressors/ _n:0.2%})'
                )


    # pd.DataFrame(demographics)
    demographics
    pd.DataFrame(demographics).transpose()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Demographics


    """)

    mo.md(r"""
    \begin{tabular}{lrr}
    \toprule
        Category & Count & Percentage \\
    \midrule

    \bottomrule
    \end{tabular}
    """)
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
def _(dataset_df):
    for _ds in dataset_df.dataset.unique():
        _per = len(dataset_df.query(f'dataset == "{_ds}"')) * 100.0 / len(dataset_df) 
        print(f'{_ds}: {_per}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Median time between MRI and baseline functional assessment
    """)
    return


@app.cell
def _(baseline_characteristics_df, pd):
    pd.to_datetime(baseline_characteristics_df.sty_date.str.replace('.0',''), unit="s", errors="coerce")
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
