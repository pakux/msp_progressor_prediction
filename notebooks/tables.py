import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def setup_1():
    import numpy as np
    import marimo as mo
    import pandas as pd
    from os.path import join, abspath, isfile
    from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact, ttest_ind, chisquare

    tests = ["pst", "mdt", "wst", "cst"]
    models = ["sfcn", "ssl-finetuned", "lora", "dense", "swin"]

    batchsize = 16
    imagesize = 96

    eval_dir = "/mnt/bulk-mars/paulkuntke/RadBrainDL_msp/evaluations/"

    baseline_characteristics_df = pd.read_csv("baseline_characteristics.csv")
    baseline_characteristics_df.mstype

    baseline_characteristics_df["mstype"] = baseline_characteristics_df[
        "mstype"
    ].replace("rr", "Relapsing Remitting MS")
    baseline_characteristics_df["mstype"] = baseline_characteristics_df[
        "mstype"
    ].replace("sp", "Secondary Progressive MS")
    baseline_characteristics_df["mstype"] = baseline_characteristics_df[
        "mstype"
    ].replace("pp", "Primary Progressive MS")
    baseline_characteristics_df["mstype"] = baseline_characteristics_df[
        "mstype"
    ].replace("Secondary Progressing MS", "Secondary Progressive MS")

    # dataset_df = pd.read_csv("eid_distribution.csv") # TODO: delete the file when this works
    dataset_df = pd.read_csv("data/dataset.csv")
    dataset_df = dataset_df.drop(columns=["site", "sex"])
    baseline_characteristics_df = baseline_characteristics_df.merge(
        right=dataset_df, left_on="mpi", right_on="eid"
    )

    baseline_characteristics_df = baseline_characteristics_df.query(
        "dataset.notnull()"
    )


    def progression_col(testname):
        return f"worst_progressor_2ycutoff_{testname}_2z"

    return (
        baseline_characteristics_df,
        batchsize,
        chi2_contingency,
        dataset_df,
        eval_dir,
        fisher_exact,
        imagesize,
        isfile,
        join,
        mannwhitneyu,
        mo,
        models,
        np,
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
def characteristics(
    baseline_characteristics_df,
    chi2_contingency,
    mannwhitneyu,
    np,
    pd,
    progression_col,
    tests,
):
    demographics = {}
    demographics["dataset"] = ["all n", "all value"]

    _sex_ds = len(baseline_characteristics_df.query("sex.notnull()"))
    _sex_f = len(baseline_characteristics_df.query('sex == "female"'))
    demographics["sex"] = [_sex_ds, f"{_sex_f} ({_sex_f / _sex_ds:0.2%})"]


    demographics["age"] = [
        len(baseline_characteristics_df.query("age.notnull()")),
        f"{baseline_characteristics_df.age.mean():0.1f} ± {baseline_characteristics_df.age.std():0.2f}",
    ]
    demographics["mstype"] = [
        len(baseline_characteristics_df.query("mstype.notnull()")),
        "",
    ]


    demographics["pdds"] = [
        len(baseline_characteristics_df.query("pdds_scr.notnull()")),
        f"{baseline_characteristics_df.pdds_scr.mean():0.1f} ± {baseline_characteristics_df.pdds_scr.std():0.2f}",
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
            baseline_characteristics_df.query(
                'mstype == "Progressive Relapsing MS"'
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

    for _t in tests:
        demographics[f"progressors_{_t}"] = [pd.NA, pd.NA]


    for ds in ["training", "validation", "external test"]:
        demographics["dataset"].append(f"{ds} n")
        demographics["dataset"].append(f"{ds} value")
        if ds in ["validation", "external test"]:
            demographics["dataset"].append(f"{ds} p")

        _sex_ds = len(
            baseline_characteristics_df.query("dataset == @ds and sex.notnull()")
        )
        _sex_f = len(
            baseline_characteristics_df.query('dataset == @ds and sex == "female"')
        )
        demographics["sex"].append(_sex_ds)
        demographics["sex"].append(f"{_sex_f} ({_sex_f / _sex_ds:0.2%})")

        # calculade p-value for test against training
        if ds in ["validation", "external test"]:
            _sex_training = np.array(
                [
                    [
                        len(
                            baseline_characteristics_df.query(
                                'sex == "female" and dataset=="training"'
                            )
                        ),
                        len(
                            baseline_characteristics_df.query(
                                'sex == "male" and dataset=="training"'
                            )
                        ),
                    ],
                    [
                        len(
                            baseline_characteristics_df.query(
                                'sex == "female" and dataset==@ds'
                            )
                        ),
                        len(
                            baseline_characteristics_df.query(
                                'sex == "male" and dataset==@ds'
                            )
                        ),
                    ],
                ]
            )
            _odds_ratio, _p_value, dof, expected = chi2_contingency(_sex_training)
            demographics["sex"].append(f"{_p_value:0.3f}")

        demographics["age"].append(
            len(
                baseline_characteristics_df.query(
                    "dataset == @ds and age.notnull()"
                )
            )
        )
        demographics["age"].append(
            f"{baseline_characteristics_df.query('dataset== @ds').age.mean():0.1f} ± {baseline_characteristics_df.query('dataset== @ds').age.std():0.2f}"
        )

        if ds in ["validation", "external test"]:
            _, _p = mannwhitneyu(
                baseline_characteristics_df.query('dataset=="training"')["age"]
                .dropna()
                .to_list(),
                baseline_characteristics_df.query("dataset==@ds")["age"]
                .dropna()
                .to_list(),
            )
            print(_p)
            demographics["age"].append(f"{_p:0.3f}")

        demographics["mstype"].append(
            len(
                baseline_characteristics_df.query(
                    "dataset == @ds and mstype.notnull()"
                )
            )
        )

        demographics["mstype"].append(pd.NA)
        if ds in ["validation", "external test"]:
            # TODO: add here mann-whitney-u
            demographics["mstype"].append(pd.NA)

        demographics["pdds"].append(
            len(
                baseline_characteristics_df.query(
                    "dataset == @ds and pdds_scr.notnull()"
                )
            )
        )
        demographics["pdds"].append(
            f"{baseline_characteristics_df.query('dataset == @ds').pdds_scr.mean():0.1f} ± {baseline_characteristics_df.query('dataset == @ds').pdds_scr.std():0.2f}"
        )

        if ds in ["validation", "external test"]:
            _, _p = mannwhitneyu(
                baseline_characteristics_df.query('dataset=="training"')[
                    "pdds_scr"
                ]
                .dropna()
                .to_list(),
                baseline_characteristics_df.query("dataset==@ds")["pdds_scr"]
                .dropna()
                .to_list(),
            )
            print(_p)

            demographics["pdds"].append(f"{_p:0.3f}")

        demographics["mstype_cis"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Clinically Isolated Syndrome"'
                )
            )
        )
        demographics["mstype_cis"].append(pd.NA)
        if ds in ["validation", "external test"]:
            # TODO: add here mann-whitney-u
            _cis_training = np.array(
                [
                    [
                        len(
                            baseline_characteristics_df.query(
                                'mstype !=  "Clinically Isolated Syndrome" and dataset=="training"'
                            )
                        ),
                        len(
                            baseline_characteristics_df.query(
                                'mstype ==  "Clinically Isolated Syndrome" and dataset=="training"'
                            )
                        ),
                    ],
                    [
                        len(
                            baseline_characteristics_df.query(
                                'mstype != "Clinically Isolated Syndrome" and dataset==@ds'
                            )
                        ),
                        len(
                            baseline_characteristics_df.query(
                                'mstype == "Clinically Isolated Syndrome" and dataset==@ds'
                            )
                        ),
                    ],
                ]
            )
            _odds_ratio, _p_value, dof, expected = chi2_contingency(_cis_training)
            demographics["mstype_cis"].append(f"{_p_value:0.3f}")

        demographics["mstype_prms"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset== @ds and mstype == "Progressmannwhitneyuive Relapsing MS"'
                )
            )
        )
        demographics["mstype_prms"].append(pd.NA)
        if ds in ["validation", "external test"]:
            # TODO: add here mann-whitney-u
            _cis_training = np.array(
                [
                    [
                        len(
                            baseline_characteristics_df.query(
                                'mstype !=  "Progressive Relapsing MS" and dataset=="training"'
                            )
                        ),
                        len(
                            baseline_characteristics_df.query(
                                'mstype ==  "Progressive Relapsing MS" and dataset=="training"'
                            )
                        ),
                    ],
                    [
                        len(
                            baseline_characteristics_df.query(
                                'mstype != "Progressive Relapsing MS" and dataset==@ds'
                            )
                        ),
                        len(
                            baseline_characteristics_df.query(
                                'mstype == "Progressive Relapsing MS" and dataset==@ds'
                            )
                        ),
                    ],
                ]
            )
            _odds_ratio, _p_value, dof, expected = chi2_contingency(_cis_training)
            demographics["mstype_prms"].append(f"{_p_value:0.3f}")

            # demographics["mstype_"].append(pd.NA)

        demographics["mstype_rrms"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Relapsing Remitting MS"'
                )
            )
        )
        demographics["mstype_rrms"].append(pd.NA)
        if ds in ["validation", "external test"]:
            # TODO: add here mann-whitney-u
            _cis_training = np.array(
                [
                    [
                        len(
                            baseline_characteristics_df.query(
                                'mstype !=  "Relapsing Remitting MS" and dataset=="training"'
                            )
                        ),
                        len(
                            baseline_characteristics_df.query(
                                'mstype ==  "Relapsing Remitting MS" and dataset=="training"'
                            )
                        ),
                    ],
                    [
                        len(
                            baseline_characteristics_df.query(
                                'mstype != "Relapsing Remitting MS" and dataset==@ds'
                            )
                        ),
                        len(
                            baseline_characteristics_df.query(
                                'mstype == "Relapsing Remitting MS" and dataset==@ds'
                            )
                        ),
                    ],
                ]
            )
            _odds_ratio, _p_value, dof, expected = chi2_contingency(_cis_training)
            demographics["mstype_rrms"].append(f"{_p_value:0.3f}")

        demographics["mstype_spms"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Secondary Progressive MS"'
                )
            )
        )
        demographics["mstype_spms"].append(pd.NA)
        if ds in ["validation", "external test"]:
            # TODO: add here mann-whitney-u
            _spms_training = np.array(
                [
                    [
                        len(
                            baseline_characteristics_df.query(
                                'mstype !=  "Secondary Progressive MS" and dataset=="training"'
                            )
                        ),
                        len(
                            baseline_characteristics_df.query(
                                'mstype ==  "Secondary Progressive MS" and dataset=="training"'
                            )
                        ),
                    ],
                    [
                        len(
                            baseline_characteristics_df.query(
                                'mstype != "Secondary Progressive MS" and dataset==@ds'
                            )
                        ),
                        len(
                            baseline_characteristics_df.query(
                                'mstype == "Secondary Progressive MS" and dataset==@ds'
                            )
                        ),
                    ],
                ]
            )
            _odds_ratio, _p_value, dof, expected = chi2_contingency(_spms_training)
            demographics["mstype_spms"].append(f"{_p_value:0.3f}")

        demographics["mstype_ppms"].append(
            len(
                baseline_characteristics_df.query(
                    'dataset == @ds and mstype == "Primary Progressive MS"'
                )
            )
        )
        demographics["mstype_ppms"].append(pd.NA)
        if ds in ["validation", "external test"]:
            _ppms_training = np.array(
                [
                    [
                        len(
                            baseline_characteristics_df.query(
                                'mstype !=  "Primary Progressive MS" and dataset=="training"'
                            )
                        ),
                        len(
                            baseline_characteristics_df.query(
                                'mstype ==  "Primary Progressive MS" and dataset=="training"'
                            )
                        ),
                    ],
                    [
                        len(
                            baseline_characteristics_df.query(
                                'mstype != "Primary Progressive MS" and dataset==@ds'
                            )
                        ),
                        len(
                            baseline_characteristics_df.query(
                                'mstype == "Primary Progressive MS" and dataset==@ds'
                            )
                        ),
                    ],
                ]
            )
            _odds_ratio, _p_value, dof, expected = chi2_contingency(_ppms_training)
            demographics["mstype_ppms"].append(f"{_p_value:0.3f}")

        for _t in tests:
            _n = len(
                baseline_characteristics_df.query(
                    f"dataset == @ds and ~{progression_col(_t)}.isna()"
                )
            )
            _n_training = len(
                baseline_characteristics_df.query(
                    f'dataset == "training" and ~{progression_col(_t)}.isna()'
                )
            )

            _n_trainingprogressors = len(
                baseline_characteristics_df.query(
                    f'dataset == "training" and {progression_col(_t)} == 1'
                )
            )

            _n_progressors = len(
                baseline_characteristics_df.query(
                    f"dataset == @ds and {progression_col(_t)} == 1"
                )
            )

            demographics[f"progressors_{_t}"].append(_n)
            demographics[f"progressors_{_t}"].append(
                f"{_n_progressors:0.0f} ({_n_progressors / _n:0.2%})"
            )
            if ds in ["validation", "external test"]:
                # TODO: add here mann-whitney-u
                _progressor_training = np.array(
                    [[_n_training, _n_trainingprogressors], [_n, _n_progressors]]
                )

                _odds_ratio, _p_value, dof, expected = chi2_contingency(_progressor_training)
                demographics[f"progressors_{_t}"].append(f"{_p_value:0.4f}")
                # demographics[f"progressors_{_t}"].append(pd.NA)
    # pd.DataFrame(demographics)
    demographics
    pd.DataFrame(demographics).transpose()
    return


@app.cell
def _(baseline_characteristics_df, fisher_exact, np):
    _sex_training = np.array(
        [
            [
                len(
                    baseline_characteristics_df.query(
                        'sex == "female" and dataset=="training"'
                    )
                ),
                len(
                    baseline_characteristics_df.query(
                        'sex == "male" and dataset=="training"'
                    )
                ),
            ],
            [
                len(
                    baseline_characteristics_df.query(
                        'sex == "female" and dataset=="validation"'
                    )
                ),
                len(
                    baseline_characteristics_df.query(
                        'sex == "male" and dataset=="validation"'
                    )
                ),
            ],
        ]
    )

    _odds_ratio, _p_value = fisher_exact(_sex_training)


    _sex_training = np.array(
        [
            [
                len(
                    baseline_characteristics_df.query(
                        'sex == "female" and dataset=="training"'
                    )
                ),
                len(
                    baseline_characteristics_df.query(
                        'sex == "male" and dataset=="training"'
                    )
                ),
            ],
            [
                len(
                    baseline_characteristics_df.query(
                        'sex == "female" and dataset=="external test"'
                    )
                ),
                len(
                    baseline_characteristics_df.query(
                        'sex == "male" and dataset=="external test"'
                    )
                ),
            ],
        ]
    )

    print(_sex_training)
    _odds_ratio, _p_value = fisher_exact(_sex_training.transpose())

    # chi2_contingency(_sex_training)

    _p_value
    # sex_training
    return


@app.cell
def _(
    baseline_characteristics_df,
    chi2_contingency,
    female1,
    fisher_exact,
    male1,
    mannwhitneyu,
    pd,
    progression_col,
    tests,
):
    _pvalues = {}
    _pvalues["dataset"] = ["training vs validation", "training vs external test"]


    def add_pvalue(var_name, test_type):
        _pvalues[var_name] = []

        for comparison in [
            ("training", "validation"),
            ("training", "external test"),
        ]:
            ds1, ds2 = comparison

            if test_type == "chi2":
                # For categorical variables (sex, mstype, progressors)
                # Create contingency table
                # This will depend on the specific variable
                pass
            elif test_type == "mannwhitney":
                # For continuous variables (age, pdds)
                pass


    _pvalues = {}
    _pvalues["dataset"] = ["training vs validation", "training vs external test"]


    # Helper function for safe Mann-Whitney U test
    def safe_mannwhitneyu(x, y):
        # Remove NaN values
        x_clean = x.dropna()
        y_clean = y.dropna()
        if len(x_clean) > 0 and len(y_clean) > 0:
            try:
                stat, p = mannwhitneyu(x_clean, y_clean)
                return p
            except:
                return pd.NA
        return pd.NA


    # Helper function for safe chi-squared test
    def safe_chi2_contingency(table):
        try:
            chi2, p, dof, expected = chi2_contingency(table)
            return p
        except:
            return pd.NA


    # Helper function for safe Fisher's exact test
    def safe_fisher_exact(table):
        try:
            odds_ratio, p = fisher_exact(table)
            return p
        except:
            return pd.NA

        # Sex (chi-squared or fisher's exact)
        _pvalues["sex"] = []
        for comparison in [
            ("training", "validation"),
            ("training", "external test"),
        ]:
            ds1, ds2 = comparison
            # Create contingency table: [female1, male1] vs [female2, male2]
            female1 = len(
                baseline_characteristics_df.query(
                    f'dataset == "{ds1}" and sex == "female"'
                )
            )
            male1 = len(
                baseline_characteristics_df.query(
                    f'dataset == "{ds1}" and sex == "male"'
                )
            )

        return pd.NA


    # Helper function for safe chi-squared test
    def safe_chi2_contingency(table):
        try:
            chi2, p, dof, expected = chi2_contingency(table)
            return p
        except:
            return pd.NA


    # Helper function for safe Fisher's exact test
    def safe_fisher_exact(table):
        try:
            odds_ratio, p = fisher_exact(table)
            return p
        except:
            return pd.NA

        female2 = len(
            baseline_characteristics_df.query(
                f'dataset == "{ds2}" and sex == "female"'
            )
        )
        male2 = len(
            baseline_characteristics_df.query(
                f'dataset == "{ds2}" and sex == "male"'
            )
        )
        table = [[female1, male1], [female2, male2]]
        _pvalues["sex"].append(safe_fisher_exact(table))


    # Age (Mann-Whitney U test)
    _pvalues["age"] = []
    for comparison in [("training", "validation"), ("training", "external test")]:
        ds1, ds2 = comparison
        age1 = baseline_characteristics_df.query(f'dataset == "{ds1}"').age
        age2 = baseline_characteristics_df.query(f'dataset == "{ds2}"').age
        _pvalues["age"].append(safe_mannwhitneyu(age1, age2))

    # MS type (chi-squared)
    _pvalues["mstype"] = []
    for comparison in [("training", "validation"), ("training", "external test")]:
        ds1, ds2 = comparison
        # Create contingency table for all MS types
        types1 = baseline_characteristics_df.query(
            f'dataset == "{ds1}" and mstype.notnull()'
        ).mstype.value_counts()
        types2 = baseline_characteristics_df.query(
            f'dataset == "{ds2}" and mstype.notnull()'
        ).mstype.value_counts()

        # Get all unique types
        all_types = set(types1.index) | set(types2.index)

        # Build table
        table = [
            [types1.get(t, 0) for t in all_types],
            [types2.get(t, 0) for t in all_types],
        ]
        _pvalues["mstype"].append(safe_chi2_contingency(table))

    # PDDS (Mann-Whitney U test)
    _pvalues["pdds"] = []
    for comparison in [("training", "validation"), ("training", "external test")]:
        ds1, ds2 = comparison
        pdds1 = baseline_characteristics_df.query(f'dataset == "{ds1}"').pdds_scr
        pdds2 = baseline_characteristics_df.query(f'dataset == "{ds2}"').pdds_scr
        _pvalues["pdds"].append(safe_mannwhitneyu(pdds1, pdds2))

    # MS type subtypes
    for mstype_var in [
        "mstype_cis",
        "mstype_prms",
        "mstype_rrms",
        "mstype_spms",
        "mstype_ppms",
    ]:
        _pvalues[mstype_var] = []
        for comparison in [
            ("training", "validation"),
            ("training", "external test"),
        ]:
            ds1, ds2 = comparison
            # Get counts (total vs specific subtype)
            total1 = len(
                baseline_characteristics_df.query(
                    f'dataset == "{ds1}" and mstype.notnull()'
                )
            )
            total2 = len(
                baseline_characteristics_df.query(
                    f'dataset == "{ds2}" and mstype.notnull()'
                )
            )

            # Parse the mstype_var name to get the condition
            mstype_map = {
                "mstype_cis": "Clinically Isolated Syndrome",
                "mstype_prms": "Progressive Relapsing MS",
                "mstype_rrms": "Relapsing Remitting MS",
                "mstype_spms": "Secondary Progressive MS",
                "mstype_ppms": "Primary Progressive MS",
            }

            subtype1 = len(
                baseline_characteristics_df.query(
                    f'dataset == "{ds1}" and mstype == "{mstype_map[mstype_var]}"'
                )
            )
            subtype2 = len(
                baseline_characteristics_df.query(
                    f'dataset == "{ds2}" and mstype == "{mstype_map[mstype_var]}"'
                )
            )

            # Create table: [count_of_interest, other_count] for each dataset
            table = [[subtype1, total1 - subtype1], [subtype2, total2 - subtype2]]
            _pvalues[mstype_var].append(safe_chi2_contingency(table))

    # Progressors for each test (Fisher's exact test)
    for test in tests:
        _pvalues[f"progressors_{test}"] = []
        for comparison in [
            ("training", "validation"),
            ("training", "external test"),
        ]:
            ds1, ds2 = comparison
            col = progression_col(test)

            # Get counts in the progressor column
            prog1 = len(
                baseline_characteristics_df.query(
                    f'dataset == "{ds1}" and ~{col}.isna() and {col} == 1'
                )
            )
            non_prog1 = len(
                baseline_characteristics_df.query(
                    f'dataset == "{ds1}" and ~{col}.isna() and {col} == 0'
                )
            )
            prog2 = len(
                baseline_characteristics_df.query(
                    f'dataset == "{ds2}" and ~{col}.isna() and {col} == 1'
                )
            )
            non_prog2 = len(
                baseline_characteristics_df.query(
                    f'dataset == "{ds2}" and ~{col}.isna() and {col} == 0'
                )
            )

            table = [[prog1, non_prog1], [prog2, non_prog2]]
            _pvalues[f"progressors_{test}"].append(safe_fisher_exact(table))

    pd.DataFrame(_pvalues).transpose()
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
        _per = (
            len(dataset_df.query(f'dataset == "{_ds}"')) * 100.0 / len(dataset_df)
        )
        print(f"{_ds}: {_per}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Median time between MRI and baseline functional assessment
    """)
    return


@app.cell
def _(baseline_characteristics_df, pd):
    pd.to_datetime(
        baseline_characteristics_df.sty_date.str.replace(".0", ""),
        unit="s",
        errors="coerce",
    )
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Performances
    """)
    return


@app.cell
def _(batchsize, eval_dir, imagesize, isfile, join, models, pd, tests):
    _performance = {}

    _performance["test"] = []
    _performance["modality"] = []
    _performance["model"] = []
    _performance["AUROC"] = []
    _performance["AUPRC"] = []
    _performance["Accuracy"] = []
    _performance["F1-Score"] = []


    for _neurotest in tests:
        for _modality in ["t1w", "flair"]:
            for _model in models:
                _performance["test"].append(_neurotest)
                _performance["modality"].append(_modality)
                _performance["model"].append(_model)
                _ftable = join(
                    eval_dir,
                    "summary",
                    _model,
                    "test",
                    "mspaths2",
                    _modality,
                    f"worst_progressor_2ycutoff_{_neurotest}_2z_e1000_b{batchsize}_im{imagesize}.csv",
                )
                if isfile(_ftable):
                    _df = pd.read_csv(_ftable)
                    _performance["AUROC"].append(
                        f"{_df['AUROC'][0]:0.2f} [{_df['AUROC_CI_lower'][0]:0.2f} - {_df['AUROC_CI_upper'][0]:0.2f}]"
                    )
                    _performance["AUPRC"].append(
                        f"{_df['AUPRC'][0]:0.2f} [{_df['AUPRC_CI_lower'][0]:0.2f} - {_df['AUPRC_CI_upper'][0]:0.2f}]"
                    )
                else:
                    _performance["AUROC"].append(pd.NA)
                    _performance["AUPRC"].append(pd.NA)

                _ftable = join(
                    eval_dir,
                    "metrics",
                    _model,
                    "test",
                    "mspaths2",
                    _modality,
                    f"worst_progressor_2ycutoff_{_neurotest}_2z_e1000_b{batchsize}_im{imagesize}.csv",
                )

                if isfile(_ftable):
                    _df = pd.read_csv(_ftable)
                    _performance["Accuracy"].append(f"{_df.accuracy[0]:0.2f}")
                    _performance["F1-Score"].append(f"{_df.f1[0]:0.2f}")
                else:
                    _performance["Accuracy"].append(pd.NA)
                    _performance["F1-Score"].append(pd.NA)


    pd.DataFrame(_performance)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Youdens metric
    """)
    return


@app.cell
def _(batchsize, eval_dir, imagesize, join, pd, tests):
    youden_df = pd.DataFrame()
    for _modality in ["t1w", "flair"]:
        for _neurotest in tests:
            _fname = join(eval_dir,"metrics", "sfcn", "test", "mspaths2", _modality,                f"worst_progressor_2ycutoff_{_neurotest}_2z_e1000_b{batchsize}_im{imagesize}_thresholds.csv")
            _df = pd.read_csv(_fname)
            _df['modality'] = _modality
            _df['neurotest'] = _neurotest
            youden_df = pd.concat((youden_df, _df), ignore_index=False)

    youden_df = youden_df[['modality', 'neurotest', 'youden_threshold', 'youden_sensitivity', 'youden_specificity', 'youden_index']]
    youden_df['Classification Rule'] = (
        "Score ≥ " + youden_df['youden_threshold'].round(4).astype(str) + 
        " → High Risk")


    # ✅ Step 1: Round all numeric columns to 4 decimal places
    # Select only numeric columns
    numeric_cols = youden_df.select_dtypes(include='number').columns

    # Apply rounding
    youden_df[numeric_cols] = youden_df[numeric_cols].round(4)

    youden_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
