import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd


    baseline_characteristics_df = pd.read_csv('baseline_characteristics.csv')
    len(baseline_characteristics_df.query('sex == "female"'))
    return baseline_characteristics_df, mo, pd


@app.cell
def _(baseline_characteristics_df, pd):
    demographics = {}

    demographics['sex'] = [len(baseline_characteristics_df.query('sex in ["male", "female"]')),    len(baseline_characteristics_df.query('sex == "female"'))]

    demographics['age'] = [len(baseline_characteristics_df.query('age.notnull()')), baseline_characteristics_df.age.mean(), baseline_characteristics_df.age.std() ]
    demographics['mstype'] = [len(baseline_characteristics_df.query('mstype.notnull()'))]


    demographics['pdds'] = [len(baseline_characteristics_df.query('pdds_scr.notnull()')), 

                            baseline_characteristics_df.pdds_scr.mean(), baseline_characteristics_df.pdds_scr.std()]


    demographics['mstype_cis'] = [pd.NA ,len(baseline_characteristics_df.query('mstype == "Clinically Isolated Syndrome"'))]
    demographics['mstype_rrms'] = [pd.NA ,len(baseline_characteristics_df.query('mstype == "Relapsing Remitting MS"'))]
    demographics['mstype_spms'] = [pd.NA ,len(baseline_characteristics_df.query('mstype == "Secondary Progressive MS"'))]
    demographics['mstype_ppms'] = [pd.NA ,len(baseline_characteristics_df.query('mstype == "Primary Progressive MS"'))]


    demographics
    return


@app.cell
def _(baseline_characteristics_df):
    mstype_counts = baseline_characteristics_df['mstype'].value_counts()
    mstype_counts_percentage = (mstype_counts / len(baseline_characteristics_df.query('mstype.notnull()'))) * 100

    print(mstype_counts)
    print(mstype_counts_percentage)
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
