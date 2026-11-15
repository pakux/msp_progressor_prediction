import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    df = pd.read_csv('~/Documents/mspaths/baseline_characteristics.csv')

    return (df,)


@app.cell
def _(df):
    df.query('site in [326,396,346]')
    return


if __name__ == "__main__":
    app.run()
