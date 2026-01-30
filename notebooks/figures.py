import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv('../../../data/neuro_progressors_distance.csv')
    df.groupby('mpi').agg({'progressor_pst': 'sum'}) > 0
    return (df,)


@app.cell
def _(df):
    df.groupby('mpi').agg({'progressor_pst': 'sum'}) > 0
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
