import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["ipynb", "html"])


@app.cell
def _(join, root_dir):
    import visualtorch
    from config import Configuration
    ### Read config for paths

    conf = Configuration()
    conf.root_dir

    models_dir = join(root_dir, "data_extEU", "models")


    return


if __name__ == "__main__":
    app.run()
