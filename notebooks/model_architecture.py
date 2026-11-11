import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["ipynb", "html"])


@app.cell
def _():
    import visualtorch
    import torch
    from config import Configuration
    from os.path import join

    ### Read config for paths

    conf = Configuration()
    conf.root_dir

    models_dir = join(conf.root_dir, "data_extEU", "models")

    return join, models_dir, torch, visualtorch


@app.cell
def _(join, model, models_dir, torch, visualtorch):
    device = torch.device("cpu")



    checkpoint = torch.load(join(models_dir, "sfcn", "t1w", "worst_progressor_2ycutoff_cst_2z_e1000_b16_im96.pth"),   
            map_location=device,
            weights_only=False,)

    img = visualtorch.render(model, input_shape=(96,96), style="flow", draw_volume=False)

    return


if __name__ == "__main__":
    app.run()
