import yaml
from os.path import abspath, join, makedirs, isdir


class Configuration:
    def __init__(self, outputdir=None):
        with open("dirs.yaml") as f:
            conf = yaml.safe_load(f)

        self.root_dir = abspath(conf["paths"]["projectroot"])

        if "tensor_dir" not in conf["paths"].keys():
            self.tensor_dir_test = join(root_dir, "data_extDD", "images")
        else:
            self.tensor_dir_test = conf["paths"]["tensor_dir"]

        if not outputdir is None:
            self.set_outputdir(outputdir)

    def set_outputdir(self, outputdir="out"):

        self.outputdir = abspath(outputdir)
        if not isdir(outputdir):
            makedirs(self.outputdir)
