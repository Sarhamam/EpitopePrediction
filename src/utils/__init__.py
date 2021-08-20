import os
import logging
import configparser
from pathlib import Path


real_path = Path(os.path.realpath(__file__))
config = configparser.ConfigParser()
config.read(os.path.join(real_path.parent.parent.absolute(), "settings.ini"))

config["TRAIN"]["FASTA"] = os.path.join(real_path.parent.parent.parent.absolute(), "resources", "test.fasta")
config["TRAIN"]["TSV"] = os.path.join(real_path.parent.parent.parent.absolute(), "resources", "test.tsv")
config["LOGGER"]["FILE"] = os.path.join(real_path.parent.parent.parent.absolute(), "logs", "epitope_prediction.log")

logging.basicConfig(filename=config["LOGGER"]["FILE"], level=config["LOGGER"]["LEVEL"])
