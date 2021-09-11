import os
import sys
import logging
import configparser

from pathlib import Path

config = configparser.ConfigParser()
real_path = Path(os.path.realpath(__file__))
config.read(os.path.join(real_path.parent.parent.absolute(), "settings.ini"))

config["TRAIN"]["FASTA"] = os.path.join(real_path.parent.parent.parent.absolute(), "resources", "test.fasta")
config["TRAIN"]["JSON"] = os.path.join(real_path.parent.parent.parent.absolute(), "resources",
                                       "iedb_linear_epitopes.json")
config["LOGGER"]["FILE"] = os.path.join(real_path.parent.parent.parent.absolute(), "logs", "epitope_prediction.log")

logger = logging.getLogger()
fhandler = logging.FileHandler(filename=config["LOGGER"]["FILE"], mode='w')
shandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.addHandler(shandler)
logger.setLevel(config["LOGGER"]["LEVEL"])

logger = logging.getLogger("EpitopePrediction")
logger.info("=" * 40)
logger.info("Program started")
logger.info("=" * 40)
