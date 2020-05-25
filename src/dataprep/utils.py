import json
from pathlib import Path
import pandas as pd


def load_config_raw_data(conf):
    """Load the configuration file that manage raw data. conf is a dictionary"""
    path = Path(conf["conf_raw_data"])
    with open(path) as f:
        txt = f.read()
        conf = json.loads(txt)
    return conf


def load_raw_table(conf, table):
    """Load as a pandas Dataframe the table specified by the name 'table' (string). Must match one of the keys in the \
    conf raw data file"""
    confrd = load_config_raw_data(conf)
    path_table = Path(confrd[table]["path"])
    sep = confrd[table]["sep"]
    encoding = confrd[table]["encoding"]
    df = pd.read_csv(path_table, sep=sep, encoding=encoding)
    return df


def load_formatted_table(conf, file_name):
    path_formatted_data = Path(conf["paths"]["formatted_data"])
    sep = ";"
    encoding = "utf-8"
    df = pd.read_csv(path_formatted_data / file_name, sep=sep, encoding=encoding)
    return df