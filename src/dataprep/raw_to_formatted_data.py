from dataprep.utils import load_raw_table
from pathlib import Path


def main_raw_data_to_formatted_data(conf):
    format_x_test_ids(conf)
    format_Y_train_predilex(conf)
    format_x_train_ids(conf)


def write_formatted_data(conf, df, file_name):
    path_formatted_data = Path(conf["paths"]["formatted_data"])
    sep = ';'
    encoding = "utf-8"
    df.to_csv(path_formatted_data / file_name, sep=sep, encoding=encoding, index=False)


def format_x_test_ids(conf):
    df = load_raw_table(conf, "x_test_ids")
    write_formatted_data(conf, df, "x_test_ids.csv")


def format_Y_train_predilex(conf):
    df = load_raw_table(conf, "y_train_predilex")
    write_formatted_data(conf, df, "y_train.csv")


def format_x_train_ids(conf):
    df = load_raw_table(conf, "x_train_ids")
    write_formatted_data(conf, df, "x_train_ids.csv")






