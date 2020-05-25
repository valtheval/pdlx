from pathlib import Path
import pandas as pd
from os import listdir
from os.path import isfile, join
from dataprep.utils import load_config_raw_data, load_formatted_table
import re


def main_create_datasets(conf):
    df_test = create_df_test(conf)
    df_train = create_df_train(conf)
    write_datasets(conf, df_test, "df_test.csv")
    write_datasets(conf, df_train, "df_train.csv")


def load_all_jurisprudences(conf, dataset="train"):
    """ Load all the jurisprudence text files in the train or test folders and put it in a Dataframe"""
    confrd = load_config_raw_data(conf)
    all_txt = []
    if dataset == "train":
        path = Path(confrd["directory_raw_train_text"])
    elif dataset == "test":
        path = Path(confrd["directory_raw_test_text"])
    else:
        raise ValueError("Please use either 'train' or 'test' for dataset argument")
    all_files = [f for f in listdir(path) if (isfile(join(path, f)) and f.endswith(".txt"))]
    for file in all_files:
        path_file = Path(path) / file
        with open(path_file, "r") as f:
            txt = f.read().replace('\xa0', ' ').replace('\n', ' ')
            txt = re.sub(r"\s+", " ", txt)
            all_txt.append((file, txt))
    df = pd.DataFrame(all_txt, columns=["filename", "txt"])
    return df


def create_df_train(conf):
    """Create a pandas Dataframe of one text per rows"""
    train_ids = load_formatted_table(conf, "x_train_ids.csv")
    y_train = load_formatted_table(conf, "y_train.csv")
    df_train_text = load_all_jurisprudences(conf, "train")
    df = train_ids.merge(y_train, on="ID", how="left")
    df = df.merge(df_train_text, on="filename", how="left")
    return df


def create_df_test(conf):
    """Create a pandas Dataframe of one text per rows"""
    test_ids = load_formatted_table(conf, "x_test_ids.csv")
    #y_test = load_formatted_table(conf, "y_test.csv")
    df_test_text = load_all_jurisprudences(conf, "test")
    df = test_ids.merge(df_test_text, on="filename", how="left")
    #df = test_ids.merge(y_test, on="ID", how="left")
    return df

def write_datasets(conf, df, file_name):
    path_tmp = Path(conf["paths"]["dataprep"])
    sep=";"
    encoding = "utf-8"
    df.to_csv(path_tmp / file_name, sep=sep, encoding=encoding, index=False)