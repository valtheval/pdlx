from pathlib import Path
import json
import os, shutil
import itertools
import pandas as pd
import pickle
import logging.config
import keras

def load_conf_file(path_conf_file):
    path = Path(path_conf_file)
    with open(path) as f:
        txt = f.read()
        conf = json.loads(txt)
    conf = eval_elem(conf)
    return conf


def setup_logging(default_path='conf/logging.json', default_level=logging.INFO, env_key='LOG_CFG'):
    """
    Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def remove_files_in_dir(path_dir):
    folder = path_dir
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_all_combi_grid_search(grid_search):
    all_keys = sorted(list(grid_search.keys()))
    combinations = itertools.product(*(grid_search[k] for k in all_keys))
    all_dict_param = [{all_keys[j]: combi[j] for j in range(len(all_keys))} for combi in combinations]
    return all_dict_param


def vprint(message, verbose=True):
    if verbose:
        print(message)


def eval_elem(e):
    if isinstance(e, dict):
        for key in e.keys():
            e[key] = eval_elem(e[key])
        return e
    elif isinstance(e, list):
        return [eval_elem(x) for x in e]
    elif isinstance(e, str):
        try:
            return eval(e)
        except (NameError, SyntaxError):
            return e
    else:
        return e

def flat_dictionary(d, all_keys, current_key):
    """Transform a dictionnary of several level to list of tuple of single level by aggregating keys with '__'.
    For instante {a : {b : 3}} becomes [(a__b,3)]"""
    if isinstance(d, dict):
        for k in d:
            if current_key == "":
                new_key = flat_dictionary(d[k], all_keys, k)
            else:
                new_key = flat_dictionary(d[k], all_keys, current_key + "__" + k)
        return all_keys
    elif isinstance(d, list):
        return all_keys.append((current_key, d))


def restruct_dict(din, dout):
    """From a flat dictionnary with keys made of '__' keys aggregation it returns the multiple level dictionnary.
    For instance {"a__"b : 3} becomes {"a" : {"b" : 3}}"""
    nb_pb = len([k for k in din.keys() if "__" in k])
    if nb_pb > 0:
        for k, v in din.items():
            if "__" in k:
                keys = k.split('__')
                head = keys[0]
                other = "__".join(keys[1:])
                if head in dout:
                    dout[head][other] = v
                else:
                    dout[head] = {other: v}
            else:
                dout[k] = v
        for k, v in dout.items():
            if isinstance(v, dict):
                dout[k] = restruct_dict(dout[k], {})
            else:
                pass
    else:
        return din
    return dout



def record_file(obj, path, sep=";", encoding="utf-8", mode="w", index=False):
    if isinstance(path, Path):
        pass
    else:
        path = Path(path)
    if path.suffix == ".csv": #TODO ajouter test si obj = pd.Dataframe
        obj.to_csv(path, sep=sep, encoding=encoding, index=index)
    elif path.suffix == ".json":
        with open(path, mode) as f:
            json.dump(obj, f)
    elif path.suffix == ".pkl":
        with open(path, mode+"b") as f:
            pickle.dump(obj, f, protocol=4)
    elif path.suffix == ".txt":
        with open(path, mode) as f:
            f.write(obj)
    elif path.suffix == ".h5":
        obj.save(path)


def load_file(path, sep=";", encoding="utf-8", usecols=None):
    if isinstance(path, Path):
        pass
    else:
        path = Path(path)
    if path.suffix == ".csv":
        obj = pd.read_csv(path, sep=sep, encoding=encoding, usecols=usecols)
    elif path.suffix == ".json":
        with open(path, "r") as f:
            obj = json.load(f)
    elif path.suffix == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
    elif path.suffix == ".h5":
        obj = keras.models.load_model(path)
    return obj


def get_all_results_files():
    all_results_files = []
    dir_path_all_exp = "../data/experiments/"
    model_list = [(d, join(dir_path_all_exp, d)) for d in listdir(dir_path_all_exp) if isdir(join(dir_path_all_exp, d))]
    for model, dir_model in model_list:
        dir_list_exp = [(dexp, join(dir_model, dexp)) for dexp in listdir(dir_model) if (isdir(join(dir_model, dexp)) and
                                                                                str(dexp).endswith("success") and
                                                                                (str(dexp).startswith("search") or
                                                                                 str(dexp).startswith("eval")))]
        for exp_name, dir_exp in dir_list_exp:
            if isfile(join(dir_exp, "results.csv")):
                all_results_files.append((model,
                                          exp_name,
                                          join(dir_exp, "results.csv")
                                         )
                                        )
    return all_results_files