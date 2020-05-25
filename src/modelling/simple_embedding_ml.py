from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_validate

import nltk
import re

import utils.utils as u
from pathlib import Path
import pickle
import json

from preprocessing.preprocessing import normalize, format_date, get_dates_from_token_list, get_context_date
import pandas as pd
import numpy as np

from time import time


class SimpleEmbeddingAndML():


    def __init__(self, ml_model="gbm", use_tfidf=True, param_bow=None, param_tfidf=None, param_model=None):
        # general_params
        self.use_tfidf = use_tfidf
        self.ml_model = ml_model
        self.preprocessor = self._build_preprocessor_transformer(param_bow, param_tfidf)
        self.model = self._build_ml_model(param_model)
        self.estimator = self._build_pipeline()

    def _build_preprocessor_transformer(self, param_bow=None, param_tfidf=None):
        # Building pipeline
        if param_bow is None:
            bow_vectorizer = CountVectorizer()
        else:
            bow_vectorizer = CountVectorizer(**param_bow)

        if self.use_tfidf:
            if param_tfidf is None:
                tfidf_transfo = TfidfTransformer()
            else:
                tfidf_transfo = TfidfTransformer(**param_tfidf)
            steps = [("bow", bow_vectorizer), ("tfidf", tfidf_transfo)]
        else:
            steps = [("bow", bow_vectorizer)]
        text_preprocessor = Pipeline(steps)

        def to_float(x):
            return x.astype(float)

        float_transformer = FunctionTransformer(to_float)

        preprocessor = ColumnTransformer([("textprep", text_preprocessor, 0), ("typing", float_transformer, [1, 2, 3])])
        return preprocessor


    def _build_ml_model(self, param_model=None):
        if self.ml_model == "gbm":
            model = GradientBoostingClassifier()
        elif self.ml_model == "adaboost":
            model = AdaBoostClassifier()
        elif self.ml_model == "rf":
            model = RandomForestClassifier()
        elif self.ml_model == "svc":
            model = SVC()
        else:
            raise ValueError("Please use either one of the following values 'gbm', 'adaboost', 'rf', 'svc'")

        if param_model is not None:
            model.set_params(param_model)
        return model


    def _build_pipeline(self):
        estimator = Pipeline([("preprocessor", self.preprocessor), ("model", self.model)])
        return estimator


    def set_params(self, params):
        # General params
        general_params_possible = ["use_tfidf", "ml_model"]
        general_params = {k : params[k] for k in params.keys() if k in general_params_possible}
        for key, value in general_params.items():
            setattr(self, key, value)

        # model params
        model_params = {}
        if "bow" in params.keys():
            for k, v in params["bow"].items():
                model_params["preprocessor__textprep__bow__%s"%str(k)] = v
        if (self.use_tfidf) and ("tfidf" in params.keys()):
            for k, v in params["tfidf"].items():
                model_params["preprocessor__textprep__tfidf__%s"%str(k)] = v
        if "model" in params.keys():
            for k, v in params["model"][self.ml_model].items():
                model_params["model__%s"%str(k)] = v
        self.preprocessor = self._build_preprocessor_transformer()
        self.model = self._build_ml_model()
        self.estimator = self._build_pipeline()
        self.estimator.set_params(**model_params)


    def fit(self, X, y):
        self.estimator.fit(X, y)


    def predict(self, X):
        return self.estimator.predict(X)


    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def preprocess_data(conf, dataset="train"):
    model_name = "simple_embedding_and_ml"
    conf_model = conf["models"][model_name]
    preprocessing = conf_model["preprocessing_%s"%dataset]
    v = conf_model["verbose"]
    context_size = conf_model["context_size"]
    path_model_dataprep_X = Path(conf_model["path"] + "X_%s_preprocessed_cs%d.pkl" % (dataset, context_size))
    path_model_dataprep_y = Path(conf_model["path"] + "y_%s_preprocessed_cs%d.pkl" % (dataset, context_size))
    if preprocessing:
        df = preprocess_data_general(conf, dataset=dataset)
        X, y = creating_matrix(conf, df)
        u.vprint("Record...", v)
        with open(path_model_dataprep_X, "wb") as f:
            pickle.dump(X, f)
        with open(path_model_dataprep_y, "wb") as f:
            pickle.dump(y, f)
    else:
        u.vprint("Loading already prepared data...", v)
        with open(path_model_dataprep_X, "rb") as f:
            X = pickle.load(f)
        with open(path_model_dataprep_y, "rb") as f:
            y = pickle.load(f)
    return X, y


def preprocess_data_general(conf, dataset="train"):
    model_name = "simple_embedding_and_ml"
    conf_model = conf["models"][model_name]
    v = conf_model["verbose"]
    path_dataprep = Path(conf["paths"]["dataprep"] + "df_%s.csv"%dataset)
    u.vprint("Loading...", v)
    df = pd.read_csv(path_dataprep, sep=';', encoding="utf-8")
    u.vprint("Lowercase text...", v)
    df["txt"] = df["txt"].str.lower()
    u.vprint("Format dates in text...", v)
    df["txt"] = df["txt"].map(format_date)
    u.vprint("Format date accident and consolidation...", v)
    df["date_accident"] = df["date_accident"].map(lambda x: re.sub(r"[-.]", "", x))
    df["date_consolidation"] = df["date_consolidation"].map(lambda x: re.sub(r"[-.]", "", x))
    u.vprint("Tokenize words...", v)
    df["txt"] = df["txt"].map(nltk.word_tokenize)
    u.vprint("Normalize text...", v)
    df["txt"] = df["txt"].map(normalize)
    u.vprint("Preprocessing dataframe completed", v)
    return df


def creating_matrix(conf, df):
    model_name = "simple_embedding_and_ml"
    conf_model = conf["models"][model_name]
    v = conf_model["verbose"]
    target_date = conf_model["date_target"]
    context_size = conf_model["context_size"]
    u.vprint("Creating numpy matrix", v)
    X = []
    y = []
    all_txt = df["txt"].values
    y_target_date = df[target_date]
    for i in range(all_txt.shape[0]):
        txt = all_txt[i]
        target_date = y_target_date[i]
        index_dates, dates_in_txt = get_dates_from_token_list(txt)
        for d in list(set(dates_in_txt)):
            left_context, right_context = get_context_date(context_size, d, txt)
            l = sum([list(c) for c in left_context], []) + sum([list(c) for c in right_context], [])
            s = " ".join(l)
            positions = index_dates[np.argwhere(dates_in_txt == d)].ravel()
            positions_mean = np.mean(positions)
            part_of_txt = positions / len(txt)
            part_of_txt_mean = np.mean(part_of_txt)
            nb_appearances = len(positions)
            # Reducing set with rules discovered with exploratory analysis
            if 0 in positions:
                # The target date can't be the first word of the text
                pass
            else:
                X.append([s, positions_mean, part_of_txt_mean, nb_appearances])
                if d == target_date:
                    y.append(1)
                else:
                    y.append(0)
    X = np.array(X)
    y = np.array(y)
    return X, y


def search(conf, X, y):
    model_name = "simple_embedding_and_ml"
    conf_model = conf["models"][model_name]
    grid_search = conf_model["grid_search"]
    date_target = conf_model["date_target"]
    v = conf_model["verbose"]
    path_model_cv_res = Path(conf_model["path"] + "full_search_results_%s.csv" % date_target)

    flat_grid_search = u.flat_dictionary(grid_search, [], "")
    all_combi = u.get_all_combi_grid_search(dict(flat_grid_search))
    nb_tot_combi = len(all_combi)

    res = []
    cv_res = []

    # Eval each set of params
    for i, combi in enumerate(all_combi):
        start = time()
        u.vprint("%d / %d" % (i + 1, nb_tot_combi), v)
        model = SimpleEmbeddingAndML()
        params = u.restruct_dict(combi, {})
        model.set_params(params)
        metrics_eval = ["f1", "recall", "precision", "accuracy", "roc_auc"]
        cv_result = cross_validate(model.estimator, X, y, cv=5, n_jobs=-1, scoring=metrics_eval,
                                   return_train_score=True)

        f1 = np.mean(cv_result["test_f1"])
        recall = np.mean(cv_result["test_recall"])
        precision = np.mean(cv_result["test_precision"])
        acc = np.mean(cv_result["test_accuracy"])
        auc = np.mean(cv_result["test_roc_auc"])
        res.append([params, f1, recall, precision, acc, auc])

        f1_train = np.mean(cv_result["train_f1"])
        recall_train = np.mean(cv_result["train_recall"])
        precision_train = np.mean(cv_result["train_precision"])
        acc_train = np.mean(cv_result["train_accuracy"])
        auc_train = np.mean(cv_result["train_roc_auc"])
        results = [params, f1_train, recall_train, precision_train, acc_train, auc_train, f1, recall, precision,
                       acc, auc]
        with open(path_model_cv_res, "a") as f:
            l = ";".join([str(r) for r in results])
            f.write(l+"\n")
        u.vprint("Iteration time %.2f"%(time() - start))

    # Get best params
    res = np.array(res)
    best_metrics_idx = np.argmax(res[:,1:], axis=0)
    best_metrics = np.max(res[:, 1:], axis=0)
    best_params = res[best_metrics_idx, 0]
    final_res = {"f1" : {"metric" : best_metrics[0], "best_params" : best_params[0]},
                 "recall" : {"metric" : best_metrics[1], "best_params" : best_params[1]},
                 "precision" : {"metric" : best_metrics[2], "best_params" : best_params[2]},
                 "acc" : {"metric" : best_metrics[3], "best_params" : best_params[3]},
                 "auc" : {"metric" : best_metrics[4], "best_params" : best_params[4]}
                 }

    #Record
    path_model_params = Path(conf_model["path"] + "best_params_%s.json"%date_target)
    with open(path_model_params, "w") as f:
        json.dump(final_res, f)

    return final_res


def main_simple_embedding_ml(conf):
    conf_model = conf["models"]["simple_embedding_and_ml"]
    learn = conf_model["learn"]
    do_search = conf_model["search"]
    assess = conf_model["assess"]
    v = conf_model["verbose"]
    date_target = conf_model["date_target"]

    X_train, y_train = preprocess_data(conf, dataset="train")
    # Get best params
    if do_search:
        u.vprint("Searching best param", v)
        final_res = search(conf, X_train, y_train)
        best_params = final_res["f1"]["best_params"]
        u.vprint("Best params are %s"%str(best_params), v)
    else: # search has already been done and we just read out best param recorded previously
        u.vprint("Reading best params in model folder", v)
        path_model_params = Path(conf_model["path"] + "best_params_%s.json"%date_target)
        with open(path_model_params, "r") as f:
            final_res = json.load(f)
        best_params = final_res["f1"]["best_params"]
    # Learning
    if assess or learn:
        if assess:
            u.vprint("Learning phase since this model cannot be pickled once trained")
        else:
            u.vprint("Learning phase", v)
        model = SimpleEmbeddingAndML()
        model.set_params(best_params)
        model.fit(X_train, y_train)
    # Assessing
    if assess:
        u.vprint("Assessing phase", v)
        metrics_eval = ["f1", "recall", "precision", "accuracy", "roc_auc"]
        cvr = cross_validate(model.estimator, X_train, y_train, cv=5, n_jobs=-1, scoring=metrics_eval,
                             return_train_score=True)
        f1_train = np.mean(cvr["train_f1"])
        recall_train = np.mean(cvr["train_recall"])
        precision_train = np.mean(cvr["train_precision"])
        accuracy_train = np.mean(cvr["train_accuracy"])
        auc_train = np.mean(cvr["train_roc_auc"])

        f1_test = np.mean(cvr["test_f1"])
        recall_test = np.mean(cvr["test_recall"])
        precision_test = np.mean(cvr["test_precision"])
        accuracy_test = np.mean(cvr["test_accuracy"])
        auc_test = np.mean(cvr["test_roc_auc"])

        print("Train : f1 = %.2f (recall = %.2f, precision = %.2f, AUC = %.2f, accuracy = .%2f)" % (f1_train,
                                                                                                    recall_train,
                                                                                                    precision_train,
                                                                                                    auc_train,
                                                                                                    accuracy_train))
        print("Test : f1 = %.2f (recall = %.2f, precision = %.2f, AUC = %.2f, accuracy = .%2f)" % (f1_test,
                                                                                                   recall_test,
                                                                                                   precision_test,
                                                                                                   auc_test,
                                                                                                   accuracy_test))



