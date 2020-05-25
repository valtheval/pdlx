from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split

import nltk
import re

import utils.utils as u
from pathlib import Path
import pickle
import json
import os
from datetime import datetime

from preprocessing.preprocessing import normalize, format_date, get_dates_from_token_list, get_context_date
from preprocessing.fasttext_embedding import FasttextTransformer
import pandas as pd
import numpy as np


class MLModel():

    def __init__(self, ml_model="gbm", text_preprocessing="tfidf", context_size=15, path_fasttext_model=None,
                 doc2vec_strategy="mean"):
        self.ml_model = ml_model
        self.text_preprocessing = text_preprocessing
        self.context_size = context_size
        self.path_fasttext_model = path_fasttext_model
        self.doc2vec_strategy = doc2vec_strategy
        self.estimator = self.get_estimator()


    def build_preprocessor_transformer(self):
        # preprocessor transformer
        if self.text_preprocessing == "tfidf":
            steps = [("bow", CountVectorizer()), ("tfidf", TfidfTransformer())]
        elif self.text_preprocessing == "bow":
            steps = [("bow", CountVectorizer())]
        elif self.text_preprocessing == "fasttext":
            ft_transformer = FasttextTransformer(self.path_fasttext_model, self.doc2vec_strategy)
            steps = [("fasttext", ft_transformer)]
        text_preprocessor = Pipeline(steps)
        self.text_preprocessor = text_preprocessor

        def to_float(x):
            return x.astype(float)

        float_transformer = FunctionTransformer(to_float, validate=True)

        preprocessor = ColumnTransformer([("textprep", text_preprocessor, 0), ("typing", float_transformer, [1, 2, 3])])
        self.preprocessor = preprocessor
        return preprocessor


    def build_ml_model(self):
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
        return model


    def get_estimator(self):
        preprocessor = self.build_preprocessor_transformer()
        model = self.build_ml_model()
        estimator = Pipeline([("preprocessor", preprocessor), ("model", model)])
        return estimator


    def fit(self, X, y):
        Xarr, yarr, ids_and_dates = self._creating_matrix(X, y)
        self.estimator.fit(Xarr, yarr)


    def predict(self, X):
        X, ids_and_dates = self._creating_matrix(X)
        proba_pred = self.estimator.predict_proba(X)
        index_target = np.argwhere(self.estimator.classes_ == 1)[0][0]
        df = pd.DataFrame(ids_and_dates, columns=["id", "date_possible"])
        df["proba"] = pd.Series(proba_pred[:, index_target].ravel(), index=df.index)
        df["rank"] = df.groupby("id")["proba"].rank("dense", ascending=False)
        self.full_prediction = df.copy()
        df_res = df[df["rank"] == 1].drop_duplicates(subset=["id"])
        date_pred = df_res["date_possible"].values.ravel()
        return date_pred


    def predict_proba(self, X):
        X, ids_and_dates = self._creating_matrix(X)
        proba_pred = self.estimator.predict_proba(X)
        index_target = np.argwhere(self.estimator.classes_ == 1)[0][0]
        df = pd.DataFrame(ids_and_dates, columns=["id", "date_possible"])
        df["proba"] = pd.Series(proba_pred[:, index_target].ravel(), index=df.index)
        df["rank"] = df.groupby("id")["proba"].rank("dense", ascending=False)
        self.full_prediction = df.copy()
        df_res = df[df["rank"] == 1].drop_duplicates(subset=["id"])
        proba_pred = df_res["proba"].values.ravel()
        return proba_pred


    def _creating_matrix(self, Xin, yin=None):
        """X is a preprocessed dataframe with one line per text, y is a series of dates (target)"""
        X = []
        ids_and_dates = []
        all_txt = Xin["txt"].values
        all_txt_id = Xin["ID"].values
        if yin is not None:
            y = []
            y_target_date = yin.values
        for i in range(all_txt.shape[0]):
            txt = all_txt[i]
            txt_id = all_txt_id[i]
            if yin is not None:
                target_date = y_target_date[i]
            index_dates, dates_in_txt = get_dates_from_token_list(txt)
            for d in list(set(dates_in_txt)):
                left_context, right_context = get_context_date(self.context_size, d, txt)
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
                    if yin is not None:
                       ids_and_dates.append([txt_id, target_date, d])
                       if d == target_date:
                           y.append(1)
                       else:
                           y.append(0)
                    else:
                        ids_and_dates.append([txt_id, d])
        if yin is not None:
            X = np.array(X)
            y = np.array(y)
            return X, y, ids_and_dates
        else:
            X = np.array(X)
            return X, ids_and_dates


def preprocess_data(conf, dataset="train"):
    model_name = "ml_model"
    conf_model = conf["models"][model_name]
    context_size = conf_model["context_size"]
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
    u.vprint("Record the dataframe")
    path_model_df_prep = Path(conf_model["path"] + "df_%s_preprocessed_cs%d.csv" % (dataset, context_size))
    df.to_csv(path_model_df_prep, sep=';', encoding="utf-8", index=False)
    u.vprint("Preprocessing dataframe completed", v)
    return df


def split_x_y(df, date="date_accident"):
    X = df["txt"].values
    y = df[date].values
    return X, y


def search(conf, estimator, X, y):
    model_name = "ml_model"
    conf_model = conf["models"][model_name]
    grid_search = conf_model["grid_search"]

    # Search
    scorings = ["f1", "recall", "precision", "accuracy", "roc_auc"]
    gs = GridSearchCV(estimator, param_grid=grid_search, scoring=scorings, n_jobs=-1, cv=5, refit="f1", error_score=0,
                      return_train_score=True, verbose=2)
    gs.fit(X, y)

    # Record
    record(conf, gs.cv_results_, "cv_results")
    return gs.best_params_


def get_experiment_directory(conf_model):
    path_model = conf_model["path"]
    context_size = conf_model["context_size"]
    date_target = conf_model["date_target"]
    ml_model = conf_model["ml_model"]
    text_preprocessing = conf_model["text_preprocessing"]

    if text_preprocessing == "fasttext":
        file_model = Path(conf_model["path_fasttext_model"]).name.replace(".", "")
        doc2vec_strat = conf_model["doc2vec_strategy"]
        label_text_preprocessing = "fasttext_%s_%s" % (file_model, doc2vec_strat)
    else:
        label_text_preprocessing = text_preprocessing

    dir_exp = Path(path_model + "experiments/%s_cs%s_ml%s_%s/" %
                   (str(date_target), str(context_size), str(ml_model), label_text_preprocessing))
    return dir_exp


def record(conf, obj, name):
    model_name = "ml_model"
    conf_model = conf["models"][model_name]

    dir_exp = get_experiment_directory(conf_model)
    if not os.path.exists(dir_exp):
        os.makedirs(dir_exp)

    if name == "cv_results":
        now = datetime.strftime(datetime.today(), "%Y%m%d_%H%M%S")
        path_file_cv_res = dir_exp / Path("full_cv_results_%s.pkl" % now)
        with open(path_file_cv_res, "wb") as f:
            pickle.dump(obj, f)
    elif name == "best_params":
        path_file_best_param = dir_exp / Path("best_params.json")
        with open(path_file_best_param, "w") as f:
            json.dump(obj, f)
    elif name == "assess_results":
        path_file_assess_res = dir_exp / Path("assess_results.txt")
        with open(path_file_assess_res, "w") as f:
            f.write(obj)


def compute_accuracy(y_true, y_pred):
    y_compar = y_pred == y_true
    accuracy = (y_compar.sum() / y_compar.shape[0])
    return accuracy


def main_ml(conf):

    conf_model = conf["models"]["ml_model"]
    ml_model = conf_model["ml_model"]
    learn = conf_model["learn"]
    text_preprocessing = conf_model["text_preprocessing"]
    path_fasttext_model = conf_model["path_fasttext_model"]
    doc2vec_strategy = conf_model["doc2vec_strategy"]
    do_search = conf_model["search"]
    context_size = conf_model["context_size"]
    assess = conf_model["assess"]
    v = conf_model["verbose"]
    date_target = conf_model["date_target"]
    preprocessing_train = conf_model["preprocessing_train"]


    dir_exp = get_experiment_directory(conf_model)

    # Get the estimator
    model = MLModel(ml_model=ml_model, text_preprocessing=text_preprocessing, context_size=context_size,
                    path_fasttext_model=path_fasttext_model, doc2vec_strategy=doc2vec_strategy)

    # Preprocess data
    if preprocessing_train:
        df_train = preprocess_data(conf, dataset="train")
    else:
        path_model_df_prep = Path(conf_model["path"] + "df_train_preprocessed_cs%d.csv" % (context_size))
        df_train = pd.read_csv(path_model_df_prep, sep=';', encoding="utf-8")
        df_train["txt"] = df_train["txt"].map(eval)

    # Splitting train and test
    df_train, df_test = train_test_split(df_train)

    # Splitting X and y
    df_X_train = df_train[["ID", "txt"]]
    df_y_train = df_train[date_target]
    df_X_test = df_test[["ID", "txt"]]
    df_y_test = df_test[date_target]
    # Get best params
    if do_search:
        u.vprint("Searching best param", v)
        X_train, y_train, ids_and_dates = model._creating_matrix(df_X_train, df_y_train)
        best_params = search(conf, model.estimator, X_train, y_train)
        u.vprint("Best params are %s"%str(best_params), v)
        u.vprint("Record best params", v)
        record(conf, best_params, "best_params")
    else: # search has already been done and we just read out best param recorded previously
        u.vprint("Reading best params in model folder", v)
        path_file_best_param = dir_exp / Path("best_params.json")
        with open(path_file_best_param, "r") as f:
            best_params = json.load(f)
        u.vprint("Best params are %s" % str(best_params), v)
    # Learning
    if learn:
        u.vprint("Learning", v)
        model.estimator.set_params(**best_params)
        model.fit(df_X_train, df_y_train)
    # Assessing
    if assess:
        u.vprint("Assessing phase", v)
        model.estimator.set_params(**best_params)

        u.vprint("First cross validate the ML estimator", v)
        metrics_eval = ["f1_micro", "f1_macro", "precision_micro", "precision_macro", "recall_micro", "recall_macro"]
        X_train, y_train, ids_and_dates = model._creating_matrix(df_X_train, df_y_train)
        cvr = cross_validate(model.estimator, X_train, y_train, cv=2, n_jobs=-1, scoring=metrics_eval,
                             return_train_score=True, verbose=0)
        f1_train = np.mean(cvr["train_f1"])
        recall_train = np.mean(cvr["train_recall"])
        precision_train = np.mean(cvr["train_precision"])
        accuracy_train = np.mean(cvr["train_accuracy"])
        auc_train = np.mean(cvr["train_roc_auc"])

        f1_val = np.mean(cvr["test_f1"])
        recall_val = np.mean(cvr["test_recall"])
        precision_val = np.mean(cvr["test_precision"])
        accuracy_val = np.mean(cvr["test_accuracy"])
        auc_val = np.mean(cvr["test_roc_auc"])

        res1_train = "Train : f1 = %.2f (recall = %.2f, precision = %.2f, AUC = %.2f, accuracy = %.2f)" % (f1_train,
                                                                                                    recall_train,
                                                                                                    precision_train,
                                                                                                    auc_train,
                                                                                                    accuracy_train)
        res1_val = "Validation : f1 = %.2f (recall = %.2f, precision = %.2f, AUC = %.2f, accuracy = %.2f)" % (f1_val,
                                                                                                         recall_val,
                                                                                                         precision_val,
                                                                                                         auc_val,
                                                                                                         accuracy_val)
        print(res1_train)
        print(res1_val)

        u.vprint("Now assess the model to the text level")
        model.fit(df_X_train, df_y_train)
        dates_pred_train = model.predict(df_X_train)
        date_true_train = df_y_train.values.ravel()
        accuracy_train = compute_accuracy(date_true_train, dates_pred_train)

        dates_pred_test = model.predict(df_X_test)
        date_true_test = df_y_test.values.ravel()
        accuracy_test = compute_accuracy(date_true_test, dates_pred_test)

        res2_train = "Accuracy train = %.2f" % accuracy_train
        res2_test = "Accuracy test = %.2f" % accuracy_test

        print(res2_train)
        print(res2_test)

        line = "\n".join([res1_train, res1_val, res2_train, res2_test])
        record(conf, line, "assess_results")