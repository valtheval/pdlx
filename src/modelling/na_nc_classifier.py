"""
This module aims to provide functions to build quickly the best model to differentiate 'n.a.' and 'n.c.' in
consolidation date analysis
"""
from pathlib import Path
import os
import utils.utils as u

import numpy as np

from preprocessing.preprocessing import preprocess_text_data
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from preprocessing.fasttext_embedding import FasttextTransformer
from sklearn.model_selection import train_test_split, cross_validate
from modelling.ml_multi_classes import search_best_params, set_estimator_params
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, f1_score


def main_na_nc_classifier(conf):
    conf_model = conf["models"]["na_nc_classifier"]
    v = conf_model["verbose"]

    # Preprocessing
    path_model_df_prep = Path(conf_model["path"] + "df_train_preprocessed.csv")
    if conf_model["preprocessing"]:
        path_dataprep = Path(conf["paths"]["dataprep"] + "df_train.csv")
        df_train = u.load_file(path_dataprep)
        df_train = preprocessing_data(df_train, verbose=v)
        u.record_file(df_train, path_model_df_prep)
    else:
        df_train = u.load_file(path_model_df_prep)
    X = df_train["txt"].values
    y = df_train["date_consolidation"].values

    ### Split train, test
    u.vprint("Splitting data in train and test", v)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    ### Learning
    # Get the estimator
    u.vprint("Initializing estimator", v)
    estimator = get_estimator(conf_model)

    # Grid search
    exp_dir = get_experiment_directory(conf_model)
    if conf_model["search_best_params"]:
        u.vprint("Performing best params search", v)
        best_params = search_best_params(conf_model, estimator, X_train, y_train)
        u.record_file(best_params, exp_dir / "best_params.json")

    # Set params
    estimator = set_estimator_params(estimator, conf_model, exp_dir)

    # Learning
    path_pickle_model = exp_dir / "fitted_model.pkl"
    if conf_model["learning"]:
        estimator.fit(X_train, y_train)
        u.record_file(estimator, path_pickle_model)

    # Assessing
    res1_train, res1_val = cross_validate_model(conf_model, estimator, X_train, y_train)
    u.vprint("Cross validation results : ", v)
    print(res1_train)
    print(res1_val)
    res_test = eval_model(estimator, X_train, y_train, X_test, y_test)
    u.vprint("Test results : ", v)
    print(res_test)


def get_experiment_directory(conf_model):
    path_model = conf_model["path"]
    pipeline_label = "_".join(conf_model["pipeline"])
    dir_exp = Path(path_model + "experiments/%s/" % (pipeline_label))

    if not dir_exp.exists():
        os.makedirs(dir_exp)

    return dir_exp


def preprocessing_data(df, verbose=True):
    df = df.loc[df["date_consolidation"].isin(["n.a.", "n.c."])].copy().reset_index(drop=True)
    df = preprocess_text_data(df, verbose)
    df.loc[:, "txt"] = df["txt"].map(lambda x : " ".join(x))
    df.loc[:, "date_consolidation"] = (df["date_consolidation"].values == "na").astype(int)
    return df


def get_estimator(conf_model):
    ### Preprocessing
    # Text preprocessing
    steps = []
    if ("bow" in conf_model["pipeline"]):
        steps = [("bow", CountVectorizer())]
    if "tfidf" in conf_model["pipeline"]:
        steps.append(("tfidf", TfidfVectorizer()))
    if "fasttext" in conf_model["pipeline"]:
        steps.append(("fasttext", FasttextTransformer()))

    ### ml model
    if "gbm" in conf_model["pipeline"]:
        model = GradientBoostingClassifier()
    elif "adaboost" in conf_model["pipeline"]:
        model = AdaBoostClassifier()
    elif "rf" in conf_model["pipeline"]:
        model = RandomForestClassifier()
    elif "svc" in conf_model["pipeline"]:
        model = SVC()
    else:
        raise ValueError("Pipeline must contain one of these estimators 'gbm', 'adaboost', 'rf' or 'svc'")
    steps.append(("model", model))
    estimator = Pipeline(steps)
    return estimator


def cross_validate_model(conf_model, estimator, X, y):
    scoring = ["f1", "precision", "recall", "balanced_accuracy", "accuracy"]
    cv = conf_model["nb_fold_assessing"]
    cvr = cross_validate(estimator, X, y, cv=cv, n_jobs=-1, scoring=scoring,
                         return_train_score=True, verbose=0)
    res1_train = "Train : \n" \
                 "\t f1 = %.2f (recall = %.2f, precision = %.2f)\n" \
                 "\t Accuracy = %.2f, Balanced accuracy = %.2f" \
                 % (np.mean(cvr["train_f1"]), np.mean(cvr["train_recall"]),
                    np.mean(cvr["train_precision"]), np.mean(cvr["train_accuracy"]),
                    np.mean(cvr["train_balanced_accuracy"]))
    res1_val = "Validation : \n" \
               "\t f1 = %.2f (recall = %.2f, precision = %.2f)\n" \
               "\t Accuracy = %.2f, Balanced accuracy = %.2f" \
               % (np.mean(cvr["test_f1"]), np.mean(cvr["test_recall"]),
                  np.mean(cvr["test_precision"]), np.mean(cvr["test_accuracy"]),
                  np.mean(cvr["test_balanced_accuracy"]))
    return res1_train, res1_val


def eval_model(estimator, X_train, y_train, X_test, y_test):
    estimator.fit(X_train, y_train)
    y_test_pred = estimator.predict(X_test)
    p = precision_score(y_test, y_test_pred)
    r = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    acc = accuracy_score(y_test, y_test_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    res_test = "Test : \n" \
               "\t f1 = %.2f (recall = %.2f, precision = %.2f)\n" \
               "\t Accuracy = %.2f,  Balanced accuracy = %.2f" \
               % (f1, r, p, acc, balanced_acc)
    return res_test