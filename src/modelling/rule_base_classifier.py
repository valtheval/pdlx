import re
import numpy as np
from collections import Counter
from pathlib import Path
import pandas as pd
from preprocessing.preprocessing import normalize, format_date, get_dates_from_token_list
import preprocessing.preprocessing as pp
import nltk
from utils.utils import get_all_combi_grid_search, vprint
import json
import pickle


class RuleBaseClassifier():

    def __init__(self, rule_size=10, context_size=5, exclude_words=None):
        self.rule_size = rule_size
        self.context_size = context_size
        self.exclude_words = exclude_words
        self.rule = None


    def set_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


    def fit(self, X, y):
        """X est un numpy array de liste de mots (1 ligne par text)"""
        left_context, right_context = self.get_contexts_target(X, y)
        rule = self.create_rule(left_context, right_context)
        self.rule = rule


    def predict(self, X):
        predictions = []
        proba = []
        all_predictions = [] #prediction for each date in the text
        for i in range(X.shape[0]):
            print("\n\n")
            txt = X[i]
            index_dates, dates = get_dates_from_token_list(txt)
            pred_1_txt = []
            for date in dates:
                p_date = self.get_probability(date, txt)
                pred_1_txt.append((date, p_date))
            best_date, best_proba = max(pred_1_txt, key=lambda x: x[1])
            all_predictions.append(pred_1_txt)
            predictions.append(best_date)
            proba.append(best_proba)

        self.proba = proba
        self.all_predictions = all_predictions

        return np.array(predictions)


    def get_context_date(self, date, txt, idxs_date=None):
        """
        For a date it returns both left and right context found in text. If the date appears more than once in the text\
        multiple left and right contexts are returned.
        """
        left_context, right_context = pp.get_context_date(self.context_size, date, txt, idxs_date)
        return left_context, right_context


    def get_probability(self, date, txt):
        """Return the probability of the date for being a target date"""
        l, r = self.get_context_date(date, txt)
        proba = []
        for i in range(len(l)):
            full_context = np.hstack((l[i], r[i]))
            p = np.intersect1d(full_context, np.array(self.rule)).shape[0] / full_context.shape[0]
            proba.append(p)
        return np.mean(proba)


    def get_contexts_target(self, X, y):
        """It returns all the contexts found for target dates. It returns 2 lists (left, right) of list of strings"""
        left_contexts = []
        right_contexts = []
        for i in range(X.shape[0]):
            s = X[i] #text
            d = y[i] #target date
            idxs = np.where(np.array(s) == d)[0]
            for idx in idxs:
                left_bound = max(idx - self.context_size, 0)
                right_bound = min(idx + self.context_size + 1, len(s))
                left_contexts.append(s[left_bound:idx])
                right_contexts.append(s[(idx + 1):right_bound])
        return left_contexts, right_contexts


    def create_rule(self, left_context, right_context):
        """From left and right contexts observed around target dates it create a rule which is the most "frequent" \
        context found"""
        if self.exclude_words is None:
            self.exclude_words = []
        c = Counter()
        c.update(sum(left_context, []))
        c.update(sum(right_context, []))
        rule = []
        i = 0
        most_common = c.most_common(len(c.items()))
        while (len(rule) < self.rule_size) and (i < len(most_common)):
            word, count = most_common[i]
            if word in self.exclude_words:
                i += 1
            else:
                rule.append(word)
                i += 1
        return rule


def preprocess_data(conf, dataset="train"):
    conf_model = conf["models"]["rule_base_classifier"]
    v = conf_model["verbose"]
    preprocessing = conf_model["preprocessing"]
    if preprocessing:
        path_dataprep = Path(conf["paths"]["dataprep"] + "df_%s.csv"%dataset)
        vprint("Loading...", v)
        df = pd.read_csv(path_dataprep, sep=';', encoding="utf-8")

        vprint("Lowercase text...", v)
        df["txt"] = df["txt"].str.lower()
        vprint("Format dates in text...", v)
        df["txt"] = df["txt"].map(format_date)
        vprint("Format date accident and consolidation...", v)
        df["date_accident"] = df["date_accident"].map(lambda x: re.sub(r"[-.]", "", x))
        df["date_consolidation"] = df["date_consolidation"].map(lambda x: re.sub(r"[-.]", "", x))
        vprint("Tokenize words...", v)
        df["txt"] = df["txt"].map(nltk.word_tokenize)
        vprint("Normalize text...", v)
        df["txt"] = df["txt"].map(normalize)

        vprint("Record...", v)
        path_rule_base_classifier = Path(conf_model["path"] + "df_%s_preprocessed.csv"%dataset)
        df.to_csv(path_rule_base_classifier, sep=';', encoding="utf-8", index=False)
        vprint("Preprocessing completed", v)
    else:
        path_rule_base_classifier = Path(conf_model["path"] + "df_%s_preprocessed.csv" % dataset)
        df = pd.read_csv(path_rule_base_classifier, sep=';', encoding="utf-8")
        df["txt"] = df["txt"].map(eval)
    return df


def split_x_y(conf, date="date_accident", df=None, dataset="train"):
    if df is None:
        conf_model = conf["models"]["rule_base_classifier"]
        path_rule_base_classifier = Path(conf_model["path"] + "df_%s_preprocessed.csv" % dataset)
        df = pd.read_csv(path_rule_base_classifier, sep=';', encoding="utf-8")
    X = df["txt"].values
    y = df[date].values
    return X, y


def search(conf, X, y):
    conf_model = conf["models"]["rule_base_classifier"]
    grid_search = conf_model["grid_search"]
    v = conf_model["verbose"]

    all_combi = get_all_combi_grid_search(grid_search)
    nb_combi = len(all_combi)

    rbclf = RuleBaseClassifier()
    res = []
    best_params = None
    best_metric = None
    for i, params in enumerate(all_combi):
        vprint("%d / %d" % (i + 1, nb_combi), v)
        vprint("\t%s" % (str(params)), v)
        rbclf.set_params(params)
        rbclf.fit(X, y)
        y_pred = rbclf.predict(X)
        accuracy = compute_accuracy(y, y_pred)
        if best_metric is None:
            best_metric = accuracy
            best_params = params
        elif best_metric < accuracy:
            best_metric = accuracy
            best_params = params
        vprint("\tAccuracy = %.2f" % accuracy)
        all_predictions = rbclf.all_predictions
        proba = rbclf.proba
        res.append((params, y_pred, all_predictions, proba))
    path_model_params = Path(conf_model["path"] + "best_params.json")
    with open(path_model_params, "w") as f:
        json.dump(best_params, f)
    return best_params


def compute_accuracy(y_true, y_pred):
    y_compar = y_pred == y_true
    accuracy = (y_compar.sum() / y_compar.shape[0])
    return accuracy


def main_rule_base_classifier(conf):
    conf_model = conf["models"]["rule_base_classifier"]
    date_target = conf_model["date_target"]
    learn = conf_model["learn"]
    do_search = conf_model["search"]
    assess = conf_model["assess"]
    v = conf_model["verbose"]
    path_model = conf_model["path"]
    path_pickle = Path(path_model + "model.pkl")

    df = preprocess_data(conf, dataset="train")
    X_train, y_train = split_x_y(conf, date=date_target, df=df, dataset="train")
    # Get best params
    if do_search:
        vprint("Searching best param", v)
        best_params = search(conf, X_train, y_train)
        vprint("Best params are %s"%str(best_params), v)
    else:
        vprint("Reading best params in model folder", v)
        path_model_params = Path(conf_model["path"] + "best_params.json")
        with open(path_model_params, "r") as f:
            best_params = json.load(f)
    # Learning
    if learn:
        vprint("Learning phase", v)
        model = RuleBaseClassifier()
        model.set_params(best_params)
        model.fit(X_train, y_train)
        with open(path_pickle, "wb") as f:
            pickle.dump(model, f)
    # Assessing
    if assess:
        vprint("Assessing phase", v)
        df_test = preprocess_data(conf, dataset="test")
        X_test, y_test = split_x_y(conf, date=date_target, df=df_test, dataset="test")
        with open(path_pickle, "rb") as f:
            model = pickle.load(f)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        accuracy_train = compute_accuracy(y_train, y_train_pred)
        accuracy_test = compute_accuracy(y_test, y_test_pred)
        print("Accuracy train = %.2f" % accuracy_train)
        print("Accuracy test = %.2f" % accuracy_test)











