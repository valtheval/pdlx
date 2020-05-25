from pathlib import Path
import os
from utils import utils as u
from datetime import datetime
import pandas as pd
import numpy as np

import preprocessing.preprocessing as pp
from preprocessing.preprocessing import create_dataframe_one_line_per_date_with_context, preprocess_text_data
from preprocessing.fasttext_embedding import FasttextTransformer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, cross_validate
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_recall_fscore_support, classification_report, multilabel_confusion_matrix, \
    accuracy_score, balanced_accuracy_score


def main_ml_multi_classes_prod(conf):
    conf_model = conf["models"]["ml_multi_classes"]
    v = conf_model["verbose"]
    exp_dir = get_experiment_directory(conf_model)
    u.vprint("Conf model : ", v)
    u.vprint(conf_model, v)
    u.vprint("Exp dir : %s" % str(exp_dir))

    # First we get the estimator fitted on full data
    u.vprint("Run main preprocessing on train data", v)
    X_train, y_train = main_preprocessing_mmc(conf, conf_model, dataset="train")
    u.vprint("Fit estimator", v)
    estimator = get_estimator(conf_model)
    estimator = set_estimator_params(estimator, conf_model, exp_dir)
    estimator.fit(X_train, y_train)
    u.vprint(estimator, v)

    # Now we preprocess test data :
    u.vprint("Preprocessing full test data", v)
    X_test = main_preprocessing_mmc(conf, conf_model, dataset="test")

    # Make prediction :
    u.vprint("Doing prediction", v)
    adjust_with_nanc_classifier = conf_model["adjust_with_nanc_classifer"]
    df_y_pred_test = predict_text_level(conf_model, estimator, X_test, adjust_with_nanc_classifier, conf,
                                        dataset="test")
    u.vprint("Final prediction : ", v)
    u.vprint(df_y_pred_test)
    return df_y_pred_test


def main_ml_multi_classes_search_best_model(conf):
    conf_model = conf["models"]["ml_multi_classes"]
    v = conf_model["verbose"]
    summary = "\n******************************************* NEW SESSION *******************************************\n"
    summary += str(datetime.now()) + "\n"
    summary += str(conf_model) + "\n"

    ### Preprocessing
    X, y = main_preprocessing_mmc(conf, conf_model, dataset="train")

    ### Split train, test
    u.vprint("Splitting data in train and test", v)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

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
    summary += str(estimator.get_params()) + "\n"

    # Learning
    u.vprint("Learning phase", v)
    estimator.fit(X_train, y_train)

    # Assessing :
    u.vprint("Assessing phase", v)
    assess_summary = main_assessing(conf, conf_model, estimator, X_train, X_test, y_train, y_test)

    summary += assess_summary
    u.record_file(summary, exp_dir / "summary_results.txt", mode="a")


def main_preprocessing_mmc(conf, conf_model, dataset="train"):
    context_size = conf_model["context_size"]
    v = conf_model["verbose"]
    path_model_df_prep = Path(conf_model["path"] + "df_%s_preprocessed.csv" % dataset)
    if conf_model["preprocessing_%s" % dataset][0]:
        path_dataprep = Path(conf["paths"]["dataprep"] + "df_%s.csv" % dataset)
        df = u.load_file(path_dataprep)
        df = preprocess_text_data(df, verbose=v)
        u.record_file(df, path_model_df_prep)
    else:
        df = u.load_file(path_model_df_prep)
        df["txt"] = df["txt"].map(eval)
    # Building learning matrix
    path_model_df_learn = conf_model["path"] + "df_%s_learning_cs%d.csv" % (dataset, context_size)
    if conf_model["preprocessing_%s" % dataset][1]:
        df = create_dataframe_one_line_per_date_with_context(df, context_size, verbose=v)
        u.vprint("Recording data", v)
        u.record_file(df, path_model_df_learn)
    else:
        u.vprint("Loading data in one line per date", v)
        df = u.load_file(Path(path_model_df_learn))
    df[["pos_moy", "part_moy", "nb_app"]] = df[["pos_moy", "part_moy", "nb_app"]].astype(float)
    X = df[["txt_id", "date", "context_date", "pos_moy", "part_moy", "nb_app"]].values
    if "target" in df.columns:
        y = df["target"].astype(int).values
        u.vprint("X shape : %s" % str(X.shape), v)
        u.vprint("y shape : %s" % str(y.shape), v)
        return X, y
    else:
        u.vprint("X shape : %s" % str(X.shape), v)
        return X


def main_assessing(conf, conf_model, estimator, X_train, X_test, y_train, y_test):
    v = conf_model["verbose"]
    assess_summary = ""
    if conf_model["assessing"][0]:
        u.vprint("Cross validation", v)
        # cross validate model
        res1_val, res1_train, cvr = cross_validate_model(conf_model, estimator, X_train, y_train)
        print(res1_train)
        print(res1_val)
        assess_summary += res1_train + "\n" + res1_val
    if conf_model["assessing"][1]:
        u.vprint("Assessing model on test data at 'date level'", v)
        # Eval on test data
        res_test, report, df0, df1, df2 = eval_model(estimator, X_test, y_test)
        assess_summary += "\n" + res_test + "\nClassification report\n" + report + "\nConfusion matrix\tClasse 0\n\t" \
                          + str(df0) + "\t Classe 1\n\t" + str(df1) + "\t Classe 2\n\t" + str(df2)
        print(res_test)
        print("\nClassification report")
        print(report)
        print("\nConfusion matrix")
        print("\t Classe 0")
        print(df0)
        print("\t Classe 1")
        print(df1)
        print("\t Classe 2")
        print(df2)
    if conf_model["assessing"][2]:
        u.vprint("Assessing model on text level", v)
        # Assessing at text level
        res3 = eval_model_text_level(conf_model, estimator, X_train, X_test, conf)
        print(res3)
        assess_summary += "\n" + res3 + "\n"
    return assess_summary



def cross_validate_model(conf_model, estimator, X, y):
    scoring = ["f1_micro", "f1_macro", "recall_micro", "recall_macro", "precision_micro", "precision_macro", "accuracy",
               "balanced_accuracy"]
    cv = conf_model["nb_fold_assessing"]
    cvr = cross_validate(estimator, X, y, cv=cv, n_jobs=-1, scoring=scoring,
                         return_train_score=True, verbose=0)
    res1_train = "Train : \n" \
                 "\t Micro : f1 = %.2f (recall = %.2f, precision = %.2f)\n" \
                 "\t Macro : f1 = %.2f (recall = %.2f, precision = %.2f)\n" \
                 "\t Accuracy = %.2f, Balanced accuracy = %.2f" \
                 % (np.mean(cvr["train_f1_micro"]), np.mean(cvr["train_recall_micro"]),
                    np.mean(cvr["train_precision_micro"]),
                    np.mean(cvr["train_f1_macro"]), np.mean(cvr["train_recall_macro"]),
                    np.mean(cvr["train_precision_macro"]), np.mean(cvr["train_accuracy"]),
                    np.mean(cvr["train_balanced_accuracy"]))
    res1_val = "Validation : \n" \
               "\t Micro : f1 = %.2f (recall = %.2f, precision = %.2f)\n" \
               "\t Macro : f1 = %.2f (recall = %.2f, precision = %.2f)\n" \
               "\t Accuracy = %.2f, Balanced accuracy = %.2f" \
               % (np.mean(cvr["test_f1_micro"]), np.mean(cvr["test_recall_micro"]),
                  np.mean(cvr["test_precision_micro"]),
                  np.mean(cvr["test_f1_macro"]), np.mean(cvr["test_recall_macro"]),
                  np.mean(cvr["test_precision_macro"]), np.mean(cvr["test_accuracy"]),
                  np.mean(cvr["test_balanced_accuracy"]))
    return res1_val, res1_train, cvr


def eval_model(estimator, X, y):
    y_test_pred = estimator.predict(X)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y, y_test_pred, average="micro")
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y, y_test_pred, average="macro")
    acc = accuracy_score(y, y_test_pred)
    balanced_acc = balanced_accuracy_score(y, y_test_pred)
    report = classification_report(y, y_test_pred)
    res_test = "Test : \n" \
               "\t Micro : f1 = %.2f (recall = %.2f, precision = %.2f)\n" \
               "\t Macro : f1 = %.2f (recall = %.2f, precision = %.2f)\n" \
               "\t Accuracy = %.2f,  Balanced accuracy = %.2f" \
               % (f1_micro, r_micro, p_micro, f1_macro, r_macro, p_macro, acc, balanced_acc)
    mcm = multilabel_confusion_matrix(y, y_test_pred, labels=estimator.classes_)
    df0 = pd.DataFrame(mcm[0], columns=["N", "P"], index=["Pred_N", "Pred_P"])
    df1 = pd.DataFrame(mcm[1], columns=["N", "P"], index=["Pred_N", "Pred_P"])
    df2 = pd.DataFrame(mcm[2], columns=["N", "P"], index=["Pred_N", "Pred_P"])
    return res_test, report, df0, df1, df2


def eval_model_text_level(conf_model, estimator, X_train, X_test, conf=None):
    adjust_with_nanc_classifier = conf_model["adjust_with_nanc_classifer"]

    path_model_df_prep = Path(conf_model["path"] + "df_train_preprocessed.csv")
    df = u.load_file(path_model_df_prep, usecols=["ID", "date_accident", "date_consolidation"])
    df = df.rename(columns={"ID": "txt_id"})

    df_y_pred_test = predict_text_level(conf_model, estimator, X_test, adjust_with_nanc_classifier, conf)
    df_y_test = df[df["txt_id"].isin(df_y_pred_test["txt_id"])].drop_duplicates().copy()
    acc1_test, acc2_test, mean_acc_test = compute_accuracy(df_y_test, df_y_pred_test)

    df_y_pred_train = predict_text_level(conf_model, estimator, X_train, adjust_with_nanc_classifier, conf)
    df_y_train = df[df["txt_id"].isin(df_y_pred_train["txt_id"])].drop_duplicates().copy()
    acc1_train, acc2_train, mean_acc_train = compute_accuracy(df_y_train, df_y_pred_train)
    res3 = "Train : accuracy = %.2f (accident = %.2f, consolidation = %.2f)\n" \
           "Test :  accuracy = %.2f (accident = %.2f, consolidation = %.2f)" % \
           (mean_acc_train, acc1_train, acc2_train, mean_acc_test, acc1_test, acc2_test)
    return res3


def split_train_test(X, y):
    gsp = GroupShuffleSplit(test_size=.25, n_splits=2, random_state=7)
    train_inds, test_inds = next(gsp.split(X, y, groups=X[:, 0]))
    X_train = X[train_inds, :]
    X_test = X[test_inds, :]
    y_train = y[train_inds]
    y_test = y[test_inds]
    return X_train, X_test, y_train, y_test


def search_best_params(conf_model, estimator, X, y):
    param_grid = conf_model["param_grid"]
    cv = conf_model["nb_fold"]
    scoring = conf_model["search_metric"]
    # Search
    gs = GridSearchCV(estimator, param_grid=param_grid, scoring=scoring, n_jobs=-1, cv=cv, error_score=0, verbose=2)
    gs.fit(X, y)
    return gs.best_params_


def set_estimator_params(estimator, conf_model, exp_dir):
    verbose = conf_model["verbose"]
    path_best_params = exp_dir / "best_params.json"
    if (len(conf_model["params"]) != 0) and (not conf_model["search_best_params"]):
        u.vprint("Using param from conf file : \n %s" % (str(conf_model["params"])), verbose)
        estimator.set_params(**conf_model["params"])
    elif path_best_params.exists():
        u.vprint("Loading best params", verbose)
        best_params = u.load_file(exp_dir / "best_params.json")
        u.vprint(str(best_params), verbose)
        estimator.set_params(**best_params)
    else:
        # Default parameters
        u.vprint("Using default params", verbose)
        u.vprint(str(estimator.get_params()), verbose)
    return estimator

def to_float(x):
    return x.astype(float)

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
    text_preprocessor = Pipeline(steps)

    # Float processing
    float_transformer = FunctionTransformer(to_float, validate=True)

    preprocessor = ColumnTransformer([("textprep", text_preprocessor, 2), ("typing", float_transformer, [3, 4, 5])])

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
    estimator = Pipeline([("preprocessor", preprocessor), ("model", model)])
    return estimator


def get_experiment_directory(conf_model):
    path_model = conf_model["path"]
    context_size = conf_model["context_size"]
    pipeline_label = "_".join(conf_model["pipeline"])
    dir_exp = Path(path_model + "experiments/cs%s_%s/" % (str(context_size), pipeline_label))

    if not dir_exp.exists():
        os.makedirs(dir_exp)

    return dir_exp


def predict_text_level(conf_model, estimator, X, adjust_with_nanc_classifier=False, conf=None, dataset="train"):
    th_nc_accident = conf_model["th_nc_accident"]
    th_nc_consolidation = conf_model["th_nc_consolidation"]
    th_na_consolidation = conf_model["th_na_consolidation"]
    proba = estimator.predict_proba(X)
    df = pd.DataFrame({"txt_id" : X[:,0],
                       "date" : X[:,1],
                       "proba0" : proba[:, 0],
                       "proba1" : proba[:, 1],
                       "proba2" : proba[:, 2]})
    df["rank1"] = df.groupby("txt_id")["proba1"].rank("dense", ascending = False)
    df["rank2"] = df.groupby("txt_id")["proba2"].rank("dense", ascending = False)

    df1 = df[df["rank1"] == 1].copy().drop_duplicates(subset = ["txt_id"])
    df1.loc[df1["proba1"] < th_nc_accident , "date"] = "nc"
    df1 = df1[["txt_id", "date"]]

    df2 = df[df["rank2"] == 1].copy().drop_duplicates(subset=["txt_id"])
    if th_nc_consolidation < th_na_consolidation:
        df2.loc[df2["proba2"] < th_nc_consolidation, "date"] = "nc"
        df2.loc[(df2["proba2"] > th_nc_consolidation) & (df2["proba2"] < th_na_consolidation), "date"] = "na"
    else:
        df2.loc[df2["proba2"] < th_na_consolidation, "date"] = "na"
        df2.loc[(df2["proba2"] > th_na_consolidation) & (df2["proba2"] < th_nc_consolidation), "date"] = "nc"
    df2 = df2[["txt_id", "date"]]

    df1 = df1.rename(columns = {"date" : "date_accident_pred"})
    df2 = df2.rename(columns = {"date": "date_consolidation_pred"})
    df1 = df1.merge(df2, on = "txt_id")

    if adjust_with_nanc_classifier:
        print("Adjusting with nan classifer")
        df1 = adjust_prediction_with_nanc_classifier(conf, conf_model, df1, dataset)

    return df1


def adjust_prediction_with_nanc_classifier(conf, conf_model, df_pred_mmc, dataset="train"):
    verbose = conf_model["verbose"]
    path_model_classif = conf_model["path_nan_classifier"]
    nanc_classifer = u.load_file(Path(path_model_classif))

    # Load dataprep reduced to where mmc has predicted na or nc
    initial_columns = df_pred_mmc.columns
    ids_pred_nanc = df_pred_mmc.loc[df_pred_mmc["date_consolidation_pred"].isin(["na", "nc"]), "txt_id"]
    # Get raw data
    path_dataprep = Path(conf["paths"]["dataprep"] + "df_%s.csv" % dataset)
    df = u.load_file(path_dataprep)
    df = df.loc[df["ID"].isin(ids_pred_nanc)]

    # Preprocesse this data the way na_nc_classifier works
    df_prep = preprocess_text_data(df, verbose)
    df_prep.loc[:, "txt"] = df["txt"].map(lambda x: " ".join(x))
    X = df_prep["txt"].values

    # Predict
    y_pred = nanc_classifer.predict(X)

    # Modify inital mmc prediction
    df_pred_nanc = pd.DataFrame({"txt_id" : df_prep["ID"].values, "nanc_pred" : y_pred})
    df_pred_mmc = df_pred_mmc.merge(df_pred_nanc, on="txt_id", how="left")
    df_pred_mmc["date_consolidation_pred"] = df_pred_mmc.apply(final_pred, axis=1)
    return df_pred_mmc[initial_columns]


def final_pred(x):
    if x["date_consolidation_pred"] in ["na", "nc"]:
        index = int(x["nanc_pred"]) # predict 0 or 1 if its nc or na
        l = ["nc", "na"]
        return l[index]
    else:
        return x["date_consolidation_pred"]


def compute_accuracy(df_y_true, df_y_pred):
    """Dataframes must contain id, date_accident and date_consolidation. 1 line per text. no duplicates on id"""
    df_y_true = df_y_true.sort_values("txt_id")
    df_y_pred = df_y_pred.sort_values("txt_id")

    if all(a == b for a, b in zip(df_y_true["txt_id"].values, df_y_pred["txt_id"].values)):
        pass
    else:
        raise ValueError("df_y_true and df_y_pred do not contain the same text ids")

    y1_true = df_y_true["date_accident"].values.astype(str)
    y1_pred = df_y_pred["date_accident_pred"].values.astype(str)
    y2_true = df_y_true["date_consolidation"].values.astype(str)
    y2_pred = df_y_pred["date_consolidation_pred"].values.astype(str)
    acc1 = (y1_pred == y1_true).sum() / y1_pred.shape[0]
    acc2 = (y2_pred == y2_true).sum() / y2_pred.shape[0]
    return acc1, acc2, np.mean([acc1, acc2])


########################################################################################################################
# Post prediction analysis function
########################################################################################################################

def get_analysis_report(id_txt, df_raw, df_full_pred, context_size, target = "date_accident"):
    if target == "date_accident":
        t = 1
    else:
        t = 2
    text_name = df_raw.loc[df_raw["ID"] == id_txt, "filename"].values[0]
    true_date = df_raw.loc[df_raw["ID"] == id_txt, target].values[0]
    pred_date = df_full_pred.loc[df_full_pred["txt_id"] == id_txt, target + "_pred"].values[0]
    if str(true_date) in df_full_pred.loc[df_full_pred["txt_id"].astype(int) == int(id_txt), "date"].values.astype(str):
        rank_true_date = df_full_pred.loc[(df_full_pred["txt_id"].astype(int) == int(id_txt)) &
                                          (df_full_pred["date"].astype(str) == str(true_date)), "rank%d"%t].values[0]
    else:
        rank_true_date = "Date not found"

    print("************ Target : %s ************"%(target.replace("_", " ").upper()))
    print("Texte : %s" % text_name)
    print("Predicted date : %s" % str(pred_date))
    print("True date : %s" % str(true_date))
    print("Rank of the true date : %s" % str(rank_true_date))
    print("\nProbabilities of the different dates : ")
    print(df_full_pred.loc[df_full_pred["txt_id"] == id_txt, ["txt_id", "date", "proba0", "proba1", "proba2", "rank1", "rank2"]])

    print("\nContext of the predicted date : ")
    txt = eval(df_raw.loc[df_raw["ID"] == id_txt, "txt"].values[0])
    print(pp.get_context_date(context_size, str(pred_date), txt))

    print("\nContext of the true date : ")
    print(pp.get_context_date(context_size, str(true_date), txt))
    return txt


def get_df_full_prediction(conf, estimator, X, dataset="train"):
    conf_model = conf["models"]["ml_muti_classes"]
    adjust_prediction = conf_model["adjust_with_nanc_classifer"]
    proba = estimator.predict_proba(X)
    df = pd.DataFrame({"txt_id" : X[:,0],
                       "date" : X[:,1],
                       "proba0" : proba[:, 0],
                       "proba1" : proba[:, 1],
                       "proba2" : proba[:, 2]})
    df["rank1"] = df.groupby("txt_id")["proba1"].rank("dense", ascending = False)
    df["rank2"] = df.groupby("txt_id")["proba2"].rank("dense", ascending = False)
    df_true = get_true_dates(conf_model)
    th_nc_accident = conf_model["th_nc_accident"]
    th_nc_consolidation = conf_model["th_nc_consolidation"]
    th_na_consolidation = conf_model["th_na_consolidation"]
    df_pred_dates = predict_text_level(estimator, X, th_nc_accident, th_nc_consolidation, th_na_consolidation,
                                       adjust_prediction, conf, dataset)
    df = df.merge(df_true, on="txt_id", how="left")
    df = df.merge(df_pred_dates, on="txt_id", how="left")
    return df


def get_true_dates(conf_model):
    path_model_df_prep = Path(conf_model["path"] + "df_train_preprocessed.csv")
    df = u.load_file(path_model_df_prep, usecols=["ID", "date_accident", "date_consolidation"])
    df = df.rename(columns={"ID": "txt_id"})
    return df


def get_rank_true_dates(df_full_pred, list_txt_id, target="date_accident"):
    if target == "date_accident":
        t = 1
    else:
        t = 2
    ranks = []
    for text_id in list_txt_id:
        true_date = df_full_pred.loc[df_full_pred["txt_id"] == text_id, target].values[0]
        if str(true_date) in df_full_pred.loc[df_full_pred["txt_id"].astype(int) == int(text_id), "date"].values.astype(str):
            rank_true_date = df_full_pred.loc[(df_full_pred["txt_id"].astype(int) == int(text_id)) &
                                              (df_full_pred["date"].astype(str) == str(true_date)), "rank%d" % t].values[0]
        else:
            rank_true_date = -1
        ranks.append(rank_true_date)
    return ranks











