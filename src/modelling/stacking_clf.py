import os
from pathlib import Path
from datetime import datetime
import utils.utils as u

from modelling import ml_multi_classes as mmc
import numpy as np

from sklearn.ensemble import StackingClassifier



def main_stacking_clf(conf):
    conf_model = conf["models"]["ml_multi_classes_ensembling"]
    v = conf_model["verbose"]
    summary = "\n******************************************* NEW SESSION *******************************************\n"
    summary += str(datetime.now()) + "\n"
    summary += str(conf_model) + "\n"
    summary += "***************************************************************************************************\n"
    u.vprint(summary, v)


    X, y = mmc.main_preprocessing_mmc(conf, conf_model, dataset="train")
    print("X shape : %s" % str(X.shape))
    print("y shape : %s" % str(y.shape))
    X_train, X_test, y_train, y_test = mmc.split_train_test(X, y)
    print("X_train shape : "+ str(X_train.shape))
    print("X_test shape : %s"%str(X_test.shape))
    print("y_train shape : %s"%str(y_train.shape))
    print("y_test shape : %s"%str(y_test.shape))


    clf = get_stacked_estimator(conf_model)
    print("Classifier")
    print(clf)
    u.vprint("Fitting model", v)
    clf.fit(X_train, y_train)

    # Assessing
    assess_summary = mmc.main_assessing(conf, conf_model, clf, X_train, X_test, y_train, y_test)
    summary += "\n" + assess_summary + "\n"


def get_stacked_estimator(conf_model):
    estimators = []
    for pipeline in conf_model["pipelines"]:
        model_name = "_".join(pipeline)
        tmp_conf_model = {"pipeline" : pipeline,
                          "path" : conf_model["path"].replace("stacking_classifier", "ml_multi_classes"),
                          "context_size" : conf_model["context_size"],
                          "verbose" : conf_model["verbose"],
                          "params" : {},
                          "search_best_params" : False}
        tmp_exp_dir = mmc.get_experiment_directory(tmp_conf_model)
        estimator = mmc.get_estimator(tmp_conf_model)
        estimator = mmc.set_estimator_params(estimator, tmp_conf_model, tmp_exp_dir)
        estimators.append((model_name, estimator))
    staked_clf = StackingClassifier(estimators = estimators)
    return staked_clf