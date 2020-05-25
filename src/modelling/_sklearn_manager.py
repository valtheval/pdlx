from modelling._model_manager import ModelManager
from pathlib import Path
from modelling import utils_modelling as mu

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import pandas as pd


class SklearnModelManager(ModelManager):

    def set_model_params(self, model, params):
        model.set_params(**params)
        return model

    def grid_search(self, conf_model, model, X_train, y_train):
        param_grid = conf_model["param_grid"]
        if "nb_fold" in conf_model:
            cv = conf_model["nb_fold"]
        else:
            cv = 5
        if "cv_metric" in conf_model:
            scoring = conf_model["cv_metric"]
        else:
            scoring = "balanced_accuracy"
        # Search
        gs = GridSearchCV(model, param_grid=param_grid, scoring=scoring, n_jobs=-1, cv=cv, error_score=0, verbose=2)
        gs.fit(X_train, y_train)
        return gs.best_params_, gs.cv_results_


    def fit_model(self, model, X=None, y=None):
        return model.fit(X, y)


    def eval_model(self, model, X_train, X_test, y_train, y_test, df_original):
        res = mu.eval_model_returning_only_text_level_metrics(self, model, X_train, X_test, y_train, y_test, df_original)
        return res


    def make_prediction(self, model, X):
        df = self.get_intermediaire_predictions(model, X)
        df_pred = mu.intermediate_prediction_to_final_predictions(df)
        return df_pred


    def get_intermediaire_predictions(self, model, X):
        proba = model.predict_proba(X)
        df = pd.DataFrame({"ID": X[:, 0],
                           "date_possible": X[:, 1],
                           "proba0": proba[:, 0],
                           "proba1": proba[:, 1],
                           "proba2": proba[:, 2]})
        df["ID"] = df["ID"].astype(int)
        df["date_possible"] = df["date_possible"].astype(str)
        return df


class TfidfGbmManager(SklearnModelManager):

    def init_model(self, init_model):
        if (init_model is None) or isinstance(init_model, dict):
            #float_transformer = FunctionTransformer(lambda x: x.astype(float), validate=True)
            preprocessor = ColumnTransformer([("tfidf", TfidfVectorizer(), 2)], remainder="drop")
            gbm = GradientBoostingClassifier()
            model = Pipeline([("preprocessor", preprocessor), ("gbm", gbm)])
            if isinstance(init_model, dict) and len(init_model) > 0:
                model = self.set_model_params(model, init_model)
            return model
        else:
            path = Path(init_model)
            model = self.load_fitted_model(path)
            return model


class BowGbmNaNcAccidentManager(SklearnModelManager):
    """
    Classifier qui classe NC/Other. Prend en entrée le df en 1l par text et en y un array de 1/0
    """

    def init_model(self, init_model):
        if (init_model is None) or isinstance(init_model, dict):
            #float_transformer = FunctionTransformer(lambda x: x.astype(float), validate=True)
            preprocessor = ColumnTransformer([("bow", CountVectorizer(), 1)], remainder="drop")
            gbm = GradientBoostingClassifier()
            model = Pipeline([("preprocessor", preprocessor), ("gbm", gbm)])
            if isinstance(init_model, dict) and len(init_model) > 0:
                model = self.set_model_params(model, init_model)
            return model
        else:
            path = Path(init_model)
            model = self.load_fitted_model(path)
            return model

    def eval_model(self, model, X_train, X_test, y_train, y_test, df_original=None):

        if self.is_fitted_:
            pass
        else:
            model = self.fit_model(model, X_train, y_train)

        #Train
        y_pred_train = self.make_prediction(model, X_train)
        acc1 = metrics.accuracy_score(y_train, y_pred_train)
        f11 = metrics.f1_score(y_train, y_pred_train)
        precision1 = metrics.precision_score(y_train, y_pred_train)
        recall1 = metrics.recall_score(y_train, y_pred_train)

        # Test
        y_pred_test = self.make_prediction(model, X_test)
        acc2 = metrics.accuracy_score(y_test, y_pred_test)
        f12 = metrics.f1_score(y_test, y_pred_test)
        precision2 = metrics.precision_score(y_test, y_pred_test)
        recall2 = metrics.recall_score(y_test, y_pred_test)

        df_res = pd.DataFrame({"accuracy" : [acc1, acc2],
                               "f1" : [f11, f12],
                               "precision" : [precision1, precision2],
                               "recall" : [recall1, recall2]}, index=["train", "test"])
        print(df_res)
        return df_res

    def make_prediction(self, model, X):
        y = model.predict(X)
        return y


class BowGbmNaNcConsoManager(SklearnModelManager):
    """
    Classifier qui classe NC/Other. Prend en entrée le df en 1l par text et en y un array de 1/0
    """

    def init_model(self, init_model):
        if (init_model is None) or isinstance(init_model, dict):
            #float_transformer = FunctionTransformer(lambda x: x.astype(float), validate=True)
            preprocessor = ColumnTransformer([("bow", CountVectorizer(), 1)], remainder="drop")
            gbm = GradientBoostingClassifier()
            model = Pipeline([("preprocessor", preprocessor), ("gbm", gbm)])
            if isinstance(init_model, dict) and len(init_model) > 0:
                model = self.set_model_params(model, init_model)
            return model
        else:
            path = Path(init_model)
            model = self.load_fitted_model(path)
            return model

    def eval_model(self, model, X_train, X_test, y_train, y_test, df_original=None):

        if self.is_fitted_:
            pass
        else:
            model = self.fit_model(model, X_train, y_train)

        #Train
        y_pred_train = self.make_prediction(model, X_train)
        balanced_acc1 = metrics.balanced_accuracy_score(y_train, y_pred_train)
        acc1 = metrics.accuracy_score(y_train, y_pred_train)

        # Test
        y_pred_test = self.make_prediction(model, X_test)
        balanced_acc2 = metrics.balanced_accuracy_score(y_test, y_pred_test)
        acc2 = metrics.accuracy_score(y_test, y_pred_test)

        df_res = pd.DataFrame({"balanced_accuracy" : [balanced_acc1, balanced_acc2],
                               "accuracy" : [acc1, acc2]},
                              index=["train", "test"])
        print(df_res)
        return df_res

    def make_prediction(self, model, X):
        y = model.predict(X)
        return y



