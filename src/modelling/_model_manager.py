from datetime import datetime
import re
import logging
import warnings
from pathlib import Path
import os
import pandas as pd
import numpy as np

import utils.utils as u




class ModelManager():

    def __init__(self, model_name, conf, train=False, search=False, eval=False, prod=False):
        logger = logging.getLogger(__name__)
        logger.info('Init a model manager')
        self.id = re.sub(r"[-:.\s]", "", str(datetime.now()))
        self.prod = prod
        self.search = search
        self.eval = eval
        self.train = train
        self.model_name = model_name
        self.conf = conf
        self.exp_dir = self.get_exp_dir()
        self.is_fitted_ = False
        logger.info("Experience directory : %s" % str(self.exp_dir))


    def run(self):
        logger = logging.getLogger(__name__)
        try:
            logger.info('Saving experiment context')
            self.save_exp_context()

            conf_model = self.conf["models"][self.model_name]
            # Load data
            logger.info('Load preprocessed data')
            if self.prod:
                X = self.load_preprocessed_data(conf_model["X_prod"])
                self.save_info_dataset([(X, conf_model["X_prod"])])
            if self.eval or self.train or self.search:
                X_train, X_test, y_train, y_test = self.load_preprocessed_data(conf_model["X_train"], conf_model["X_test"],
                                                                               conf_model["y_train"], conf_model["y_test"])
                X = np.vstack((X_train, X_test))
                y = np.hstack((y_train, y_test))
                self.save_info_dataset([(X_train, conf_model["X_train"]), (X_test, conf_model["X_test"]),
                                        (y_train, conf_model["y_train"]), (y_test, conf_model["y_test"])])  # Enregistrement du head(), et du path
            if self.eval or self.search:
                logger.info('Load original data from prepared data')
                df_original = self.load_prepared_data(conf_model["prepared_data"])

            # Init model
            logger.info('Init model')
            model = self.init_model(conf_model["init_model"])

            if self.search:
                logger.info('Perform grid search')
                best_params, grid_search_result = self.grid_search(conf_model, model, X_train, y_train)
                logger.info('Saving grid search results')
                self.save_grid_search_results(grid_search_result)
                logger.info('Setting best parameters found in grid search')
                model = self.set_model_params(model, params=best_params)

            if self.eval or self.search:
                logger.info('Evaluate model')
                res = self.eval_model(model, X_train, X_test, y_train, y_test, df_original)
                self.save_evaluation_results(res)

            if self.train:
                logger.info('Fit model on all data')
                self.fit_model(model, X, y)
                model.fit(X, y)
                logger.info('Saving fitted model')
                self.save_model(model)

            if self.prod:
                logger.info('Making predictions')
                df_pred = self.make_prediction(model, X)  # ID|date_conso_pred|date_accident_pred|proba 1|proba 2
                self.save_prediction(df_pred)

            self.stamp_experience("success")
            logger.info("Process completed with success")
        except BaseException as e:
            self.stamp_experience("fail")
            logger.error("Error in manager run", exc_info=True)


    def load_preprocessed_data(self, *args):
        preprocess_dir = self.conf["paths"]["preprocessed_data"]
        data = []
        for filename in args:
            data_ = u.load_file(Path(preprocess_dir) / filename)
            data.append(data_)
        if len(data) > 1:
            return data
        else:
            return data[0]

    def load_prepared_data(self, filename):
        dataprep_dir = self.conf["paths"]["dataprep"]
        return u.load_file(Path(dataprep_dir) / filename)

    def init_model(self, init_model):
        raise NotImplementedError("Please implement this function in your code")

    def set_model_params(self, model, params):
        warnings.warn("set_model_params is not implemented")
        return model

    def fit_model(self, model, X=None, y=None):
        warnings.warn("fit_model is not implemented")
        return model

    def load_fitted_model(self, path_model):
        self.is_fitted_ = True
        return u.load_file(path_model)

    def grid_search(self, conf_model, model, X_train, y_train):
        warnings.warn("grid_search is not implemented")
        return None, None

    def eval_model(self, model, X_train, X_test, y_train, y_test, df_original):
        warnings.warn("eval_model is not implemented")
        return None

    def make_prediction(self, model, X): # ID|date_conso_pred|date_accident_pred|proba 1|proba 2
        warnings.warn("make_prediction is not implemented. It must return df with ID|date_conso_pred|date_accident_pred|proba 1|proba 2")
        return None

    # functions to save objects
    def save_exp_context(self):
        u.record_file(str(self.__dict__), self.exp_dir / "context.json")

    def save_info_dataset(self, info_datasets):
        for dataset, path in info_datasets:
            name = Path(path).name
            if isinstance(dataset, pd.DataFrame):
                sample = dataset.head(min(5, len(dataset)))
            else:
                sample = pd.DataFrame(dataset[0:5])
            shape = dataset.shape
            u.record_file(sample, self.exp_dir / Path("dataset_%s_sample.csv" % name))
            info = "shape : %s\n" % str(shape) + "path : %s" % str(path)
            u.record_file(info, self.exp_dir / Path("dataset_%s_info.txt" % str(name)))


    def save_model(self, model):
        u.record_file(model, self.exp_dir / Path("model_%s.pkl" % str(self.model_name)))

    def save_evaluation_results(self, results):
        u.record_file(results, self.exp_dir / Path("results.csv"), index=True)

    def save_prediction(self, predictions):
        u.record_file(predictions, self.exp_dir / "prediction.csv")


    def save_grid_search_results(self, grid_search_results):
        u.record_file(grid_search_results, self.exp_dir / "grid_search_results.pkl")
        pass

    def get_exp_dir(self):
        mode = []
        if self.train:
            mode.append("train")
        if self.eval:
            mode.append("eval")
        if self.search:
            mode.append("search")
        if self.prod:
            mode.append("prod")
        mode.append(str(self.id))
        name_exp = "_".join(mode)

        exp_dir = Path(self.conf["paths"]["experiments"]) / str(self.model_name) / name_exp
        if not exp_dir.exists():
            os.makedirs(exp_dir)
        return exp_dir

    def stamp_experience(self, stamp):
        exp_dir = str(self.exp_dir)
        if exp_dir.endswith("/"):
            target = exp_dir[0:-1] + "_" + stamp
        else:
            target = exp_dir + "_" + stamp
        os.rename(exp_dir, target)