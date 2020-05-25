import logging
from pathlib import Path
import re
import numpy as np
import pandas as pd

from modelling import utils_modelling as mu
from modelling._model_manager import ModelManager
import utils.utils as u
from preprocessing import fasttext_embedding

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
import keras
from keras.callbacks import LearningRateScheduler
from keras.models import load_model



class KerasManager(ModelManager):

    def set_model_params(self, model, params):
        return model

    def fit_model(self, model, X=None, y=None):
        conf_model = self.conf["models"][self.model_name]
        if "fit_params" in conf_model:
            fit_params = conf_model["fit_params"]
        else:
            fit_params = {}
        self.txt_info_ = X[:, 0:2]
        X = X[:, 2]
        X = self.txt_to_seq(X)
        y = keras.utils.np_utils.to_categorical(y)
        model.fit(X, y, **fit_params)
        return model

    def eval_model(self, model, X_train, X_test, y_train, y_test, df_original):
        res = mu.eval_model_returning_only_text_level_metrics(self, model, X_train, X_test, y_train, y_test, df_original)
        return res

    def make_prediction(self, model, X):
        df = self.get_intermediaire_predictions(model, X)
        df_pred = mu.intermediate_prediction_to_final_predictions(df)
        return df_pred

    def get_intermediaire_predictions(self, model, X):
        X_text = X[:,2]
        X_seq = self.txt_to_seq(X_text)
        proba = model.predict(X_seq)
        df = pd.DataFrame({"ID": X[:, 0],
                           "date_possible": X[:, 1],
                           "proba0": proba[:, 0],
                           "proba1": proba[:, 1],
                           "proba2": proba[:, 2]})
        df["ID"] = df["ID"].astype(int)
        df["date_possible"] = df["date_possible"].astype(str)
        return df

    def txt_to_seq(self, X):
        conf_model = self.conf["models"][self.model_name]
        logger = logging.getLogger(__name__)
        fte = fasttext_embedding.FasttextTransformer(strategy=None)
        fte.fit(X)
        X = fte.transform(X)
        logger.info("Text to embeddings : %s" % str(X.shape))
        X = sequence.pad_sequences(X, maxlen=conf_model["max_length_seq"], dtype="float", padding="post",
                                   truncating="post")
        logger.info("Padding sequence : %s" % str(X.shape))
        return X


class LSTMManager(KerasManager):


    def init_model(self, init_model):
        if (init_model is None) or (len(init_model) == 0):
            model = Sequential()
            model.add(LSTM(128))
            model.add(BatchNormalization())
            model.add(Dense(3, activation="softmax"))

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            print(model.summary())
            return model
        else:
            path = Path(init_model)
            model = self.load_fitted_model(path)
            print(model.summary())
            return model