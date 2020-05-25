"""
We use the data from https://www.data.gouv.fr/fr/datasets/liste-de-prenoms/#_ to improve our predictions
"""
import spacy
from collections import Counter
import utils.utils as u
from pathlib import Path
import numpy as np
import re
from sklearn.metrics import accuracy_score


class NERGenderClassifier():

    def __init__(self, model="fr_core_news_sm", top_n=3, path_name_gender="../data/raw_data/Prenoms.csv"):
        self.model = model
        self.path_name_gender = path_name_gender
        self.top_n = top_n


    def fit(self, X, y):
        self._nlp = spacy.load(self.model)
        self._dict_name_gender = self._load_dict_name_gender()
        return self


    def predict(self, X):
        y_pred = []
        for txt in X:
            gender = self._get_gender(txt)
            y_pred.append(gender)
        return np.array(y_pred)


    def predict_proba(self, X):
        raise NotImplementedError


    def _get_gender(self, txt):
        doc = self._nlp(txt)
        items = [x.text for x in doc.ents if x.label_ in ["PER", "MISC"]]
        counts = Counter(items).most_common(self.top_n)
        for entity, count in counts:
            gender = self._guess_gender_entity(entity)
            if gender is not None:
                return gender
        return "n.c."


    def _guess_gender_entity(self, s):
        if re.match(r"M\.|[Mm]onsieur|M\s", s):
            return "homme"
        elif re.search(r"[Mm]onsieur", s):
            return "homme"
        elif re.match(r"Mme\.|[Mm]adame|Mme\s|[Mm]lle|[Mm]elle", s):
            return "femme"
        elif re.search(r"[Mm]adame|[Mm]ademoiselle", s):
            return "femme"
        else:
            for word in s.split():
                word = re.sub(r"[^\w\s]", "", word).lower()
                try:
                    gender = self._dict_name_gender[word]
                    if gender == "m":
                        return "homme"
                    elif gender == "f":
                        return "femme"
                except KeyError:
                    pass

    def _load_dict_name_gender(self):
        path_file = self.path_name_gender
        df = u.load_file(path_file, encoding="latin-1")
        return dict(df[["01_prenom", "02_genre"]].values)


def main_gender_classifier(conf):
    conf_model = conf["models"]["gender_classifier"]
    v = conf_model["verbose"]

    # Preprocessing
    path_dataprep = Path(conf["paths"]["dataprep"] + "df_train.csv")
    df = u.load_file(path_dataprep)
    X = df["txt"].values
    y = df["sexe"].values

    # Get estimator
    clf = NERGenderClassifier()

    # Learning
    clf.fit(X, y) # fit doesn't use X, y actually

    # Assessing
    y_pred = clf.predict(X)
    print("Accuracy : %.2f" % accuracy_score(y, y_pred))


