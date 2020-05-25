import fasttext
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FasttextTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert document to one vector using one strategy. If no strategy define it returns for each doc
    the list of the embeddings"""

    def __init__(self, path_ft_model="fasttext_models/cc.fr.300.bin", strategy="mean"):
        self.path_ft_model = path_ft_model
        self.strategy = strategy


    def fit(self, X, y=None):
        self._ftm = fasttext.load_model(self.path_ft_model)
        return self


    def transform(self, X):
        X = X.reshape(-1, 1)
        Xemb = []
        for i in range(X.shape[0]):
            text = X[i, 0].split()
            text_emb = np.array([self._ftm.get_word_vector(word) for word in text])
            if self.strategy is None:
                pass
            else:
                text_emb = self._doc2vec(text_emb)
            Xemb.append(text_emb)
        return np.array(Xemb)


    def _doc2vec(self, doc):
        """doc est un array d'embeddings 1 ligne par mot"""
        if self.strategy == "mean":
            return np.mean(doc, axis=0)
        elif self.strategy == "min_max":
            vect_min = np.min(doc, axis=0).ravel()
            vect_max = np.max(doc, axis=0).ravel()
            return np.hstack((vect_min, vect_max))
        elif self.strategy == "sum":
            return np.sum(doc, axis=0)
        else:
            raise ValueError("strategy must be either 'mean', 'sum', or 'min_max'")