import preprocessing.preprocessing as pp
import utils.utils as u
import re

import numpy as np

import nltk
import transformers

from tqdm import tqdm


def preprocessing(conf_model, df):
    X = []
    y = []
    target = conf_model["target"]
    for i in tqdm(df.index, desc="Preprocessing"):
        id_txt = df.loc[i, "ID"]
        txt = df.loc[i, "txt"]
        txt = pp.format_text(txt)
        if target in df.columns:
            date_target = df.loc[i, target]
            date_target = re.sub(r"[-.]", "", date_target)
        for s in nltk.sent_tokenize(txt, language="french"):
            if len(s.split()) > 512: # Max length sequence for bert, we then split again using ";"
                for p in s.split(";"): # For each proposition...
                    p2 = re.sub(r'[^\w\s]', '', p)
                    idx, dates = pp.get_dates_from_token_list(p2.split())
                    if len(dates) > 0:
                        if target in df.columns:
                            if date_target in dates:
                                y.append(1)
                            else:
                                y.append(0)
                        X.append([id_txt, date_target, dates, p.capitalize()])
                    else:
                        pass
            else:
                s2 = re.sub(r'[^\w\s]', '', s)
                idx, dates = pp.get_dates_from_token_list(s2.split())
                if len(dates) > 0:
                    if target in df.columns:
                        if date_target in dates:
                            y.append(1)
                        else:
                            y.append(0)
                    X.append([id_txt, date_target, dates, s.capitalize()])
                else:
                    pass
    if target in df.columns:
        return np.array(X), np.array(y)
    else:
        return np.array(X)



