import utils.utils as u
import re
import nltk
import unicodedata
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm


def create_dataframe_one_line_per_date_with_context(df, context_size, verbose=False):
    """df is a preprocessed dataframe in one line per text with information on each text (date_accident, date conso,
    ID...). It returns a data frame in one line per date with target as 1 for date_accident, 2 for date_conso, 0 otherwise"""
    u.vprint("Creating dataframe in one line per date with context size of %d"%context_size, verbose)
    X = []
    all_txt = df["txt"].values
    all_txt_id = df["ID"].values
    if ("date_accident" in df.columns) and ("date_consolidation" in df.columns):
        date_accident = df["date_accident"].values
        date_conso = df["date_consolidation"].values
    for i in range(all_txt.shape[0]):
        txt = all_txt[i]
        txt_id = all_txt_id[i]
        index_dates, dates_in_txt = get_dates_from_token_list(txt)
        for d in list(set(dates_in_txt)):
            left_context, right_context = get_context_date(context_size, d, txt)
            l = sum([list(c) for c in left_context], []) + sum([list(c) for c in right_context], [])
            s = " ".join(l)
            positions = index_dates[np.argwhere(dates_in_txt == d)].ravel()
            positions_mean = np.mean(positions)
            part_of_txt = positions / len(txt)
            part_of_txt_mean = np.mean(part_of_txt)
            nb_appearances = len(positions)
            # Reducing set with rules discovered with exploratory analysis
            if ("date_accident" in df.columns) and ("date_consolidation" in df.columns):
                if 0 in positions:
                    # The target date can't be the first word of the text
                    pass
                else:
                    if d == date_accident[i]:
                        y_target = 1
                    elif d == date_conso[i]:
                        y_target = 2
                    else:
                        y_target = 0
                    X.append([txt_id, d, s, positions_mean, part_of_txt_mean, nb_appearances, y_target])
            else:
                X.append([txt_id, d, s, positions_mean, part_of_txt_mean, nb_appearances])
    if ("date_accident" in df.columns) and ("date_consolidation" in df.columns):
        df_out = pd.DataFrame(X, columns=["txt_id", "date", "context_date", "pos_moy", "part_moy", "nb_app", "target"])
    else:
        df_out = pd.DataFrame(X, columns=["txt_id", "date", "context_date", "pos_moy", "part_moy", "nb_app"])
    u.vprint("Dataframe completed.", verbose)
    return df_out


def create_one_line_per_date_hard_text_formatting_and_sent_token(df, min_len_sent=2, max_len_sent=512, verbose=False):
    X = []
    y = []
    target1 = "date_accident"
    target2 = "date_consolidation"
    is_train = (target1 in df.columns) and (target2 in df.columns)
    for i in tqdm(df.index, desc="Preprocessing"):
        id_txt = df.loc[i, "ID"]
        txt = df.loc[i, "txt"]
        txt = format_text_updated(txt)
        txt = remove_obvious_dates(txt)
        sentences = nltk.sent_tokenize(txt, language="french")
        if is_train:
            date_target1 = df.loc[i, target1]
            date_target2 = df.loc[i, target2]
            date_target1 = re.sub(r"[-.]", "", date_target1)
            date_target2 = re.sub(r"[-.]", "", date_target2)
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence.lower(), "french")
            tokens = remove_stopwords(tokens)
            list_one_date_phrases = separate_dates_in_sentences(tokens, [], min_len=min_len_sent, max_len=max_len_sent)
            for phrase in list_one_date_phrases:
                idx, dates = get_dates(phrase)
                if len(dates) > 0: # There should be one date only
                    X.append([id_txt, dates[0], phrase])
                    if is_train:
                        if (date_target1 in dates) and (date_target1 != "nc"):
                            y.append(1)
                        elif (date_target2 in dates) and (not date_target2 in ["na", "nc"]):
                            y.append(2)
                        else:
                            y.append(0)
                else:
                    pass
    X = np.array(X)
    if is_train:
        y = np.array(y)
        return X, y
    else:
        return X



def separate_dates_in_sentences(token_list, output, max_len=512, min_len=2):
    if "." in token_list:
        token_list.remove(".")
    index_dates, dates = get_dates_from_token_list(token_list)
    set_dates = set(dates)
    if len(set_dates) == 0:
        return output
    else:
        if len(set_dates) == 1:
            if len(token_list) <= max_len:
                if len(token_list) >= min_len:
                    output.append(" ".join(token_list))
                return output
            else:
                left_bound = max(0, int(index_dates[0] - (max_len/2)))
                right_bound = min(len(token_list), int(index_dates[0] + (max_len)/2))
                output.append(" ".join(token_list[left_bound:right_bound]))
                return output
        else: # len(index_dates) > 1
            if ";" in token_list:
                index = token_list.index(";")
                right_list = token_list[0:index]
                left_list = token_list[index+1::]
            else:
                median = int((index_dates[0] + index_dates[1])/2)
                if median == index_dates[0]: #Les dates se suivent
                    right_list = token_list[0:(median+1)]
                    left_list = token_list[(median+1)::]
                else:
                    right_list = token_list[0:median]
                    left_list = token_list[median::]
            separate_dates_in_sentences(right_list, output, max_len)
            separate_dates_in_sentences(left_list, output, max_len)
            return output


def create_one_line_per_date_sent_tokenize(df, min_len_sent=10, verbose = False):
    """Creating matrix with one line per date base on sentence tokenizer helped with regex splitting on ':', ';', '-'
    if necesary"""
    X = []
    y = []
    target1 = "date_accident"
    target2 = "date_consolidation"
    is_train = (target1 in df.columns) and (target2 in df.columns)
    for i in tqdm(df.index, desc="Preprocessing"):
        id_txt = df.loc[i, "ID"]
        txt = df.loc[i, "txt"]
        txt = format_text(txt)
        txt = remove_obvious_dates(txt)
        if is_train:
            date_target1 = df.loc[i, target1]
            date_target2 = df.loc[i, target2]
            date_target1 = re.sub(r"[-.]", "", date_target1)
            date_target2 = re.sub(r"[-.]", "", date_target2)

        sent_token = tokenize_sentences(txt)
        for i, sent in enumerate(sent_token):
            if len(sent) > min_len_sent:
                idx, dates = get_dates(sent)
                if len(dates) > 0: # There should be one date only
                    X.append([id_txt, dates[0], sent])
                    if is_train:
                        if (date_target1 in dates) and (date_target1 != "nc"):
                            y.append(1)
                        elif (date_target2 in dates) and (not date_target2 in ["na", "nc"]):
                            y.append(2)
                        else:
                            y.append(0)
                else:
                    pass
            else:
                pass
    X = np.array(X)
    if is_train:
        y = np.array(y)
        return X, y
    else:
        return X


def create_1l_per_text_for_nanc_classifier(df):
    X = []
    y_accident = []
    y_conso = []
    target1 = "date_accident"
    target2 = "date_consolidation"
    is_train = (target1 in df.columns) and (target2 in df.columns)
    for i in tqdm(df.index, desc="Preprocessing"):
        id_txt = df.loc[i, "ID"]
        txt = df.loc[i, "txt"]
        txt = format_text(txt)
        X.append([id_txt, txt])
        if is_train:
            date_target1 = df.loc[i, target1]
            date_target2 = df.loc[i, target2]
            date_target1 = re.sub(r"[-.]", "", date_target1)
            date_target2 = re.sub(r"[-.]", "", date_target2)
            if date_target1 == "nc":
                y_accident.append(1)
            else:
                y_accident.append(0)
            if date_target2 == "nc":
                y_conso.append(1)
            elif date_target2 == "na":
                y_conso.append(2)
            else:
                y_conso.append(0)
    X = np.array(X)
    if is_train:
        y_accident = np.array(y_accident)
        y_conso = np.array(y_conso)
        return X, y_accident, y_conso
    else:
        return X



def tokenize_sentences(txt):
    # Tokenize with nltk classic tokenizer
    sents0 = nltk.sent_tokenize(txt, language="french")
    # Tokenize the remaining sentence with DATE. (YYYYMMDD. )
    dates_regex = re.compile(r"((19|20)([0-9][0-9])(0[1-9]|1[012])(0[1-9]|[12][0-9]|3[01])\. )")
    sents1 = []
    for s in sents0:
        # We need to keep the delimiter date inside each split
        delimiters = [e[0] for e in dates_regex.findall(s)]
        if len(delimiters) > 0:
            re_tmp = re.compile(r"|".join(delimiters))
            s_split = re_tmp.split(s)
            delimiters.append("")
            sents1.extend([s_split[i] + delimiters[i] for i in range(len(delimiters))])
        else:
            sents1.append(s)
    # Tokenize sentence according to punctuation ';', ':', '-'
    sents2 = []
    re_punct = re.compile(r"[;:-]")
    for s in sents1:
        idx, dates = get_dates(s)
        if len(dates) > 1:
            s_split2 = re_punct.split(s)
            if len(s_split2) > 1:
                sents2.extend(s_split2)
            else:
                sents2.append(s)
        else:
            sents2.append(s)
    # If there are still sentences with several dates we separate text with one date in each part.
    sents3 = []
    for s2 in sents2:
        idx, dates = get_dates(s2)
        if len(dates) > 1:
            list_parts = separate_dates(s2, idx = idx)
            sents3.extend(list_parts)
        else:
            sents3.append(s2)
    return sents3


def separate_dates(txt, idx=None):
    tokens = re.sub(r'[^\w\s]', '', txt).split()
    if idx is None:
        idx, dates = get_dates(txt)
    len_idx = len(idx)
    parts = []
    for i in range(len_idx):
        if i == 0:
            start = 0
            end = idx[i+1]
        elif i < (len_idx - 1):
            start = idx[i-1]+1
            end = idx[i+1]
        else:
            start = idx[i-1]+1
            end = len(tokens)
        parts.append(" ".join(tokens[start:end]))
    return parts


def format_text(txt):
    txt = format_amount_money(txt)  # This avoid bad sentence tokenization
    txt = txt.lower()
    txt = re.sub(r" 1 er", " 1er", txt)
    txt = format_date(txt)
    txt = re.sub(r"(\*\s?)+", "", txt)
    txt = re.sub(r"(--+\s?)", "", txt)
    return txt

def format_text_updated(txt):
    txt = format_amount_money(txt)  # This avoid bad sentence tokenization
    txt = remove_useless_dots(txt)
    txt = remove_accronymes(txt)
    sentences = nltk.sent_tokenize(txt, language="french")
    formated_sentences = []
    for s in sentences:
        s = s.lower()
        s = re.sub(r" 1 er", " 1er", s)
        s = format_date(s)
        s = re.sub(r"(\*\s?)+", "", s)
        s = re.sub(r"(--+\s?)", "", s)
        s = re.sub(r"\.{3}$|\.$", "", s)
        s = re.sub(r"[^\w\s\.\;€]", "", s)
        s = re.sub(r"\s\s+", " ", s)
        formated_sentences.append(s.capitalize())
    txt = ". ".join(formated_sentences)
    return txt


def remove_obvious_dates(txt):
    """For text lowered and date formatted (YYYYMMDD) we remove dates that are useless 'né le YYYYMMDD',
    'la loi du YYYYMMDD'"""
    regex = re.compile(r"((tribunal|conseil de prudhommes)( de grande instance)?( de commerce)?( des affaires de sécurité sociale)?( correctionnel)?( de)?( \w*)?( en date)? (du|le)"
                       r"|la cour(s)? dappel( de)? \w*( en date)? du"
                       r"|[Aa]u nom du peuple francais du"
                       r"|[Nn]ée? le"
                       r"|([Ll]a|[Ll]e)? (loi|décret)( no?\s?\d*)? (du)?"
                       r"|([Aa]ppel de cette décision|[Dd]écision attaquée)( par déclaration)?( en date)? (du|le)"
                       r"|([Ll]'|l|)?[Aa]rr[êe]t(é)?( no?\s?\d*)?( de la cour)?( en date)?( prononcé( publiquement)?)? (du|le)"
                       r"|([Pp]ar )?[Jj]ugement( qualifié de)?( contradictoire)?( définitif)?( par déclaration (au greffe)?)?( en date)?( rendu)? (du|le)"
                       r"|[Aa]udience( publique)? (du|le)"
                       r"|(mise à disposition|déposée?s?|reçues?|déclarations?|mention) au (greffe|plumitif)( de la cour)?( en date)? (le|du|au|les)"
                       r"|([Ll]|[Ll]'|)ordonnance( de référé)?( de cl[oô]ture)?( a été prise)?( en date)? (du|le)"
                       r"|([Ll]'|[Ll])affaire( renvoyée pour être)?( a été)? (débattue|plaidée) (le|au)"
                       r"|appel a été interjeté le"
                       r"|ses écritures( en date)? du"
                       r"|actes? (d|d')huissier( en date)? du"
                       r"|déclaration (d|d')appel (du)?)"
                       r" ((19|20)([0-9][0-9])(0[1-9]|1[012])(0[1-9]|[12][0-9]|3[01]))")
    txt = regex.sub(r"\1", txt)
    regex = re.compile(
        r"[Ll]e (19|20)([0-9][0-9])(0[1-9]|1[012])(0[1-9]|[12][0-9]|3[01]) cour(s)? dappel")
    txt = regex.sub("La date cours dappel", txt)
    txt = re.sub(r"\s\s+", " ", txt)
    return txt


def preprocess_text_data(df, verbose=True):
    u.vprint("Formating text", verbose)
    df["txt"] = df["txt"].map(convert_text_to_word_tokens)
    if "date_accident" in df.columns:
        df["date_accident"] = df["date_accident"].map(lambda x: re.sub(r"[-.]", "", x))
    if "date_consolidation" in df.columns:
        df["date_consolidation"] = df["date_consolidation"].map(lambda x: re.sub(r"[-.]", "", x))
    return df


def convert_text_to_word_tokens(txt):
    txt = txt.lower()
    txt = re.sub(r" 1 er", "1er", txt)
    txt = format_date(txt)
    txt = re.sub(r'[^\w\s]', '', txt)
    tokens = nltk.word_tokenize(txt, "french")
    tokens = remove_stopwords(tokens)
    return tokens

def remove_accronymes(txt):
    """Replace single dots that are preceded by an uppercase single letter provided the single letter is not immediately\
    preceded by anything in the '\w' character set. The later criterion is enforced by the negative
    lookbehind assertion - (?<!\w)"""
    return re.sub(r'(?<!\w)([A-Z])\.', r'\1', txt)


def remove_useless_dots(txt):
    txt = re.sub(r'\.{3}', "", txt)  # Remove pattern like madame X..., qui a
    return txt


def format_amount_money(txt):
    pattern = "(?P<millier>[1-9][0-9]*)[\.\s]+(?P<unit>[0-9]+)(?P<sep_cent>,?\s?)(?P<cents>[0-9][0-9])(?P<spce>\s?)(?P<devise>(€|francs))"
    txt = re.sub(pattern, r"\g<millier>\g<unit>\g<sep_cent>\g<cents>\g<devise>", txt)
    txt = re.sub(r"([0-9]+)(,) ([0-9]{1,2})\s?(€|francs)", r"\1\2\3\4", txt)
    txt = re.sub(r"([0-9]+)\s(€|francs)", r"\1\2", txt)
    return txt


def convert_month(match):
    """Function to use in in re.sub to convert month found in re match"""
    day, month, year = tuple(match.group().split())
    list_full_months = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
                        "novembre", "décembre"]
    list_trunc_month = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juill.", "août", "sept.", "oct.",
                        "nov.", "déc."]
    if month in list_full_months:
        month_number = list_full_months.index(month) + 1
    elif month in list_trunc_month:
        month_number = list_trunc_month.index(month) + 1
    if day in ["1°", "1er"]:
        day = 1
    return "%s%02d%02d" % (year, month_number, int(day))


def convert_date_year_2_digits(match):
    day = match.groupdict()["day"]
    month = match.groupdict()["month"]
    year = match.groupdict()["y2"]
    current_year = datetime.today().year % 100
    if int(year) < current_year:
        return "20%s%02d%02d" % (str(year), int(month), int(day))
    else:
        return "19%s%02d%02d" % (str(year), int(month), int(day))


def format_date(txt):
    """Look up dates in (lower) text and convert it to format YYYYMMDD"""
    pattern1 = r"(?P<day>0?[1-9]|[12][0-9]|3[01])([/.-]|\s|\s/\s|\.\s)" \
               r"(?P<month>0[1-9]|1[012])([/.-]|\s?|\s/\s|\.\s)" \
               r"(?P<y1>19|20)(?P<y2>[0-9][0-9])"
    txt = re.sub(pattern1, r"\g<y1>\g<y2>\g<month>\g<day>", txt)

    pattern_month = "(?P<month>janvier|janv.|févr.|février|mars|avr.|avril|mai|juin|juill.|juillet|août|sept.|septembre|oct.|octobre|nov.|novembre|déc.|décembre)"
    pattern2 = r"(?P<day>0?[1-9]|[12][0-9]|3[01]|(1er)|(1°)) %s (?P<y1>19|20)(?P<y2>[0-9][0-9])" % pattern_month
    txt = re.sub(pattern2, convert_month, txt)

    pattern3 = r"(?P<day>0?[1-9]|[12][0-9]|3[01])([/.-]|\s|(\s/\s)|(.\s))" \
               r"(?P<month>0[1-9]|1[012])([/.-]|\s|(\s/\s)|(.\s))" \
               r"(?P<y2>[0-9][0-9])"
    txt = re.sub(pattern3, convert_date_year_2_digits, txt)
    return txt


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in nltk.corpus.stopwords.words('french'):
            new_words.append(word)
    return new_words


def normalize(words):
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words


def get_dates(txt):
    """Get dates from formatted text (lower and date formatted)"""
    txt = re.sub(r'[^\w\s]', '', txt)
    txt_token = txt.split()
    return get_dates_from_token_list(txt_token)


def get_dates_from_token_list(txt):
    """get all dates in a list of token (formatted strings) with dates formatted as yyyymmdd. Return both the index and
    the values of the dates found."""
    pattern = r"(?P<y1>19|20)(?P<y2>[0-9][0-9])(?P<month>0[1-9]|1[012])(?P<day>0[1-9]|[12][0-9]|3[01])$"
    match_date = np.array([re.match(pattern, word) for word in txt])
    index_dates = np.argwhere(match_date).ravel()
    dates = np.array(txt)[index_dates]
    return index_dates, dates


def get_context_date(context_size, date, txt, idxs_date=None):
    """
    For a date it returns both left and right context found in text. If the date appears more than once in the text\
    multiple left and right contexts are returned. Text is a list of words (normalize tokens).
    """
    txt = np.array(txt)
    if idxs_date is None:
        idxs_date = np.argwhere(txt == date).ravel()

    left_context = []
    right_context = []
    for idx in idxs_date:
        left_bound = max(idx - context_size, 0)
        right_bound = min(idx + context_size + 1, len(txt))
        left_context.append(txt[left_bound:idx])
        right_context.append(txt[(idx + 1):right_bound])
    return left_context, right_context