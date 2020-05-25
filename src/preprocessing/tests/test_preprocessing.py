from preprocessing import preprocessing as pp


def test_format_date():
    text1 = "bonjour nous sommes le 24 mars 2020 en plein confinement depuis le 15/03/2020"
    text2 = "30-04-1991 un beau jour de printemps pour nous tous. 30 avr. 1991..."
    text3 = "20200310 départ au ski"
    text4 = "Nous sommes le 1er avril 2019 et tout va bien, j'ai dit 1° avril 2019"
    text5 = "date consolidation le 14 / 10 / 2002 i.e. 14. 10. 2002"
    text6 = "date de consolidation le 14/10/02 i.e. le 14 avril 2002 mais pas le 14/10/78"
    out1 = "bonjour nous sommes le 20200324 en plein confinement depuis le 20200315"
    out2 = "19910430 un beau jour de printemps pour nous tous. 19910430..."
    out3 = "20200310 départ au ski"
    out4 = "Nous sommes le 20190401 et tout va bien, j'ai dit 20190401"
    out5 = "date consolidation le 20021014 i.e. 20021014"
    out6 = "date de consolidation le 20021014 i.e. le 20020414 mais pas le 19781014"

    res1 = pp.format_date(text1)
    res2 = pp.format_date(text2)
    res3 = pp.format_date(text3)
    res4 = pp.format_date(text4)
    res5 = pp.format_date(text5)
    res6 = pp.format_date(text6)

    assert res1 == out1
    assert res2 == out2
    assert res3 == out3
    assert res4 == out4
    assert res5 == out5
    assert res6 == out6

def test_get_dates_from_token_list():
    text1 = ["bonjour", "nous", "sommes", "le", "20200324", "en", "plein", "confinement", "depuis", "le", "20200315"]
    text2 = ["19910430", "un", "beau", "jour", "de", "printemps", "pour", "nous", "tous", "19910430"]
    text3 = ["salut", "monde", "il", "n", "y", "a", "rien", "a", "voir", "ici"]

    index_dates1, dates1 = pp.get_dates_from_token_list(text1)
    index_dates2, dates2 = pp.get_dates_from_token_list(text2)
    index_dates3, dates3 = pp.get_dates_from_token_list(text3)

    assert all(a == b for a, b in zip(index_dates1, [4, 10]))
    assert all(a == b for a, b in zip(dates1, ["20200324", "20200315"]))
    assert all(a == b for a, b in zip(index_dates2, [0, 9]))
    assert all(a == b for a,b in zip(dates2, ["19910430", "19910430"]))
    assert len(index_dates3) == 0
    assert len(dates3) == 0

def test_get_context_date():
    text1 = ["bonjour", "nous", "sommes", "le", "20200324", "en", "plein", "confinement", "depuis", "le", "20200315"]
    text2 = ["19910430", "un", "beau", "jour", "de", "printemps", "pour", "nous", "tous", "19910430"]
    text3 = ["salut", "monde", "il", "n", "y", "a", "rien", "a", "voir", "ici"]

    left_context1, right_context1 = pp.get_context_date(3, "20200324", text1)
    assert all(a == b for a, b in zip(left_context1[0], ["nous", "sommes", "le"]))
    assert all(a == b for a, b in zip(right_context1[0], ["en", "plein", "confinement"]))
    assert len(left_context1) == 1
    assert len(right_context1) == 1

    left_context1, right_context1 = pp.get_context_date(10, "20200324", text1)
    assert all(a == b for a, b in zip(left_context1[0], ["bonjour", "nous", "sommes", "le"]))
    assert all(a == b for a, b in zip(right_context1[0], ["en", "plein", "confinement", "depuis", "le", "20200315"]))

    left_context12, right_context12 = pp.get_context_date(0, "20200324", text1)
    assert len(left_context12[0]) == 0
    assert len(right_context12[0]) == 0

    left_context13, right_context13 = pp.get_context_date(3, "20200315", text1)
    assert all(a == b for a, b in zip(left_context13[0], ["confinement", "depuis", "le"]))
    assert len(right_context13[0]) == 0

    left_context21, right_context21 = pp.get_context_date(3, "19910430", text2)
    assert all(a == b for a, b in zip(right_context21[0], ["un", "beau", "jour"]))
    assert all(a == b for a, b in zip(left_context21[1], ["pour", "nous", "tous"]))
    assert len(left_context21[0]) == 0
    assert len(right_context21[1]) == 0
    assert len(left_context21) == 2
    assert len(right_context21) == 2

    left_context31, right_context31 = pp.get_context_date(3, "19910430", text3)
    assert len(left_context31) == 0
    assert len(right_context31) == 0


def test_remove_punctuation():
    text = ["salut", ",", "monde", "il", "n", "'", "y", "a", "rien", "a", "voir", "ici", ".", "n", "'", "est", "-",
            "ce", "pas" "?"]
    text_out = pp.remove_punctuation(text)
    expected = ["salut", "monde", "il", "n", "y", "a", "rien", "a", "voir", "ici", "n", "est", "ce", "pas"]
    assert all(a == b for a, b in zip(text_out, expected))

def test_remove_stop_words():
    """Mostly to check if the nltk stopwords list still available"""
    text = ['au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'qu', 'que', 'qui', 'sa', 'se', 'ses', 'son',
            'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre']
    out = pp.remove_stopwords(text)
    assert len(out) == 0
