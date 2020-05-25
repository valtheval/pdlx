from dataprep.utils import *


def test_load_config_raw_data(conf_test):
    conf_raw_data = load_config_raw_data(conf_test)
    assert len(conf_raw_data.keys()) == 7
    assert type(conf_raw_data) == dict


def test_load_raw_table(conf_test):
    df1 = load_raw_table(conf_test, "y_test_pred")
    df1_expected = pd.DataFrame({"filename" : [772, 773, 774, 776],
                                 "sexe" : ["femme", "femme", "homme", "femme"],
                                 "date_accident" : ["1997-09-26", "1982-08-07", "1996-11-26", "1999-11-14"],
                                 "date_consolidation" : ["n.c.", "1982-11-07", "n.c.", "n.c."]})
    pd.testing.assert_frame_equal(df1, df1_expected)

    df2 = load_raw_table(conf_test, "y_train_predilex")
    df2_expected = pd.DataFrame({"ID": [2, 3, 8, 15],
                                 "sexe": ["femme", "femme", "femme", "homme"],
                                 "date_accident": ["1997-09-26", "1982-08-07", "1998-08-01", "1993-05-06"],
                                 "date_consolidation": ["n.c.", "1982-11-07", "1999-04-08", "n.a."]})
    pd.testing.assert_frame_equal(df2, df2_expected)

    df3 = load_raw_table(conf_test, "x_test_ids")
    df3_expected = pd.DataFrame({"ID": [772, 773, 774, 776],
                                 "filename": ["Agen_21013.txt", "Agen_31076.txt", "Agen_3436.txt", "Agen_51586.txt"]})
    pd.testing.assert_frame_equal(df3, df3_expected)

    df4 = load_raw_table(conf_test, "x_train_ids")
    df4_expected = pd.DataFrame({"ID": [2, 3, 8, 15],
                                 "filename": ["Agen_1613.txt", "Agen_2118.txt", "Agen_295.txt", "Agen_908.txt"]})
    pd.testing.assert_frame_equal(df4, df4_expected)