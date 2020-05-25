from dataprep.raw_to_formatted_data import main_raw_data_to_formatted_data
from utils.utils import remove_files_in_dir
from dataprep.utils import load_formatted_table
import pandas as pd


def test_main_raw_data_to_formatted_data(conf_test):
    # First we clean up the formatted data directory
    remove_files_in_dir(conf_test["paths"]["formatted_data"])
    # Then apply the function we want to test
    main_raw_data_to_formatted_data(conf_test)
    # Then check created files
    df2 = load_formatted_table(conf_test, "y_train.csv")
    df2_expected = pd.DataFrame({"ID": [2, 3, 8, 15],
                                 "sexe": ["femme", "femme", "femme", "homme"],
                                 "date_accident": ["1997-09-26", "1982-08-07", "1998-08-01", "1993-05-06"],
                                 "date_consolidation": ["n.c.", "1982-11-07", "1999-04-08", "n.a."]})
    pd.testing.assert_frame_equal(df2, df2_expected)

    df3 = load_formatted_table(conf_test, "x_test_ids.csv")
    df3_expected = pd.DataFrame({"ID": [772, 773, 774, 776],
                                 "filename": ["Agen_21013.txt", "Agen_31076.txt", "Agen_3436.txt", "Agen_51586.txt"]})
    pd.testing.assert_frame_equal(df3, df3_expected)

    df4 = load_formatted_table(conf_test, "x_train_ids.csv")
    df4_expected = pd.DataFrame({"ID": [2, 3, 8, 15],
                                 "filename": ["Agen_1613.txt", "Agen_2118.txt", "Agen_295.txt", "Agen_908.txt"]})
    pd.testing.assert_frame_equal(df4, df4_expected)

