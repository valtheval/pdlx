from dataprep.create_datasets import *
from dataprep.raw_to_formatted_data import main_raw_data_to_formatted_data
import pytest

def test_load_all_jurisprudences(conf_test):
    dftrain = load_all_jurisprudences(conf_test)
    assert dftrain.shape[0] == 4
    assert dftrain.shape[1] == 2
    assert dftrain.columns[0] == "filename"
    assert dftrain.columns[1] == "txt"

    dftest = load_all_jurisprudences(conf_test, "test")
    assert dftest.shape[0] == 4
    assert dftest.shape[1] == 2
    assert dftest.columns[0] == "filename"
    assert dftest.columns[1] == "txt"

    with pytest.raises(ValueError):
        load_all_jurisprudences(conf_test, "wrong_arg")


def test_create_df_train(conf_test):
    # We need to create formatted data before
    main_raw_data_to_formatted_data(conf_test)
    # Now we can check the create_df_train function
    df = create_df_train(conf_test)
    assert df.shape[0] == 4
    assert df.shape[1] == 6
    assert df.isnull().sum().sum() == 0
    s1 = pd.DataFrame(df["filename"])
    s2 = pd.DataFrame(pd.Series(["Agen_1613.txt", "Agen_2118.txt", "Agen_295.txt", "Agen_908.txt"], name="filename"))
    assert s1.equals(s2)