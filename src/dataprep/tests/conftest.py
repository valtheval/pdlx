import pytest
from pathlib import Path
from utils.utils import load_conf_file

@pytest.fixture(scope="module")
def path_conf_test():
    yield Path("../../conf/CONFFORTEST.json")


@pytest.fixture(scope="module")
def conf_test(path_conf_test):
    yield load_conf_file(path_conf_test)