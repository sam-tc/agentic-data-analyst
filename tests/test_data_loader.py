
from io import StringIO

import pytest

from src.data_loader import load_csv


def test_valid_csv():
    csv = StringIO(
        "name,marks\nAsha,88\nRiya,92"
    )

    df = load_csv(csv)

    assert df.shape == (2, 2)
    assert df["marks"].mean() == 90


def test_empty_csv():
    csv = StringIO("name,marks\n")

    with pytest.raises(ValueError):
        load_csv(csv)
