import pytest
from linear_regression.tools import read_csv


def test_valid_csv(tmp_path):
    """Test reading a valid CSV file."""
    data = "km,price\n15000,2000\n30000,3000"
    file = tmp_path / "valid.csv"
    file.write_text(data)

    km, price = read_csv(file)
    assert km == [15000, 30000]
    assert price == [2000, 3000]


def test_missing_file():
    """Test behavior when the file does not exist."""
    with pytest.raises(FileNotFoundError, match="was not found"):
        read_csv("nonexistent.csv")


def test_empty_file(tmp_path):
    """Test behavior when the file is empty."""
    file = tmp_path / "empty.csv"
    file.write_text("")  # Create an empty file

    with pytest.raises(ValueError, match="The file is empty"):
        read_csv(file)


def test_invalid_format(tmp_path):
    """Test behavior when the file does not have a .csv extension."""
    data = "This is not a valid CSV content"
    file = tmp_path / "invalid.pdf"  # Non-CSV file extension
    file.write_text(data)

    with pytest.raises(ValueError, match="The file must have a .csv extension"):
        read_csv(str(file))



def test_missing_columns(tmp_path):
    """Test behavior when required columns are missing."""
    data = "mileage,cost\n15000,2000"
    file = tmp_path / "missing_columns.csv"
    file.write_text(data)

    with pytest.raises(ValueError, match="The CSV file must contain the following columns: "):
        read_csv(file)


def test_missing_values(tmp_path):
    """Test behavior when there are missing values."""
    data = "km,price\n15000,2000\n30000,"
    file = tmp_path / "missing_values.csv"
    file.write_text(data)

    with pytest.raises(ValueError, match="Warning: Missing values detected."):
        read_csv(file)


def test_non_numeric_values(tmp_path):
    """Test behavior when non-numeric values are present."""
    data = "km,price\n15000,abc\n30000,3000"
    file = tmp_path / "non_numeric.csv"
    file.write_text(data)

    with pytest.raises(ValueError, match="must contain numeric values"):
        read_csv(file)


def test_negative_values(tmp_path):
    """Test behavior when negative values are present."""
    data = "km,price\n-15000,2000\n30000,-3000"
    file = tmp_path / "negative_values.csv"
    file.write_text(data)

    with pytest.raises(ValueError, match="must have non-negative values"):
        read_csv(file)


def test_unrealistic_values(tmp_path):
    """Test behavior when unrealistic values are present."""
    data = "km,price\n15000,2000\n1000000000,300000000000000000"
    file = tmp_path / "unrealistic_values.csv"
    file.write_text(data)

    with pytest.raises(ValueError, match="Detected unrealistic values"):
        read_csv(file)


def test_valid_with_missing_rows_dropped(tmp_path):
    """Test handling valid data with missing rows (drop them)."""
    data = "km,price\n15000,2000\n30000,\n50000,4000"
    file = tmp_path / "valid_with_dropped_rows.csv"
    file.write_text(data)

    with pytest.raises(ValueError, match="Warning: Missing values detected."):
        km, price = read_csv(file)
        assert km == [15000, 50000]  # Row with missing price is dropped
        assert price == [2000, 4000]
