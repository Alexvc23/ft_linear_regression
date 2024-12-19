import pytest
from linear_regression.tools import read_csv


def test_valid_csv(tmp_path):
    """
    Test the read_csv function with a valid CSV file.

    This test creates a temporary CSV file with two rows of data,
    reads the file using the read_csv function, and asserts that
    the returned km and price lists match the expected values.

    Args:
        tmp_path (pathlib.Path): Temporary directory provided by pytest.

    Raises:
        AssertionError: If the returned km and price lists do not match
                        the expected values.
    """
    """Test reading a valid CSV file."""
    data = "km,price\n15000,2000\n30000,3000"
    file = tmp_path / "valid.csv"
    file.write_text(data)

    km, price = read_csv(file)
    assert km == [15000, 30000]
    assert price == [2000, 3000]

# ------------------------------------------------------------

def test_missing_file():
    """
    Test the read_csv function to ensure it raises a FileNotFoundError
    when attempting to read a file that does not exist.

    This test verifies that the appropriate exception is raised with the
    correct error message when the specified file is not found.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    """Test behavior when the file does not exist."""
    with pytest.raises(FileNotFoundError, match="was not found"):
        read_csv("nonexistent.csv")

# ------------------------------------------------------------

def test_empty_file(tmp_path):
    """
    Test the behavior of the read_csv function when provided with an empty file.

    This test creates an empty CSV file and verifies that the read_csv function
    raises a ValueError with the message "The file is empty".

    Args:
        tmp_path (pathlib.Path): A temporary directory provided by pytest to create
                                 temporary files and directories for testing.

    Raises:
        ValueError: If the read_csv function does not raise a ValueError with the
                    expected message when the file is empty.
    """
    """Test behavior when the file is empty."""
    file = tmp_path / "empty.csv"
    file.write_text("")  # Create an empty file

    with pytest.raises(ValueError, match="The file is empty"):
        read_csv(file)

# ------------------------------------------------------------

def test_invalid_format(tmp_path):
    """
    Test the behavior of the read_csv function when provided with a file that does not have a .csv extension.

    This test creates a temporary file with a .pdf extension and writes invalid CSV content to it.
    It then checks if the read_csv function raises a ValueError with the appropriate error message
    indicating that the file must have a .csv extension.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest's tmp_path fixture.

    Raises:
        ValueError: If the file does not have a .csv extension.
    """
    data = "This is not a valid CSV content"
    file = tmp_path / "invalid.pdf"  # Non-CSV file extension
    file.write_text(data)

    with pytest.raises(ValueError, match="The file must have a .csv extension"):
        read_csv(str(file))

# ------------------------------------------------------------

def test_missing_columns(tmp_path):
    """
    Test the behavior of the read_csv function when required columns are missing from the CSV file.

    This test creates a temporary CSV file with missing columns and verifies that the read_csv 
    function raises a ValueError with an appropriate error message.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest.

    Raises:
        ValueError: If the CSV file does not contain the required columns.
    """
    """Test behavior when required columns are missing."""
    data = "mileage,cost\n15000,2000"
    file = tmp_path / "missing_columns.csv"
    file.write_text(data)

    with pytest.raises(ValueError, match="The CSV file must contain the following columns: "):
        read_csv(file)

# ------------------------------------------------------------

def test_missing_values(tmp_path):
    """
    Test the behavior of the read_csv function when the input CSV file contains missing values.

    This test creates a temporary CSV file with missing values and verifies that the read_csv 
    function raises a ValueError with the appropriate warning message.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest's tmp_path fixture.

    Raises:
        ValueError: If the read_csv function detects missing values in the CSV file.
    """
    """Test behavior when there are missing values."""
    data = "km,price\n15000,2000\n30000,"
    file = tmp_path / "missing_values.csv"
    file.write_text(data)

    with pytest.raises(ValueError, match="Warning: Missing values detected."):
        read_csv(file)

# ------------------------------------------------------------

def test_non_numeric_values(tmp_path):
    """
    Test the behavior of the read_csv function when the CSV file contains non-numeric values.

    This test creates a temporary CSV file with non-numeric values in the 'price' column
    and verifies that the read_csv function raises a ValueError with an appropriate error message.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest.

    Raises:
        ValueError: If the CSV file contains non-numeric values.
    """

    """Test behavior when non-numeric values are present."""

    data = "km,price\n15000,abc\n30000,3000"
    file = tmp_path / "non_numeric.csv"
    file.write_text(data)

    with pytest.raises(ValueError, match="must contain numeric values"):
        read_csv(file)

# ------------------------------------------------------------


def test_negative_values(tmp_path):
    """
    Test the behavior of the read_csv function when the CSV file contains negative values.

    This test creates a temporary CSV file with negative values for both 'km' and 'price' columns.
    It then checks if the read_csv function raises a ValueError with the appropriate error message
    indicating that the values must be non-negative.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest's tmp_path fixture.

    Raises:
        ValueError: If the read_csv function does not raise a ValueError when negative values are present.
    """
    """Test behavior when negative values are present."""
    data = "km,price\n-15000,2000\n30000,-3000"
    file = tmp_path / "negative_values.csv"
    file.write_text(data)

    with pytest.raises(ValueError, match="must have non-negative values"):
        read_csv(file)

# ------------------------------------------------------------

def test_unrealistic_values(tmp_path):
    """
    Test the behavior of the read_csv function when the input CSV file contains unrealistic values.

    This test creates a temporary CSV file with unrealistic values for the 'km' and 'price' columns.
    It then checks if the read_csv function raises a ValueError with the message "Detected unrealistic values"
    when attempting to read this file.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest's tmp_path fixture.

    Raises:
        ValueError: If the read_csv function detects unrealistic values in the CSV file.
    """
    data = "km,price\n15000,2000\n1000000000,300000000000000000"
    file = tmp_path / "unrealistic_values.csv"
    file.write_text(data)

    with pytest.raises(ValueError, match="Detected unrealistic values"):
        read_csv(file)

# ------------------------------------------------------------

def test_valid_with_missing_rows_dropped(tmp_path):
    """
    Test the `read_csv` function to ensure it correctly handles valid data with missing rows by dropping them.

    This test creates a temporary CSV file with some missing values in the 'price' column. It then verifies that the 
    `read_csv` function raises a ValueError with the appropriate warning message about missing values. Additionally, 
    it checks that the function correctly drops the rows with missing values and returns the expected lists for 'km' 
    and 'price'.

    Args:
        tmp_path (pathlib.Path): A temporary directory path provided by pytest's tmp_path fixture.

    Raises:
        ValueError: If the `read_csv` function does not raise a ValueError with the expected warning message.
    """
    """Test handling valid data with missing rows (drop them)."""
    data = "km,price\n15000,2000\n30000,\n50000,4000"
    file = tmp_path / "valid_with_dropped_rows.csv"
    file.write_text(data)

    with pytest.raises(ValueError, match="Warning: Missing values detected."):
        km, price = read_csv(file)
        assert km == [15000, 50000]  # Row with missing price is dropped
        assert price == [2000, 4000]
