import pandas as pd

def normalize_input(value, mean, std):
    """
    Normalize input value using mean and standard deviation.
    :param value: Input value to normalize
    :param mean: Mean of the training data
    :param std: Standard deviation of the training data
    :return: Normalized value
    """
    return (value - mean) / std

def denormalize_output(value, mean, std):
    """
    Denormalize output value to return to the original scale.
    :param value: Normalized value to denormalize
    :param mean: Mean of the target data
    :param std: Standard deviation of the target data
    :return: Denormalized value
    """
    return (value * std) + mean

def read_csv(file_path):
    """
    Read mileage (km) and price data from a CSV file using Pandas with validation.

    Validations:
    - File extension check
    - File existence and format validation
    - Column presence validation
    - Missing value handling
    - Numeric value validation
    - Range checks for unrealistic values
    """
    # Step 1: Ensure the file has a .csv extension
    # This prevents non-CSV files from being processed.
    if not str(file_path).endswith('.csv'):
        raise ValueError("The file must have a .csv extension.")

    try:
        # Step 2: Attempt to read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Raise an error if the file is not found
        raise FileNotFoundError(f"The file at '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        # Raise an error if the file is empty
        raise ValueError("The file is empty. Please provide a valid dataset.")
    except pd.errors.ParserError:
        # Raise an error if the file cannot be parsed as a valid CSV
        raise ValueError("The file could not be parsed. Ensure it is in a valid CSV format.")

    # Step 3: Validate that required columns ('km' and 'price') are present
    required_columns = {"km", "price"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

    # Step 4: Check for and handle missing values in 'km' and 'price'
    if df[["km", "price"]].isnull().values.any():
        # Drop rows with missing values and raise a warning
        df = df.dropna(subset=["km", "price"])
        raise ValueError("Warning: Missing values detected. These rows will be dropped.")

    # Step 5: Validate that 'km' and 'price' columns contain numeric values
    if not pd.api.types.is_numeric_dtype(df["km"]) or not pd.api.types.is_numeric_dtype(df["price"]):
        raise ValueError("Both 'km' and 'price' columns must contain numeric values.")

    # Step 6: Validate that 'km' and 'price' have non-negative values
    if (df["km"] < 0).any() or (df["price"] < 0).any():
        raise ValueError("Both 'km' and 'price' must have non-negative values.")

    # Step 7: Check for unrealistic values in 'price' and 'km'
    if (df["price"] > 1e7).any():
        raise ValueError("Detected unrealistic values in the 'price' column. Please check your data.")
    if (df["km"] > 1e6).any():
        raise ValueError("Detected unrealistic values in the 'km' column. Please check your data.")

    # Step 8: Convert valid 'km' and 'price' data to lists for further processing
    km = df["km"].to_list()
    price = df["price"].to_list()

    # Success message with the number of rows loaded
    print(f"Successfully loaded {len(km)} rows of data.")
    return km, price
