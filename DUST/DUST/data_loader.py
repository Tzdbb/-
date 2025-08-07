import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """
    Loads the CIR data from a CSV file and returns the dataframe.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit()

    try:
        data = pd.read_csv(file_path, encoding='utf-8', sep=',')
        print("File loaded successfully.")
        print(data.head())
    except Exception as e:
        print("Failed to load file.")
        print(e)
        exit()

    return data


def parse_complex(value):
    """
    Parses a string representing a complex number into a complex type.
    """
    try:
        return complex(value.strip("()"))
    except:
        return 0 + 0j  # If parsing fails, return 0 + 0j


def prepare_data(data):
    """
    Splits the data into training, validation, and test sets and returns the features and labels.
    """
    y = data.iloc[:, 0].apply(lambda x: 1 if x.strip().upper() == "NLOS" else 0).values  # NLOS as 1, LOS as 0
    X_complex = data.iloc[:, 1:].applymap(parse_complex).values

    X_train_temp, X_test_raw, y_train_temp, y_test = train_test_split(
        X_complex, y, test_size=0.2, random_state=42
    )

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=0.2, random_state=42
    )

    return X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test


def print_data_info(X_train_raw, X_val_raw, X_test_raw):
    """
    Prints the sizes of the train, validation, and test sets.
    """
    print(f"Training data size: {len(X_train_raw)}")
    print(f"Validation data size: {len(X_val_raw)}")
    print(f"Test data size: {len(X_test_raw)}")
