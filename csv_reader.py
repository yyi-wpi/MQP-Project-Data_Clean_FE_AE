import pandas as pd
import numpy as np

def analyze_csv(file_path, num_rows=5):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        print(f"Total number of rows: {len(df)}")
        print(f"Total number of columns: {len(df.columns)}")

        print("\nColumn names:")
        print(df.columns.tolist())

        print(f"\nFirst {num_rows} rows of the CSV file:")
        print(df.head(num_rows).to_string())

        print("\nData types:")
        print(df.dtypes)

        print("\nBasic statistics for numeric columns:")
        print(df.describe())

        print("\nUnique values in each column:")
        for column in df.columns:
            unique_values = df[column].nunique()
            print(f"{column}: {unique_values} unique values")

        print("\nMissing values:")
        print(df.isnull().sum())

        # Check if the last column might be the target variable
        last_column = df.columns[-1]
        unique_last = df[last_column].nunique()
        print(f"\nLast column '{last_column}' has {unique_last} unique values.")
        if unique_last < 10:  # Assuming if it has few unique values, it might be a classification target
            print(f"Value counts for last column:")
            print(df[last_column].value_counts())

    except Exception as e:
        print(f"An error occurred while analyzing the CSV file: {str(e)}")

if __name__ == "__main__":
    file_path = 'NF-UNSW-NB15-v2_cleaned.csv'  # Replace with your actual file path
    analyze_csv(file_path)