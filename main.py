import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(file_path, missing_value_threshold=0.5):
    try:
        # Load the data
        logging.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)

        # Display initial info
        logging.info(f"Initial shape: {df.shape}")
        logging.info(f"Columns: {df.columns.tolist()}")

        # Drop columns with excessive missing values
        threshold = len(df) * missing_value_threshold
        df = df.dropna(axis=1, thresh=threshold)
        logging.info(f"Shape after dropping columns with >{missing_value_threshold*100}% missing values: {df.shape}")

        # Fill missing values in numerical columns with the median
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            median = df[col].median()
            df[col] = df[col].fillna(median)
        logging.info("Filled missing values in numerical columns with median")

        # Fill missing values in categorical columns with the mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode)
        logging.info("Filled missing values in categorical columns with mode")

        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        logging.info(f"Removed {removed_duplicates} duplicate rows")

        # Convert timestamp if the column exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            logging.info("Converted 'timestamp' column to datetime")
        else:
            logging.warning("'timestamp' column not found in the dataset")

        # Log some basic statistics for sanity check
        logging.info(f"Final shape of the cleaned data: {df.shape}")
        logging.info(f"First few rows of the cleaned data: \n{df.head()}")

        return df

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return None

def main():
    file_path = r'C:\Users\yiyan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\MQP\Data\NF-UNSW-NB15-v2.csv'
    cleaned_file_path = r'C:\Users\yiyan\OneDrive - Worcester Polytechnic Institute (wpi.edu)\MQP\Data\NF-UNSW-NB15-v2_cleaned.csv'

    cleaned_df = clean_data(file_path)

    if cleaned_df is not None:
        cleaned_df.to_csv(cleaned_file_path, index=False)
        logging.info(f"Cleaned data saved to {cleaned_file_path}")
    else:
        logging.error("Failed to clean the data.")

if __name__ == "__main__":
    main()
