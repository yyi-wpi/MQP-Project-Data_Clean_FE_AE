import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging
import yaml
import os

# Load configuration
config_file = 'config.yaml'
try:
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        if config is None:
            raise ValueError(f"The configuration file {config_file} is empty or has a parsing error.")
except FileNotFoundError:
    raise FileNotFoundError(f"The configuration file {config_file} was not found.")
except yaml.YAMLError as e:
    raise ValueError(f"Error parsing the configuration file {config_file}: {str(e)}")

# Set up logging
log_file = config.get('log_file', 'feature_engineering.log')
logging.basicConfig(filename=log_file,
                    level=config.get('log_level', 'INFO'),
                    format='%(asctime)s - %(levelname)s - %(message)s')


def feature_engineering(df, config):
    """
    Perform feature engineering on the input DataFrame.
    """
    try:
        logging.info("Starting feature engineering...")
        epsilon = config['feature_engineering']['bytes_per_packet_epsilon']
        df['TOTAL_BYTES'] = df['IN_BYTES'] + df['OUT_BYTES']
        df['TOTAL_PACKETS'] = df['IN_PKTS'] + df['OUT_PKTS']
        df['BYTES_PER_PACKET'] = df['TOTAL_BYTES'] / (df['TOTAL_PACKETS'] + epsilon)

        epsilon = config['feature_engineering']['in_out_ratio_epsilon']
        df['IN_OUT_BYTES_RATIO'] = df['IN_BYTES'] / (df['OUT_BYTES'] + epsilon)
        df['IN_OUT_PACKETS_RATIO'] = df['IN_PKTS'] / (df['OUT_PKTS'] + epsilon)

        logging.info("Feature engineering completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error during feature engineering: {str(e)}")
        raise


def preprocessing_pipeline(numeric_features, categorical_features, config):
    """
    Create a preprocessing pipeline for numeric and categorical features.
    """
    try:
        logging.info("Creating preprocessing pipeline...")
        logging.info(f"Numeric features: {numeric_features}")
        logging.info(f"Categorical features: {categorical_features}")

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=config['numeric_imputer_strategy'])),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=config['categorical_imputer_strategy'])),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        return preprocessor
    except Exception as e:
        logging.error(f"Error creating preprocessing pipeline: {str(e)}")
        raise


def main():
    try:
        # Load data
        df = pd.read_csv(config['input_file'])
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        logging.info(f"Columns in the dataframe: {df.columns.tolist()}")

        # Print the first few rows of the DataFrame
        logging.info(f"First few rows of the dataframe:\n{df.head().to_string()}")

        # Perform feature engineering
        df = feature_engineering(df, config)

        # Get target column
        target_column = config['target_column']
        if target_column not in df.columns:
            logging.error(f"Target column '{target_column}' not found in the dataframe.")
            logging.info(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Target column '{target_column}' not found in the dataframe.")

        # Identify numeric and categorical columns, excluding the target column
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_features = [col for col in numeric_features if col != target_column]
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        categorical_features = [col for col in categorical_features if col != target_column]

        logging.info(f"Numeric features: {numeric_features}")
        logging.info(f"Categorical features: {categorical_features}")

        # Create preprocessing pipeline
        pipeline = preprocessing_pipeline(numeric_features, categorical_features, config)

        # Fit and transform the data
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        logging.info(f"Shape of X before transformation: {X.shape}")
        logging.info(f"Columns in X: {X.columns.tolist()}")
        X_transformed = pipeline.fit_transform(X)
        logging.info(f"Shape of X after transformation: {X_transformed.shape}")

        # Save transformed data
        pd.DataFrame(X_transformed).to_csv(config['output_file'], index=False)
        logging.info(f"Transformed data saved to {config['output_file']}")

    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")
        logging.error(f"DataFrame info:\n{df.info()}")
        raise


if __name__ == "__main__":
    main()