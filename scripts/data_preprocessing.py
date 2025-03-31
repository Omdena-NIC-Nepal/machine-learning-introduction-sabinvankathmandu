import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    df = pd.read_pickle(file_path)
    print(f"Data loaded from {file_path}")
    return df

def handle_missing_values(df):
    missing_values = df.isnull().sum()
    print(f"Missing Values:\n{missing_values}")

    # Drop rows with missing values
    df_cleaned = df.dropna()
    print(f"Missing values handled, new shape: {df_cleaned.shape}")

    return df_cleaned

def remove_outliers(df):
    Q1 = df.quantile(0.25)  # 25th percentile
    Q3 = df.quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1           # Interquartile range

    # Define outlier threshold
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    df_cleaned = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]
    print(f"Outliers removed, new shape: {df_cleaned.shape}")

    return df_cleaned

def plot_boxplots(df):
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.columns, 1):
        plt.subplot(4, 4, i)  # 4x4 layout for 16 features
        sns.boxplot(y=df[column])
        plt.title(f"Boxplot of {column}")
    plt.tight_layout()
    plt.show()

def normalize_data(df):
    # Select only numerical features, excluding target 'medv'
    features = df.drop(columns=["medv"])
    target = df["medv"]

    # StandardScaler fitting
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(features)

    # Create a new dataframe with scaled features
    df_scaled = pd.DataFrame(feature_scaled, columns=features.columns)

    # Append target 'medv' to the scaled dataframe
    df_scaled["medv"] = target.values

    return df_scaled, features, target

def split_data(df_scaled):
    """
    Split the data into train and test sets (80-20 split).
    """
    X = df_scaled.drop(columns=['medv'])
    y = df_scaled['medv']

    # Splitting into 80-20 train-test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test

def save_data(df_scaled, X_train, X_test, y_train, y_test):
    """
    Save the processed data to pickle files.
    """
    df_scaled.to_pickle("data/datapreprocessing_df.pkl")
    X_train.to_pickle("data/X_train.pkl")
    X_test.to_pickle("data/X_test.pkl")
    y_train.to_pickle("data/y_train.pkl")
    y_test.to_pickle("data/y_test.pkl")

    print("Processed data saved to pkl files.")

def main():
    # Load the dataset
    df = load_data("data/eda_df.pkl")

    # Handle missing values
    df = handle_missing_values(df)

    # Remove outliers
    df = remove_outliers(df)

    # Visualize boxplots for outlier detection
    plot_boxplots(df)

    # Normalize the data
    df_scaled, features, target = normalize_data(df)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df_scaled)

    # Save the processed data to pickle files
    save_data(df_scaled, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
