# evaluate_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data():
    """
    Load the preprocessed data and models.
    """
    # Load the preprocessed data
    X_train = pd.read_pickle("data/X_train.pkl")
    X_test = pd.read_pickle("data/X_test.pkl")
    y_train = pd.read_pickle("data/y_train.pkl")
    y_test = pd.read_pickle("data/y_test.pkl")

    # Load the models
    linear_model = joblib.load("model/linear_regression_model.pkl")
    ridge_best = joblib.load("model/ridge_model.pkl")
    lasso_best = joblib.load("model/lasso_model.pkl")

    return X_train, X_test, y_train, y_test, linear_model, ridge_best, lasso_best

def evaluate_linear_model(X_train, X_test, y_train, y_test, linear_model):
    """
    Evaluate the performance of the Linear Regression model.
    """
    # Linear model predictions
    y_train_pred = linear_model.predict(X_train)
    y_test_pred = linear_model.predict(X_test)

    # Evaluate Linear Regression
    mae_lr = mean_absolute_error(y_test, y_test_pred)
    mse_lr = mean_squared_error(y_test, y_test_pred)
    r2_lr = r2_score(y_test, y_test_pred)
    rmse_lr = np.sqrt(mse_lr)

    print(f"Linear Regression MAE {mae_lr:.4f}, MSE: {mse_lr:.4f}, R²: {r2_lr:.4f}, RMSE: {rmse_lr:.4f}")

    return y_test_pred, y_test, mae_lr, mse_lr, r2_lr, rmse_lr

def evaluate_ridge_lasso_models(X_test, y_test, ridge_best, lasso_best):
    """
    Evaluate the performance of the Ridge and Lasso models.
    """
    # Ridge Model Predictions
    y_pred_ridge = ridge_best.predict(X_test)

    # Lasso Model Predictions
    y_pred_lasso = lasso_best.predict(X_test)

    # Evaluate Ridge Model
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)

    # Evaluate Lasso Model
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)

    print(f"Ridge Model MSE: {mse_ridge:.4f}, R²: {r2_ridge:.4f}")
    print(f"Lasso Model MSE: {mse_lasso:.4f}, R²: {r2_lasso:.4f}")

    return y_pred_ridge, y_pred_lasso, mse_ridge, r2_ridge, mse_lasso, r2_lasso

def plot_actual_vs_predicted(y_test, y_test_pred, y_pred_ridge, y_pred_lasso):
    """
    Plot Actual vs Predicted values for Linear, Ridge, and Lasso models.
    """
    plt.figure(figsize=(12, 6))

    # Linear model: Actual vs Predicted
    sns.scatterplot(x=y_test, y=y_test_pred, label="Linear Regression", alpha=0.7)

    # Ridge model: Actual vs Predicted
    sns.scatterplot(x=y_test, y=y_pred_ridge, label="Ridge", alpha=0.7)

    # Lasso model: Actual vs Predicted
    sns.scatterplot(x=y_test, y=y_pred_lasso, label="Lasso", alpha=0.7)

    # Identity line (for reference)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r", lw=2)

    # Titles and labels
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices (Linear, Ridge, Lasso)")
    plt.legend()
    plt.show()

def plot_residuals(y_test, y_test_pred, y_pred_ridge, y_pred_lasso):
    """
    Plot the residuals for Linear, Ridge, and Lasso models.
    """
    plt.figure(figsize=(12, 6))

    # Linear Regression Residuals
    sns.histplot(y_test - y_test_pred, bins=30, kde=True, color="blue", label="Linear Regression", alpha=0.6)

    # Ridge Residuals
    sns.histplot(y_test - y_pred_ridge, bins=30, kde=True, color="green", label="Ridge", alpha=0.6)

    # Lasso Residuals
    sns.histplot(y_test - y_pred_lasso, bins=30, kde=True, color="orange", label="Lasso", alpha=0.6)

    plt.xlabel("Residuals (Error)")
    plt.title("Distribution of Residuals (Linear, Ridge & Lasso)")
    plt.legend()
    plt.show()

def feature_importance(ridge_best, lasso_best, X_train):
    """
    Plot the feature importance from Ridge and Lasso models.
    """
    # Get feature names
    feature_names = X_train.columns

    # Get coefficients for Ridge and Lasso models
    ridge_coefs = ridge_best.coef_
    lasso_coefs = lasso_best.coef_

    # Create DataFrame for easier comparison
    coefs_df = pd.DataFrame({
        "Feature": feature_names,
        "Ridge Coefficients": ridge_coefs,
        "Lasso Coefficients": lasso_coefs
    })

    # Sort by absolute importance
    coefs_df['Ridge Importance'] = np.abs(coefs_df['Ridge Coefficients'])
    coefs_df['Lasso Importance'] = np.abs(coefs_df['Lasso Coefficients'])
    coefs_df_sorted = coefs_df.sort_values(by="Ridge Importance", ascending=False)

    # Plot Feature Importance (Ridge & Lasso)
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Ridge Importance", y="Feature", data=coefs_df_sorted, color="blue", label="Ridge", alpha=0.7)
    sns.barplot(x="Lasso Importance", y="Feature", data=coefs_df_sorted, color="orange", label="Lasso", alpha=0.7)
    plt.title("Feature Importance (Ridge & Lasso Models)")
    plt.legend()
    plt.show()

def save_evaluation_results(mse_lr, r2_lr, mse_ridge, r2_ridge, mse_lasso, r2_lasso):
    """
    Save the evaluation results to a CSV file.
    """
    results_df = pd.DataFrame({
        "Model": ["Linear Regression", "Ridge", "Lasso"],
        "MSE": [mse_lr, mse_ridge, mse_lasso],
        "R²": [r2_lr, r2_ridge, r2_lasso]
    })

    # Save the results to CSV
    results_df.to_csv("model/ridge_lasso_evaluation_results.csv", index=False)
    print("Evaluation results saved to CSV!")

def main():
    # Load data and models
    X_train, X_test, y_train, y_test, linear_model, ridge_best, lasso_best = load_data()

    # Evaluate Linear Model
    y_test_pred, y_test, mae_lr, mse_lr, r2_lr, rmse_lr = evaluate_linear_model(X_train, X_test, y_train, y_test, linear_model)

    # Evaluate Ridge and Lasso Models
    y_pred_ridge, y_pred_lasso, mse_ridge, r2_ridge, mse_lasso, r2_lasso = evaluate_ridge_lasso_models(X_test, y_test, ridge_best, lasso_best)

    # Plot Actual vs Predicted values
    plot_actual_vs_predicted(y_test, y_test_pred, y_pred_ridge, y_pred_lasso)

    # Plot residuals
    plot_residuals(y_test, y_test_pred, y_pred_ridge, y_pred_lasso)

    # Plot Feature Importance
    feature_importance(ridge_best, lasso_best, X_train)

    # Save evaluation results to CSV
    save_evaluation_results(mse_lr, r2_lr, mse_ridge, r2_ridge, mse_lasso, r2_lasso)

if __name__ == "__main__":
    main()
