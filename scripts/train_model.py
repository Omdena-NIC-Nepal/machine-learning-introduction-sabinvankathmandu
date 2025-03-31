import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib

def load_data():
    """Load preprocessed data from .pkl files."""
    X_train = pd.read_pickle("data/X_train.pkl")
    X_test = pd.read_pickle("data/X_test.pkl")
    y_train = pd.read_pickle("data/y_train.pkl")
    y_test = pd.read_pickle("data/y_test.pkl")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    """Train Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression Model Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate the model's performance on training and test data."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f"Training MSE: {mse_train:.2f}, Training R²: {r2_train:.2f}")
    print(f"Test MSE: {mse_test:.2f}, Test R²: {r2_test:.2f}")

    return y_test, y_test_pred

def plot_actual_vs_predicted(y_test, y_test_pred):
    """Visualize Actual vs Predicted Prices."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r", lw=2)  # Identity line
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted House Prices")
    plt.show()

def save_model(model):
    """Save the trained model."""
    joblib.dump(model, "model/linear_regression_model.pkl")
    print("Linear Regression model saved successfully!")

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for Ridge and Lasso models."""
    alpha_values = np.logspace(-4, 2, 50)  # 50 values from 10^(-4) to 10^(2)

    # Ridge Regression
    ridge = Ridge()
    ridge_params = {'alpha': alpha_values}

    # Lasso Regression
    lasso = Lasso()
    lasso_params = {'alpha': alpha_values}

    # Perform grid search for best hyperparameters
    ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring='r2', n_jobs=-1)
    ridge_grid.fit(X_train, y_train)

    lasso_grid = GridSearchCV(lasso, lasso_params, cv=5, scoring='r2', n_jobs=-1)
    lasso_grid.fit(X_train, y_train)

    print(f"Best Alpha for Ridge: {ridge_grid.best_params_['alpha']}")
    print(f"Best Alpha for Lasso: {lasso_grid.best_params_['alpha']}")

    return ridge_grid, lasso_grid

def train_optimized_models(X_train, y_train, ridge_grid, lasso_grid):
    """Train the optimized Ridge and Lasso models."""
    ridge_best = Ridge(alpha=ridge_grid.best_params_['alpha'])
    ridge_best.fit(X_train, y_train)

    lasso_best = Lasso(alpha=lasso_grid.best_params_['alpha'])
    lasso_best.fit(X_train, y_train)

    # Save the best models (Ridge & Lasso)
    joblib.dump(ridge_best, "model/ridge_model.pkl")
    joblib.dump(lasso_best, "model/lasso_model.pkl")

    print("Optimized Ridge & Lasso models saved!")
    return ridge_best, lasso_best

def main():
    # Load Data
    X_train, X_test, y_train, y_test = load_data()

    # Train Linear Regression Model
    linear_model = train_linear_regression(X_train, y_train)

    # Evaluate Linear Model
    y_test, y_test_pred = evaluate_model(linear_model, X_train, X_test, y_train, y_test)

    # Visualize Actual vs Predicted Prices for Linear Regression
    plot_actual_vs_predicted(y_test, y_test_pred)

    # Save the Linear Regression Model
    save_model(linear_model)

    # Hyperparameter Tuning for Ridge and Lasso
    ridge_grid, lasso_grid = hyperparameter_tuning(X_train, y_train)

    # Train Optimized Ridge and Lasso Models
    ridge_best, lasso_best = train_optimized_models(X_train, y_train, ridge_grid, lasso_grid)

    # Evaluate the optimized models on the test set
    ridge_pred = ridge_best.predict(X_test)
    lasso_pred = lasso_best.predict(X_test)

    # Calculate R² for Ridge and Lasso Models
    ridge_r2 = r2_score(y_test, ridge_pred)
    lasso_r2 = r2_score(y_test, lasso_pred)

    print(f"Ridge R²: {ridge_r2:.4f}")
    print(f"Lasso R²: {lasso_r2:.4f}")

if __name__ == "__main__":
    main()
