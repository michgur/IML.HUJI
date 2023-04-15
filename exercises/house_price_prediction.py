from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os.path
pio.templates.default = "simple_white"


def pearson_correlation(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Compute Pearson Correlation between each feature and the response
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    Returns
    -------
    Series of shape (n_features, ) containing Pearson Correlation between each feature and the response
    """
    return np.cov(X, y, rowvar=False)[:-1, -1] / (np.std(X, axis=0) * np.std(y))


def preprocess_data(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    train_columns: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    train_columns : array-like of shape (n_features, )
        Columns of preprocessed training data, used to structure test data correctly. Omit for training data

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    irrelevant_or_missing_data_features = ["id", "date", "yr_renovated"]
    low_correlation_features = ["yr_built", "condition", "long"]
    categorical_features = ["waterfront", "view", "grade"]
    positive_features = ["bedrooms", "bathrooms", "floors"]
    feature_limit = {
        "sqft_lot": 150000,
        "sqft_lot15": 150000,
    }
    
    # Create dummy variables for zipcode
    X = pd.get_dummies(X, prefix="zipcode_", columns=["zipcode"])
    # Combine yr_renovated and yr_built. Give yr_renovated a higher weight
    X["yr_renovated"] = (X["yr_renovated"] - 1900) * 3 + 1900
    X["yr_renovated_or_built"] = X[["yr_renovated", "yr_built"]].max(axis=1)

    # Remove duplicates and irrelevant features
    X = X.drop_duplicates(subset="id")
    X = X.drop(irrelevant_or_missing_data_features + low_correlation_features, axis=1)
    
    if train_columns is not None:
        # Add missing columns
        for column in set(train_columns) - set(X.columns):
            X[column] = 0
        # Remove columns not in training data
        X = X[train_columns]
    else:
        # Remove samples with missing categorical data
        for feature in categorical_features:
            size = X.shape[0]
            X = X[X[feature] >= 0]
            # print(f"Removed {size - X.shape[0]} samples with {feature} < 0")
        
        # Remove samples with invalid numeric data (unexpected nonpositive values)
        for feature in positive_features:
            size = X.shape[0]
            X = X[X[feature] > 0]
            # print(f"Removed {size - X.shape[0]} samples with {feature} <= 0")
        
        # Remove outliers
        for feature, limit in feature_limit.items():
            size = X.shape[0]
            X = X[X[feature] < limit]
            print(f"Removed {size - X.shape[0]} samples with {feature} >= {limit}")

    if y is not None:
        # match X and y, and remove samples with missing price
        y = y[X.index]
        y = y[y > 0]
        X = X.loc[y.index]

    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> None:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    corr = pearson_correlation(X, y)
    for feature in X.columns:
        if feature.startswith("zipcode_"):
            continue # for better performance, skip zipcode dummy features
        px.scatter(X, x=feature, y=y,
            marginal_x="histogram",
            title=f"{feature} - Pearson Correlation: {corr[feature]:.6f}",
            height=500,
            opacity=0.05,
            labels={"y": "price"},
            trendline="ols",
        ).write_image(os.path.join(output_path, f"{feature}.png"))


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    X, y = df.drop("price", axis=1), df["price"]
    train_X, train_y, test_X, test_y = split_train_test(X, y, train_proportion=0.75)

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)
    test_X, test_y = preprocess_data(test_X, test_y, train_columns=train_X.columns)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y, "../exercises/images/ex2/")

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    
    traning_sizes = np.arange(0.1, 1.01, 0.01)
    loss_mean, loss_std = [], []
    for p in traning_sizes:
        loss = []
        for _ in range(10):
            sample_X = train_X.sample(frac=p)
            sample_y = train_y[sample_X.index]
            model = LinearRegression(include_intercept=True).fit(sample_X, sample_y)
            loss.append(model.loss(test_X, test_y))
        loss_mean.append(np.mean(loss))
        loss_std.append(np.std(loss))

    traning_sizes *= 100
    loss_mean, loss_std = np.array(loss_mean), np.array(loss_std)

    go.Figure(
        [
            go.Scatter(x=traning_sizes, y=loss_mean, mode="markers+lines", name="Mean Loss", line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
            go.Scatter(x=traning_sizes, y=loss_mean-2*loss_std, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
            go.Scatter(x=traning_sizes, y=loss_mean+2*loss_std, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False),
        ],
        layout=go.Layout(
            title="Mean Loss as Function of Training Size (% of Overall Training Data)",
            xaxis_title="Training Size (%)",
            yaxis_title="Loss"
        )
    ).show()
