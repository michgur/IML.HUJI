from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(as_frame=True, return_X_y=True)
    X_train, y_train, X_test, y_test = [
        samples.to_numpy() for samples in
        split_train_test(X, y, train_proportion=n_samples / len(X))
    ]

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_lambdas = np.linspace(0, 1.5, n_evaluations)
    lasso_lambdas = np.linspace(0, 3, n_evaluations)
    ridge_errors, lasso_errors = np.zeros((n_evaluations, 2)), np.zeros((n_evaluations, 2))
    for i in range(n_evaluations):
        ridge = RidgeRegression(lam=ridge_lambdas[i], include_intercept=True)
        lasso = Lasso(alpha=lasso_lambdas[i], fit_intercept=True)
        ridge_errors[i] = cross_validate(ridge, X_train, y_train, mean_square_error, 5)
        lasso_errors[i] = cross_validate(lasso, X_train, y_train, mean_square_error, 5)

    for name, lambdas, errors in [
        ('Ridge', ridge_lambdas, ridge_errors),
        ('Lasso', lasso_lambdas, lasso_errors),
    ]:
        go.Figure(
            data=[
                go.Scatter(x=lambdas, y=errors[:, 0], name='train', mode='lines'),
                go.Scatter(x=lambdas, y=errors[:, 1], name='validation', mode='lines'),
            ],
            layout=go.Layout(
                title=f'{name} Regularization - train / validation errors',
                xaxis_title='Lambda',
                yaxis_title='MSE',
            )
        ).show()

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_lambda = ridge_lambdas[np.argmin(ridge_errors[:, 1])]
    best_lasso_lambda = lasso_lambdas[np.argmin(lasso_errors[:, 1])]
    ridge = RidgeRegression(lam=best_ridge_lambda, include_intercept=True).fit(X_train, y_train)
    lasso = Lasso(alpha=best_lasso_lambda, fit_intercept=True).fit(X_train, y_train)
    ls = LinearRegression(include_intercept=True).fit(X_train, y_train)

    print(f'Best Ridge model: lambda={best_ridge_lambda}')
    print(f'Best Lasso model: lambda={best_lasso_lambda}')
    print()
    print(f'Ridge test error={mean_square_error(y_test, ridge.predict(X_test))}')
    print(f'Lasso test error={mean_square_error(y_test, lasso.predict(X_test))}')
    print(f'Least Squares test error={mean_square_error(y_test, ls.predict(X_test))}')


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
