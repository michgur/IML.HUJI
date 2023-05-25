from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_e, validation_e = np.zeros(cv), np.zeros(cv)
    # indices to split data into train and validation sets
    cv_indices = np.linspace(0, X.shape[0], cv + 1, dtype=int)
    for i in range(cv):
        # train on all data except S_i
        train_X = np.concatenate((X[:cv_indices[i]], X[cv_indices[i + 1]:]))
        train_y = np.concatenate((y[:cv_indices[i]], y[cv_indices[i + 1]:]))
        validation_X = X[cv_indices[i]:cv_indices[i + 1]]
        validation_y = y[cv_indices[i]:cv_indices[i + 1]]

        # fit estimator on train data
        h = deepcopy(estimator).fit(train_X, train_y)
        train_e[i] = scoring(train_y, h.predict(train_X))
        validation_e[i] = scoring(validation_y, h.predict(validation_X))
    return np.mean(train_e), np.mean(validation_e)
