from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
from ...metrics import misclassification_error
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # find error for each feature, threshold, and sign
        errors = [
            (self._find_threshold(X[:, i], y, sign), i, sign)
            for i in range(X.shape[1])
            for sign in [-1, 1]
        ]
        # find minimum error, set attributes accordingly
        (self.threshold_, _), self.j_, self.sign_ = min(
            errors,
            key=lambda x: x[0][1],
        )

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] < self.threshold_, -self.sign_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # for each t, the error is the (weighted) sum of
        # + number of values below t labeled  sign
        # + number of values above t labeled -sign
        # by sorting values, we avoid having to count these per t
        sort_index = np.argsort(values)
        values, labels = values[sort_index], labels[sort_index]
        # if the last t is chosen (all labels = sign), don't split
        values[-1] = np.inf

        # weight samples by absolute value to simulate distribution
        signs, weights = np.sign(labels), np.abs(labels)
        # number of values below t labeled  sign. count from left to right
        pos_below = np.cumsum((signs == sign) * weights)
        # number of values above t labeled -sign. count from right to left
        neg_above = np.cumsum(((signs == -sign) * weights)[::-1])[::-1]
        # error of t is pos_below[t] + neg_above[t + 1]
        errors = pos_below + np.roll(neg_above, -1)
        # fix the last error (remove neg_above[0] which rolls to the end)
        errors[-1] = pos_below[-1]
        # find and return threshold with minimum error
        t = np.argmin(errors)
        return values[t], errors[t]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))
