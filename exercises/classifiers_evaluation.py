import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple, Callable
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def record_loss(p: Perceptron, sample_x: np.ndarray, sample_y: int):
            losses.append(p._loss(X, y))
            # print(f"X: {sample_x}, y: {sample_y}, loss: {losses[-1]}")
        Perceptron(callback=record_loss).fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(losses)), y=losses, mode="lines"))
        fig.update_layout(title=f"Perceptron Loss ({n} Dataset)", xaxis_title="Iteration", yaxis_title="Loss")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        gnb, lda = GaussianNaiveBayes().fit(X, y), LDA().fit(X, y)
        gnb_predictions, lda_predictions = gnb.predict(X), lda.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        fig = go.Figure()
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            f"algorithm: Gaussian Naive Bayes - accuracy: {accuracy(y, gnb_predictions)}",
            f"algorithm: LDA - accuracy: {accuracy(y, lda_predictions)}",
        ])
        # Add traces for data-points setting symbols and colors
        fig.add_traces(
            [
                go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker_color=p, marker_symbol=y)
                for p in [gnb_predictions, lda_predictions]
            ],
            rows=[1, 1], cols=[1, 2]
        )
        # Add `X` dots specifying fitted Gaussians' means
        # Add ellipses depicting the covariances of the fitted Gaussians
        K = np.unique(y).size
        mu = [gnb.mu_, lda.mu_]
        cov = [
            # gnb - cov matrix is diagonal, since features are independent
            [np.diag(v) for v in gnb.vars_],
            # lda - cov matrix is the same for all classes
            [lda.cov_] * K,
        ]
        trace_center = lambda m, c: go.Scatter(x=[m[0]], y=[m[1]], mode="markers", marker_color="black", marker_symbol="x")
        trace_ellipse = lambda m, c: get_ellipse(m, c)
        fig.add_traces(
            [
                trace(m[k], c[k])
                for k in range(K)
                for m, c in zip(mu, cov)
                for trace in [trace_center, trace_ellipse]
            ],
            rows=[1] * 4 * K, cols=[1, 1, 2, 2] * K
        )
        fig.update_layout(title=f"dataset: {f}", showlegend=False)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
