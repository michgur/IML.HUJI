import numpy as np
from typing import Tuple, List
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics.loss_functions import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model: AdaBoost = AdaBoost(lambda: DecisionStump(), n_learners).fit(train_X, train_y)
    n_iter = np.arange(1, n_learners + 1)
    train_e, test_e = [
        np.array([model.partial_loss(X, y, t) for t in n_iter])
        for X, y in [[train_X, train_y], [test_X, test_y]]
    ]
    df = pd.DataFrame({'x': n_iter, 'train': train_e, 'test': test_e})
    px.line(
        df,
        x='x',
        y=['train', 'test'],
        labels={'value': 'error', 'x': '# of learners', 'variable': 'dataset'},
        title="Train and test error as a function of # of learners in ensemble",
    ).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    def trace_decision_surface(X, y, t, D=None) -> List[go.Trace]:
        return [
            go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode='markers',
                marker=dict(symbol=np.where(y == -1, 0, 1), color=y, size=D),
                showlegend=False,
            ),
            decision_surface(lambda X: model.partial_predict(X, t), *lims, showscale=False),
        ]

    fig = make_subplots(rows=1, cols=4, subplot_titles=[f"T={t}" for t in T])
    for i, t in enumerate(T):
        fig.add_traces(
            trace_decision_surface(test_X, test_y, t),
            rows=1,
            cols=i + 1,
        )
    fig.update_layout(
        title="Decision surfaces of AdaBoost",
        xaxis=dict(title="x1"),
        yaxis=dict(title="x2"),
    )
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_iter = n_iter[np.argmin(test_e)]
    acc = accuracy(test_y, model.predict(test_X))
    go.Figure(
        data=trace_decision_surface(test_X, test_y, best_iter),
        layout=go.Layout(
            title=f"Decision surface of using {best_iter} learners - accuracy: {acc:.2f}",
            xaxis=dict(title="x1"),
            yaxis=dict(title="x2"),
        ),
    ).show()

    # Question 4: Decision surface with weighted samples
    D = model.D_ / np.max(model.D_) * 5
    go.Figure(
        data=trace_decision_surface(train_X, train_y, n_learners, D),
        layout=go.Layout(
            title=f"Weighted train set",
            xaxis=dict(title="x1"),
            yaxis=dict(title="x2"),
        ),
    ).show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
