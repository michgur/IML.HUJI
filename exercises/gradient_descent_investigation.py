import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import misclassification_error

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
from sklearn.metrics import roc_curve, auc


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []
    return lambda **kwargs: (values.append(kwargs["val"]), weights.append(kwargs["weights"])), values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    path_figures = []
    convergence = []
    for eta in etas:
        path_figures.append([])
        convergence.append([])
        for f in [L1(init.copy()), L2(init.copy())]:
            callback, values, weights = get_gd_state_recorder_callback()
            weights.append(init)
            GradientDescent(FixedLR(eta), callback=callback).fit(f, None, None)
            path = plot_descent_path(type(f), np.array(weights), f"Module: {type(f).__name__}, Eta: {eta}")
            path_figures[-1].append(path)
            convergence[-1].append(values)

    path_subplots = make_subplots(2, len(etas))
    for i in range(len(path_figures)):
        for j in range(len(path_figures[i])):
            path_subplots.add_traces(path_figures[i][j].data, rows=j + 1, cols=i + 1)
            path_subplots.update_xaxes(title_text=f"eta={etas[i]}", row=1, col=i + 1)
            path_subplots.update_xaxes(title_text=f"eta={etas[i]}", row=2, col=i + 1)
            path_subplots.update_yaxes(title_text=f"L{j + 1}", row=j + 1, col=i + 1)
    path_subplots.update_layout(showlegend=False, title_text="Fixed Learning Rate - L1/L2 Learning Path with Different Step Sizes")

    path_subplots.show()

    convergence_subplots = make_subplots(1, 2)
    for i in range(2):
        xrange = np.arange(len(max(convergence, key=lambda x: len(x[i]))[i]))
        convergence_subplots.add_traces(
            [
                go.Scatter(
                    x=xrange,
                    y=convergence[j][i],
                    name=f"Eta={etas[j]}",
                    mode="lines",
                    line_color=DEFAULT_PLOTLY_COLORS[j],
                    showlegend=i == 0
                )
                for j in range(len(etas))
                # if j == 2 # uncomment to plot only eta=0.01
            ],
            rows=1, cols=i + 1
        )
        convergence_subplots.update_xaxes(title_text="iteration", row=1, col=i + 1)
        convergence_subplots.update_yaxes(title_text=f"L{i+1}", row=1, col=i + 1)
    convergence_subplots.update_layout(title_text="Convergence of L1/L2 norm with Different Step Sizes as a Function of GD Iterations")
    convergence_subplots.show()



def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    convergence = []
    for gamma in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        weights.append(init)
        f = L1(init.copy())
        GradientDescent(ExponentialLR(eta, gamma), callback=callback).fit(f, None, None)
        convergence.append(values)
        if gamma == 0.95:
            path = plot_descent_path(L1, np.array(weights), f"Module: L1, Eta: {eta}, Gamma: {gamma}")

    # Plot algorithm's convergence for the different values of gamma
    xrange = np.arange(len(max(convergence, key=len)))
    fig = go.Figure(
        data=[
            go.Scatter(
                x=xrange,
                y=convergence[i],
                name=f"Gamma={gammas[i]}",
                mode="lines",
            )
            for i in range(len(gammas))
        ],
    )
    fig.update_xaxes(title_text="iteration")
    fig.update_yaxes(title_text="L1")
    fig.update_layout(title_text="Convergence of L1 norm with Different Decay Rates as a Function of GD Iterations")
    fig.show()

    # Plot descent path for gamma=0.95
    path.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)
    # taken from lab 4
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    go.Figure(
        data=[
            go.Scatter(
                x=fpr, y=tpr, mode='markers+lines',text=thresholds, name="", showlegend=False, marker_size=5,
                hovertemplate="<b>Threshold:</b>%{text:.5f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}"
            ),
        ],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$", 
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"),
        )
    ).show()
    # find threshold that maximizes TPR - FPR
    model.alpha_ = thresholds[np.argmax(tpr - fpr)]
    print(f"Threshold that maximizes TPR - FPR: {model.alpha_:.5f}")
    print(f"Error rate: {model.loss(X_test, y_test):.5f}")


    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    gd = GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4), tol=1e-9)
    for penalty in ('l1', 'l2'):
        lam = min(
            [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
            key=lambda lam: cross_validate(
                LogisticRegression(penalty=penalty, lam=lam, alpha=0.5, solver=gd),
                X_train, y_train,
                scoring=misclassification_error
            )[1] # select by validation error
        )
        print(f"Best lambda for {penalty}-regularized logistic regression: {lam}")
        model = LogisticRegression(penalty=penalty, lam=lam, alpha=0.5, solver=gd).fit(X_train, y_train)
        print(f"Error rate: {model.loss(X_test, y_test):.5f}")
        

if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
