import sys
sys.path.append("./")

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, var = 10, 1
    X = np.random.normal(mu, var, size=1000)

    model = UnivariateGaussian().fit(X)
    print(f"({model.mu_}, {model.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = np.linspace(10, 1000, 100, dtype=int)
    mu_distance = [
        abs(UnivariateGaussian().fit(X[:m]).mu_ - mu)
        for m in sample_sizes
    ]

    go.Figure(
        [go.Scatter(x=sample_sizes, y=mu_distance, mode='markers+lines')],
        layout=go.Layout(
            title=r"$\text{Q2. Absolute distance between estimation and true expectation of} \mathcal{N}(10, 1) \text{ distribution, as a function sample size}$",
            xaxis_title=r"$\text{samples size}$", 
            yaxis_title=r"$|\hat\mu-\mu|$",
            height=300
        )
    # ).show()
    )

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = model.pdf(X)

    go.Figure(
        [go.Scatter(x=X, y=pdf, mode='markers')],
        layout=go.Layout(
            title=r"$\text{Q3. PDF of fitted model}$",
            xaxis_title=r"$\text{sample value}$", 
            yaxis_title=r"$\text{estimated PDF}$",
            height=300
        )
    # ).show()
    )


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
