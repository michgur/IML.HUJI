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
    mu_distance = np.array([
        abs(UnivariateGaussian().fit(X[:m]).mu_ - mu)
        for m in sample_sizes
    ])

    go.Figure(
        [go.Scatter(x=sample_sizes, y=mu_distance, mode='markers+lines')],
        layout=go.Layout(
            title=r"$\text{Q2. Absolute distance between estimation and true expectation of }\mathcal{N}(10, 1)\text{ distribution, as a function of sample size}$",
            titlefont=dict(size=12),
            xaxis_title=r"$\text{sample size}$", 
            yaxis_title=r"$|\hat\mu-\mu|$",
            height=300
        )
    ).write_image("./exercises/images/ex1/q2.png")

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
    ).write_image("./exercises/images/ex1/q3.png")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([
        [  1, 0.2, 0, 0.5],
        [0.2,   2, 0,   0],
        [  0,   0, 1,   0],
        [0.5,   0, 0,   1]
    ])
    X = np.random.multivariate_normal(mu, cov, size=1000)

    model = MultivariateGaussian().fit(X)
    print(f"{model.mu_}\n{model.cov_}")

    # Question 5 - Likelihood evaluation
    f = np.linspace(-10, 10, 200)
    log_likelihood = np.array([
        [
            MultivariateGaussian.log_likelihood(
                mu=np.array([f1, 0, f3, 0]),
                cov=cov,
                X=X,
            ) for f3 in f
        ] for f1 in f
    ])

    go.Figure(
        [go.Heatmap(x=f, y=f, z=log_likelihood)],
        layout=go.Layout(
            title=r"$\text{Q5. Multivariate Gaussian log-likelihood of expectation }\hat\mu=(f_1, 0, f_3, 0)$",
            xaxis_title=r"$f_1$", 
            yaxis_title=r"$f_3$",
            height=300
        )
    ).write_image("./exercises/images/ex1/q5.png")

    # Question 6 - Maximum likelihood
    maximum_likelihood = np.unravel_index(np.argmax(log_likelihood), log_likelihood.shape)
    f1, f3 = f[maximum_likelihood[0]], f[maximum_likelihood[1]]

    print(f"{f1:.3f}, {f3:.3f}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
