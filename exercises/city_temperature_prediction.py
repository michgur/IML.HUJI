import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    
    # Get day of year
    df["DayOfYear"] = df["Date"].dt.dayofyear
    # Remove invalid data
    df = df[df["Temp"] > 0]
    # Convert year to string for discrete color
    df["Year"] = df["Year"].astype(str)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel = df[df["Country"] == "Israel"]

    px.scatter(israel, x="DayOfYear", y="Temp", color="Year", title="Temprature in Israel as a function of day-of-year").show()
    px.bar(
        israel.groupby("Month", as_index=False).agg({"Temp": "std"}),
        x="Month", y="Temp",
        title="Standard deviation of daily temparture by month",
    ).show()

    # Question 3 - Exploring differences between countries
    px.line(
        df.groupby(["Country", "Month"], as_index=False).agg(TempSTD=("Temp", "std"), TempMean=("Temp", "mean")),
        x="Month", y="TempMean", color="Country",
        error_y="TempSTD",
        title="Average daily temperature by month for different countries",
    ).show()

    # Question 4 - Fitting model for different values of `k`
    X, y = israel["DayOfYear"], israel["Temp"]
    train_X, train_y, test_X, test_y = split_train_test(X, y, train_proportion=0.75)
    
    k_values = np.arange(1, 11)
    loss = []

    for k in k_values:
        model = PolynomialFitting(k).fit(train_X, train_y)
        loss.append(
            round(model.loss(test_X, test_y), 2),
        )
    
    k_loss = pd.DataFrame({"k": k_values, "loss": np.array(loss)})
    print(k_loss)
    px.bar(k_loss, x="k", y="loss", title=r"$\text{Loss as a function of polynomial degree }k$").show()

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(5).fit(X, y)
    countries = [country for country in df["Country"].unique() if country != "Israel"]
    loss = []

    for country in countries:
        country_data = df[df["Country"] == country]
        loss.append(
            round(model.loss(country_data["DayOfYear"], country_data["Temp"]), 2),
        )

    country_loss = pd.DataFrame({"Country": countries, "loss": np.array(loss)})
    px.bar(country_loss, x="Country", y="loss", title=r"Loss by country").show()
