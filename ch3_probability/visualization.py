import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import poisson


def plot_actual(records, ticker_name):
    plt.style.use("seaborn-v0_8")
    records.plot(x="time", y="stock price")
    plt.title(f"Actual stock prices for {ticker_name}")
    plt.show()


def plot_uniform_and_exponential_cf(ϕ_ω):
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(1, 2)
    sns.lineplot(
        ax=ax[0],
        data=ϕ_ω["Exponential"],
        x="ω",
        y="ϕ(ω)",
        hue="λ",
        style="λ",
        lw=2,
    )
    sns.lineplot(
        ax=ax[1],
        data=ϕ_ω["Uniform"],
        x="ω",
        y="ϕ(ω)",
        hue="(b, a)",
        style="(b, a)",
        lw=2,
    )
    ax[0].set_title("Exponential")
    ax[1].set_title("Uniform")
    fig.tight_layout()
    plt.show()


def plot_cf(cf):
    plt.style.use("seaborn-v0_8")
    n = len(cf)
    row = int(n / 2)
    fig, ax = plt.subplots(row, 2)
    i = 0
    r = 0

    def _axis_plot_cf(record, col):
        ax[r, col].set_title(record[0])
        sns.lineplot(ax=ax[r, col], data=record[1], x="ω", y="ϕ(ω)", lw=2)

    with i < n:
        _axis_plot_cf(cf[i], 0)
        i = i + 1
        _axis_plot_cf(cf[i], 1)
        i = i + 1
        r = r + 1

    fig.tight_layout()
    plt.show()


def poisson_plot(poisson_lambda_x_prob):
    plt.style.use("seaborn-v0_8")
    lamdas = list(poisson_lambda_x_prob.keys())
    n = len(lamdas)
    row = int(n / 2)
    fig, ax = plt.subplots(row, 2)
    i = 0
    r = 0

    def _axis_plot_lambda(record, col):
        ax[r, col].set_title(f"λ = {str(lamdas[i])}")
        ax[r, col].vlines(x=record[0], ymin=0, ymax=record[1])
        ax[r, col].set_xlabel("X")
        ax[r, col].set_ylabel("Probs")

    while i < n:
        _axis_plot_lambda(poisson_lambda_x_prob[lamdas[i]], 0)
        i = i + 1
        _axis_plot_lambda(poisson_lambda_x_prob[lamdas[i]], 1)
        i = i + 1
        r = r + 1

    fig.tight_layout()
    plt.show()


def uniform_plot(x, probs):
    plt.style.use("seaborn-v0_8")
    sns.lineplot(
        data=pd.DataFrame.from_dict({"X": x, "Probs": probs}), x="X", y="Probs"
    )
    plt.show()


def exponential_plot(exponential_x_prob):
    records_df = pd.DataFrame()
    for e in exponential_x_prob.keys():
        record = {}
        record["λ"] = [e] * len(exponential_x_prob[e][0])
        record["X"] = exponential_x_prob[e][0]
        record["Prob Density"] = exponential_x_prob[e][1]

        records_df = pd.concat(
            [records_df, pd.DataFrame(record)], ignore_index=True
        )

    plt.style.use("seaborn-v0_8")
    sns.color_palette("bright")
    sns.lineplot(
        data=records_df, x="X", y="Prob Density", hue="λ", style="λ", lw=3
    )
    plt.show()


def gaussian_plot(gaussian_x_prob, key="", ax=None):
    records_df = pd.DataFrame()
    for e in gaussian_x_prob.keys():
        record = {}
        record[key] = [e] * len(gaussian_x_prob[e][0])
        record["X"] = gaussian_x_prob[e][0]
        record["Prob"] = gaussian_x_prob[e][1]

        records_df = pd.concat(
            [records_df, pd.DataFrame(record)], ignore_index=True
        )

    if ax is None:
        plt.style.use("seaborn-v0_8")
    sns.lineplot(
        ax=ax, data=records_df, x="X", y="Prob", hue=key, style=key, lw=3
    )
    if ax is None:
        plt.show()


def bayesian_estimation_plot(prob_x_arr):
    plt.style.use("seaborn-v0_8")
    n = len(prob_x_arr)
    row = int(n / 2)
    fig, ax = plt.subplots(row, 2)
    i = 0
    r = 0

    def _anntate_prior_posterior(ax):
        ax.axvline(prob_x_arr[i][1], color="r", ls=":")
        ax.text(
            prob_x_arr[i][1],
            0.99,
            "prior α=" + str(round(prob_x_arr[i][1])),
            ha="right",
            va="top",
            rotation=90,
            transform=ax.get_xaxis_transform(),
        )
        ax.axvline(prob_x_arr[i][2], color="r", ls=":")
        ax.text(
            prob_x_arr[i][2],
            0.99,
            "posterior α=" + str(round(prob_x_arr[i][2])),
            ha="right",
            va="top",
            rotation=90,
            transform=ax.get_xaxis_transform(),
        )

    while i < n:
        gaussian_plot(prob_x_arr[i][0], key="Distribution Type", ax=ax[r, 0])
        _anntate_prior_posterior(ax=ax[r, 0])

        i = i + 1
        gaussian_plot(prob_x_arr[i][0], key="Distribution Type", ax=ax[r, 1])
        _anntate_prior_posterior(ax=ax[r, 1])

        i = i + 1
        r = r + 1

    fig.tight_layout()
    plt.show()
