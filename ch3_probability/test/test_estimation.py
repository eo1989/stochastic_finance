import numpy as np

from ch2_StockDataAdapter import visualizations as vs2
from ch2_StockDataAdapter.stock_price_dataset_adapter import (
    YahooFinancialsAdapter,
)
from ch3_probability import visualization as vs3
from ch3_probability.estimation import (
    ExponentialLogLikelihoodFunctionAnalysis,
    GaussianLogLikelihoodFunctionAnalysis,
    iterative_gaussian_gaussian_bayesian_estimation_with_prior,
)


def test_exponential_likelihood_func_analysis():
    datasets = [
        {
            "source": "Dataset 1",
            "x": np.linspace(start=200, stop=300, num=1_000),
        },
        {
            "source": "Dataset 2",
            "x": np.linspace(start=2, stop=8, num=1_000),
        },
    ]
    θ_sets = {"λ": np.linspace(start=0, stop=3, num=500)}

    ExponentialLogLikelihoodFunctionAnalysis.for_parameters_and_datasets(
        θ_sets=θ_sets, datasets=datasets
    ).plot()


def test_gaussian_likelihood_func_analysis():
    datasets = [
        {
            "source": "Apple Inc",
            "x": YahooFinancialsAdapter(
                ticker="AAPL",
                training_set_date_range=("2025-02-01", "2025-04-30"),
            ).training_set["stock price"],
        },
        {
            "source": "ADP",
            "x": YahooFinancialsAdapter(
                ticker="ADP",
                training_set_date_range=("2025-02-01", "2025-04-30"),
            ).training_set["stock price"],
        },
    ]

    θ_sets = {
        "μ": np.linspace(start=100.0, stop=200, num=10),
        "σ2": np.linspace(start=100.0, stop=400, num=10),
    }

    GaussianLogLikelihoodFunctionAnalysis.for_parameters_and_datasets(
        θ_sets=θ_sets, datasets=datasets
    ).plot(θ_names=["μ", "σ2"])


def test_iterative_bayesian_estimation():
    # Observations for Bayesian parameter estimation.
    yf_adapter = YahooFinancialsAdapter(
        ticker="AAPL",
        training_set_date_range=("2025-02-01", "2025-04-30"),
    )
    x = yf_adapter.training_set["stock price"]

    σ2 = 100  # assume σ2 is constant & known to us, so no prior distribution is set on this

    # copmpute posterior distributions for a collection of four prior paramter settings (for μ)
    prob_x_arr = [
        iterative_gaussian_gaussian_bayesian_estimation_with_prior(
            x=x,
            prior_α=110,
            prior_Β_2=5,
            σ2=σ2,
        ),
        iterative_gaussian_gaussian_bayesian_estimation_with_prior(
            x=x,
            prior_α=146,
            prior_Β_2=2,
            σ2=σ2,
        ),
        iterative_gaussian_gaussian_bayesian_estimation_with_prior(
            x=x,
            prior_α=150,
            prior_Β_2=20,
            σ2=σ2,
        ),
        iterative_gaussian_gaussian_bayesian_estimation_with_prior(
            x=x,
            prior_α=160,
            prior_Β_2=0.5,
            σ2=σ2,
        ),
    ]
    vs3.bayesian_estimation_plot(prob_x_arr=prob_x_arr)
