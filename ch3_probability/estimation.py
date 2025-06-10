from abc import ABC, abstractmethod
from typing import Dict, List, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import expon, norm

from ch2_StockDataAdapter.stock_price_dataset_adapter import (
    YahooFinancialsAdapter,
)
from ch3_probability import visualization as vs3


class LogLikelihoodFunctionAnalysis(ABC):
    """
    Base class for Loglikelihood function for any continuous distribution. This should be extended, and _compute_likelihood function should be overridden to have any density-specific behavior.
    This class provides a study of the log-likelihood function with appropriate visualization.
    """

    __instance_key = object()

    class Dataset(TypedDict):
        """
        This specifically typed dictionary works as a key-valued dataset.
        'source' is the name of the data source, and 'x' is the data array.
        """

        source: str
        x: list

    def __init__(
        self, instance_key, θ_sets: dict[str, list], datasets: list[Dataset]
    ):
        assert instance_key == LogLikelihoodFunctionAnalysis.__instance_key, (
            "LogLikelihoodFunctionAnalysis cant be instantiatted explicitly from the outside. Always use the instantiate function."
        )

        self._θ_sets = θ_sets
        self._datasets = datasets
        self._total_loglikelihood = self._compute_total_loglikelihood()
        self._max_loglikelihood_details = self._get_max_loglikelihoods()

    @abstractmethod
    def _compute_total_loglikelihood(self, x, **θ):
        """
        The sub-class should override this function. It should return the likelihood of x.
        You may use the readily available likelihood functions or implement any custom one.
        """

    @classmethod
    def for_parameters_and_datasets(
        cls, θ_sets: dict[str, list], datasets: list[Dataset]
    ):
        """
        Factory function to create a new instance of LogLikelihoodFunctionAnalysis.
        """
        return cls(
            LogLikelihoodFunctionAnalysis.__instance_key, θ_sets, datasets
        )

    @property
    def max_loglikelihood_details(self):
        return self._max_loglikelihood_details

    """
    Function to prepare combinations of parameters for the specific density.
    """

    def _prepare_combinations_for_θ(self) -> dict[str, list]:
        """
        Prepare combination of parameters from the list of supplied simulated values.
        For example, in a two-parameter setting, if the supplied values are [3, 5, 10] and [-6, 8, 190] respectively, then the
        combinations will be (3, -6), (3, 190), (5, 8), (5, -6), etc.
        The function returns combinations as a dictionary qof values, keeping the positional indices intact.
        """
        θ_grid = None
        θ_name_grid_index = {}
        for i, (θ_name, θ_val) in enumerate(self._θ_sets.items()):
            if i == 0:
                θ_grid = np.meshgrid(θ_val)
            else:
                θ_grid = np.meshgrid(θ_grid, θ_val)
            θ_name_grid_index[θ_name] = i

        return {
            θ_name: θ_grid[θ_index].flatten()
            for θ_name, θ_index in θ_name_grid_index.items()
        }

    """
    NOTE: Function to compute total log-likelihood for each of the data sources.
    """

    def _compute_total_loglikelihood(self):
        total_llh = {}

        def _get_single_name_value_for_θ(index, θ_combs):
            return {
                θ_combs_k: θ_combs_v[index]
                for θ_combs_k, θ_combs_v in θ_combs.items()
            }

        θ_combs = self._prepare_combinations_for_θ()
        num_θ_values = len(list(θ_combs.values())[0])

        # Create dictionaries of tuples of format(θ, likelihood) for ea dataset
        for ds in self._datasets:
            llh = [
                (
                    _get_single_name_value_for_θ(i, θ_combs),
                    self.get_loglikelihood_for_observations(
                        ds["x"], **_get_single_name_value_for_θ(i, θ_combs)
                    ),
                )
                for i in range(num_θ_values)
            ]
            total_llh[ds["source"]] = llh

        return total_llh

    def get_loglikelihood_for_observations(self, x, **θ):
        """
        Gets total log-likelihood for a given observation.
        This function can be used for parameter optimization by external components.
        """
        return np.sum(np.log(self._compute_likelihood(x, **θ)))

    """
    NOTE:  _get_max_loglikelihoods simply finds the maximum by iterating over the list of computed log-likelihood values
    for a given set of parameters, and this is the *maximum likelihood estimate (MLE)* found by exploration. Doing it
    this way helps to understand the log-likelihood curve and its asymptotic properties through visual inspection.
    """

    def _get_max_loglikelihoods(self):
        """
        It iterates over all log-likelihoods and returns the maximum for each data source.
        """
        return {
            k: max(v, key=lambda t: t[1])
            for k, v in self._total_loglikelihood.items()
        }

    def plot(self, θ_names: list[str] = None):
        plt.style.use("seaborn-v0_8")

        def _annotate_max_likelihood_point(ax, source):
            max_loglikelihood_point = self._max_loglikelihood_details[source]
            liklihood_val = max_loglikelihood_point[1]
            if len(self._θ_sets) == 1:
                θ_name = list(max_loglikelihood_point[0].keys())[0]
                θ_val = list(max_loglikelihood_point[0].values())[0]
                ax.text(
                    θ_val,
                    liklihood_val,
                    θ_name + " = " + str(round(θ_val, 3)),
                )
            else:
                θ_val_1 = max_loglikelihood_point[0][θ_names[0]]
                θ_val_2 = max_loglikelihood_point[0][θ_names[1]]
                ax.text(θ_val_1, θ_val_2, likelihood_val, "X", color="green")
                ax.text(
                    θ_val_1 + 10,
                    θ_val_2 + 10,
                    likelihood_val + 1,
                    "Optimal (" + θ_names[0] + ",",
                    +θ_names[1]
                    + ") = ("
                    + str(round(θ_val_1))
                    + ","
                    + str(round(θ_val_2))
                    + ")",
                )

        if len(self._θ_sets) == 1:
            θ_names = list(self._θ_sets.keys())[0]
            records_df = pd.DataFrame()
            for source, likelihood_details in self._total_loglikelihood.items():
                record = {}
                record["Source"] = source

                t_arr = np.array(likelihood_details)
                θ_name_val_df = pd.DataFrame.from_records(t_arr[:, 0])

                record[θ_name] = θ_name_val_df[θ_name]
                record["Log Likelihood"] = t_arr[:, 1]

                records_df = pd.concat(
                    [records_df, pd.DataFrame(record)], ignore_index=True
                )
                _annotate_max_likelihood_point(plt.gca(), source)

            sns.lineplot(
                data=records_df,
                x=θ_name,
                y="Log Likelihood",
                hue="Source",
                style="Source",
                lw=3,
            )
        else:
            fig = plt.figure(figsize=(10, 7))
            n = len(self._total_loglikelihood)
            row = int(n / 2)

            def _plot_for_single_sourec(source, likelihood_details, i):
                # ax = fig.add_subplot(
                #     nrows=row, ncols=2,  index = i, projection="3d", computed_zorder=False
                # )
                ax = fig.add_subplot(
                    nrows=row,
                    ncols=2,
                    index=1,
                    subplot_kw={"projection": "3d"},
                    computed_zorder=False,
                )
                t_arr = np.array(likelihood_details)
                θ_name_val_df = pd.DataFrame.from_records(t_arr[:, 0])
                ax.plot_trisurf(
                    θ_name_val_df[θ_names[0]],
                    θ_name_val_df[θ_names[1]],
                    list(t_arr[:, 1]),
                    cmap=plt.cm.gnuplot2,
                    edgecolor="black",
                    linewidth=0.2,
                    zorder=1,
                )
                ax.set(
                    xlabel=θ_names[0],
                    ylabel=θ_names[1],
                    zlabel="Log Likelihood",
                    title=source,
                )
                _annotate_max_likelihood_point(ax, source)

            i = 1
            for source, likelihood_details in self._total_loglikelihood.items():
                _plot_for_single_sourec(source, likelihood_details, i)
                i += 1

            fig.tight_layout()
        plt.show()


class ExponentialLogLikelihoodFunctionAnalysis(LogLikelihoodFunctionAnalysis):
    """
    Class for studying the likelihood function of Exponential distribution with parameter λ.
    """

    def _compute_likelihood(self, x, λ):
        return expon.pdf(
            x, loc=0, scale=1 / λ
        )  # loc = 0 and scale = 1 are the defaults, and *optional*


class GaussianLogLikelihoodFunctionAnalysis(LogLikelihoodFunctionAnalysis):
    """
    Class for studying the likelihood function of Gaussian distribution with parameters μ and σ2.
    """

    def _compute_likelihood(self, x, μ, σ2):
        return norm.pdf(x, loc=μ, scale=np.sqrt(σ2))


def iterative_gaussian_gaussian_bayesian_estimation_with_prior(
    x, prior_α=None, prior_Β_2=None, σ2=None
):
    """
    Bayesian belief update algorithm for Gaussian-Guassian settings.
    """
    posterior_α = 0.0
    posterior_Β_2 = 0.0

    temp_prior_α = prior_α
    temp_prior_Β_2 = prior_Β_2

    # Iteratively cmopute posteriors in closed form for the Gaussian-Guassian settings assuking x is a streaming dataset.
    for x_i in x:
        posterior_Β_2 = (temp_prior_Β_2 * σ2) / (temp_prior_Β_2 + σ2)

        posterior_α = posterior_Β_2 * (
            (x_i / σ2) + (temp_prior_α / temp_prior_Β_2)
        )

        # update priors with computed posteriors for next iterations
        temp_prior_Β_2 = posterior_Β_2
        temp_prior_α = posterior_α

    # draw samples from prior & posterior distributions
    prior_μ_rvs = norm.rvs(loc=prior_α, scale=np.sqrt(prior_Β_2), size=1_000)
    posterior_μ_rvs = norm.rvs(
        loc=posterior_α, scale=np.sqrt(posterior_Β_2), size=1_000
    )

    # compute all sample likelihoods from prior & posterior distributions for vis
    prob_x = {
        "Prior μ": (
            prior_μ_rvs,
            norm.pdf(prior_μ_rvs, loc=prior_α, scale=np.sqrt(prior_Β_2)),
        ),
        "Posterior μ": (
            posterior_μ_rvs,
            norm.pdf(
                posterior_μ_rvs, loc=posterior_α, scale=np.sqrt(posterior_Β_2)
            ),
        ),
    }
    return prob_x, prior_α, posterior_α
