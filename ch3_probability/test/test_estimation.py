import ch2_StockDataAdapter.visualization as vs2
import numpy as np

import ch3_probability.visualization as vs3
from ch2_StockDataAdapter.stock_price_dataset_adapter import (
    YahooFinancialsAdapter,
)
from ch3_probability.estimation import (
    ExponentialLogLikelihoodFunctionAnalysis,
    GaussianLogLikelihoodFunctionAnalysis,
    iterative_gaussian_gaussian_bayesian_estimation_with_prior,
)

def