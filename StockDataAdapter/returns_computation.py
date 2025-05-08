import visualizations as vis

from StockDataAdapter.stock_price_dataset_adapter import (
    Frequency,
    YahooFinancialsAdapter,
)


def compute_returns():
    """

    :::math
    \text{Return} := R_{t} = \frac{S_{t}}{S_{t-1}} - 1

    """

    monthly = YahooFinancialsAdapter(frequency=Frequency.MONTHLY).training_set
    monthly["Return"] = (
        monthly["stock price"] / monthly["stock price"].shift(1) - 1
    )

    weekly = YahooFinancialsAdapter(frequency=Frequency.WEEKLY).training_set
    weekly["Return"] = (
        weekly["stock price"] / weekly["stock price"].shift(1) - 1
    )

    daily = YahooFinancialsAdapter(frequency=Frequency.DAILY).training_set
    daily["Return"] = daily["stock price"] / daily["stock price"].shift(1) - 1

    # periodic_returns = [("Daily", daily), ("Weekly", weekly), ("Monthly", monthly)]
    return [("Daily", daily), ("Weekly", weekly), ("Monthly", monthly)]


def test_plot_periodic_returns():
    periodic_returns = compute_returns()
    vis.plot_returns_for_different_periods("Nvidia", periodic_returns)
