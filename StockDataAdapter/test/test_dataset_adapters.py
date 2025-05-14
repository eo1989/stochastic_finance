import StockDataAdapter.visualizations as vis
from StockDataAdapter.stock_price_dataset_adapter import (
    FinancialModelingPrepAdapter,
    YahooFinancialsAdapter,
)


def test_actual_financials_adapter():
    records = {
        "Apple Inc": YahooFinancialsAdapter(
            ticker="aapl", training_set_date_range=("2024-11-01", "2024-11-30")
        ).training_set,
        "Netflix": YahooFinancialsAdapter(
            ticker="nflx", training_set_date_range=("2024-11-01", "2024-11-30")
        ).training_set,
        "Nvidia": FinancialModelingPrepAdapter(
            ticker="nvda", training_set_date_range=("2025-05-01", "2024-05-12")
        ).training_set,
    }

    vis.plot_security_prices(records, "stock price")
