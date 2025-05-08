import mkt_notes_general.stochastic_finance.StockDataAdapter.visualizations as vis
from StockDataAdapter.stock_price_dataset_adapter import YahooFinancialsAdapter


def test_yahoo_financials_adapter():
    records = {
        "Apple Inc": YahooFinancialsAdapter(
            ticker="AAPL", training_set_date_range=("2024-09-01", "2024-11-30")
        ).training_set,
        "Google": YahooFinancialsAdapter(
            ticker="GOOGL", training_set_date_range=("2024-09-01", "2024-11-30")
        ).training_set,
    }

    vis.plot_security_prices(records, "stock price")
