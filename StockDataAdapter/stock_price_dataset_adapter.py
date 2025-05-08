import asyncio
import enum
import os
import sys
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import aiohttp as aio

# EO contributions
import dotenv
import pandas as pd
import requests as req
import yfinance as yf
from fluid.utils.data import compact_dict
from fluid.utils.http_client import AioHttpClient
from yahoofinancials import YahooFinancials

# dotenv.load_dotenv("~/.config/zsh/.env").
_key = dotenv.get_key(
    "/Users/eo/.config/zsh/.env", "FINANCIAL_MODELING_PREP_KEY"
)
FMP_KEY = _key

__all__ = [
    # ------------------- OG Book API -----------------------
    "Frequency",
    "StockPriceDatasetAdapter",
    "BaseStockPriceDatasetAdapter",
    "YahooFinancialAdapter",
    "MarketStackAdapter",
    # ------------------- new async API by EO -------------------
    "AsyncBaseStockPriceAdapter",
    "FMPConfig",
    "FinancialModelingPrepAdapter",
    "YFinanceAdapter",
]


# Helpers
class RequiresAPIKeyMixin:
    """Mixin that lazily fetches & validates an API Key.

    Sub-classes **declare** the env var name via class attribute
    ``_API_KEY_ENV`` and call ``self._require_api_key(explicit_key)`` whenever
    they actually need the key.
    """

    _API_KEY_ENV: str  # overridden by subclasses

    def _require_api_key(self, explicit: str | None = None) -> str:
        key = explicit or os.getenv(self._API_KEY_ENV, "")
        if not key:
            raise RuntimeError(
                f"{self.__class__.__name__} needs an API key in env var "
                f"{self._API_KEY_ENV} or as an argument."
            )
        return key


# ================================================================
# OG Synchronous adapters (book code slightly patched)
# ================================================================
class Frequency(enum.Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class StockPriceDatasetAdapter(
    metaclass=ABCMeta
):  # ABCMeta replaced to silence mypy
    """
    Interface to access any data source of stock price quotes.
    Multiple implementations can be made to support different data sources.
    (training/validation style).
    """

    DEFAULT_TICKER = "NVDA"

    # ---------------------- Abstract interface ----------------------
    @property
    @abstractmethod
    def training_set(self): ...

    # def training_set(self):
    #     raise NotImplementedError

    """
    Property to get training dataset for a given stock symbol (ticker).
    This dataset can be used to train a stock price model.
    Although there are no such restrictions on using it elsewhere.

    Returns

    ----

    A data frame. Each data frame has two columns: stock price & time
    """

    @property
    @abstractmethod
    def validation_set(self): ...

    # def validation_set(self):
    #     raise NotImplementedError

    """
      Function to get validation dataset for a given stock symbol (ticker). This dataset can be used to train a stock price model.
      Although there is no such restrictions on using it elsewhere.

      Returns
      ----
      A dataframe. Each dataframe has two columns: stock price & time
    """


class BaseStockPriceDatasetAdapter(StockPriceDatasetAdapter):
    def __init__(self, ticker: str | None = None) -> None:
        self._ticker = ticker
        self._training_set: pd.DataFrame | None = None
        self._validation_set: pd.DataFrame | None = None

    # ---------------- public proxies ------------------
    @abstractmethod
    def _connect_and_prepare(self, date_range: tuple[str, str]): ...

    """
    This function should be overriden by implementing data source adapter. It should connect to the stock price data source and return records within the specified date range.
    """

    @property
    def training_set(self) -> pd.DataFrame | None:
        # return self._training_set.copy()
        return (
            self._training_set.copy()
            if self._training_set is not None
            else None
        )

    @property
    def validation_set(self) -> pd.DataFrame | None:
        # return self._validation_set.copy()
        return (
            self._validation_set.copy()
            if self._validation_set is not None
            else None
        )


class YahooFinancialsAdapter(BaseStockPriceDatasetAdapter):
    """
    Dataset adapter for Yahoo Financials (finance.yahoo.com)
    """

    def __init__(
        self,
        ticker: str = StockPriceDatasetAdapter.DEFAULT_TICKER,
        frequency: Frequency = Frequency.DAILY,
        training_set_date_range: tuple[str, str] = ("2020-01-01", "2024-12-31"),
        validation_set_date_range: tuple[str, str] = (
            "2019-11-01",
            "2019-12-01",
        ),
    ) -> None:
        super().__init__(ticker=ticker)
        self._frequency = frequency
        self._yf = YahooFinancials(self._ticker)
        self._training_set = self._connect_and_prepare(training_set_date_range)
        self._validation_set = self._connect_and_prepare(
            validation_set_date_range
        )

    def _connect_and_prepare(self, date_range: tuple[str, str]) -> pd.DataFrame:
        records = self._yf.get_historical_price_data(
            date_range[0], date_range[1], self._frequency.value
        )[self._ticker]
        # stock_price_records = pd.DataFrame(data=records["prices"])[
        #     ["formatted_date", "close"]
        # ]
        # rename columns for convenience, you will be thankful later
        # stock_price_records.rename(
        #     columns={"formatted_date": "time", "close": "stock price"}, inplace=True
        # )
        # return stock_price_records

        # df = pd.DataFrame(records["prices"])[["formatted_date", "close"]]
        # return df.rename(columns={"formatted_date": "time", "close": "stock_price"})

        # df is actually used elsewhere so its best to lave it as-is, even if it is scoped locally..
        df = [
            pd.DataFrame(records["prices"])[["formatted_date", "close"]].rename(
                columns={"formatted_date": "time", "close": "stock_price"}
            )
        ]
        return df


class MarketStackAdapter(RequiresAPIKeyMixin, BaseStockPriceDatasetAdapter):
    """
    Dataset adapter for Market Stack (https://marketstack.com/).
    It can be used for symbols not supported by Yahoo Fiancials.
    """

    # eo
    _API_KEY_ENV = "MARKETSTACK_API_KEY"
    _PAGE_LIMIT = 500
    # eo

    # dictionary of requests parameters
    _REQ_PARAMS = {
        "access_key": "ce72d47022d573ffb1c47820c7e98f15",
        "limit": 500,
    }

    # REST API url to get EOD quotes
    _EOD_API_URL = "http://api.marketstack.com/v1/eod"

    # REST API url to get list of all stock symbols
    _TICKER_API_URL = "http://api.marketstack.com/v1/tickers"

    class _Paginated:
        """
        Market stack API sends paginated response with offset,
        limit & total records.Inner class _PaginatedRecords
        provides a stateful page navigation mechanism to
        iterated over records.
        """

        def __init__(self, url: str, params: dict):
            self.url, self.params = url, params
            ## maybe url = _api_url && params = _req_params.keys() ??
            # self._req_params = req_params
            # self._api_url = api_url
            self.offset = 0
            self.total = sys.maxsize

        def __iter__(self):
            return self

        def __next__(self):
            if self.offset >= self.total:
                raise StopIteration
            self.params["offset"] = self.offset
            resp = req.get(self.url, self.params).json()
            self.ftotal = resp["pagination"]["total"]
            self.offset += MarketStackAdapter._PAGE_LIMIT
            return resp["data"]

        # def __getitem__(self, index):
        #     """
        #     Ducktyped function to get the current page records &
        #     increment the offset accordingly
        #     """

        #     if (self._offset + self._req_params["limit"]) >= self._total_records:
        #         raise StopIteration()

        #     self._req_params["offset"] = self._offset
        #     api_response = req.get(self._api_url, self._req_params).json()
        #     self._total_records = api_response["pagination"]["total"]
        #     self._offset = self._offset + self._req_params["limit"] + 1
        #     return api_response["data"]

    def __init__(
        self,
        ticker: str | None = None,
        training_set_date_range: tuple[str, str] = ("2020-01-01", "2024-12-31"),
        validation_set_date_range: tuple[str, str] = (
            "2019-11-01",
            "2019-12-01",
        ),
        api_key: str | None = None,
    ):
        super().__init__(ticker)
        self._api_key = self._require_api_key(api_key)
        self._training_set = self._connect_and_prepare(training_set_date_range)
        self._validation_set = self._connect_and_prepare(
            validation_set_date_range
        )

    def _connect_and_prepare(self, date_range: tuple[str, str]):
        # def _extract_stock_price_details(stock_price_records, page):
        #     """
        #     Inner function to extract fields: 'close', 'date', 'symbol' of current element obtained from json response.
        #     """
        #     ticker_symbol = page["symbol"]
        #     stock_record_per_symbol = stock_price_records.get(ticker_symbol)
        #     if stock_record_per_symbol is None:
        #         stock_record_per_symbol = pd.DataFrame()

        #     entry = {
        #         "stock price": [page["close"]],
        #         "time": [page["date"].split("T")[0]],
        #     }

        #     stock_price_records[ticker_symbol] = pd.concat(
        #         [stock_record_per_symbol, pd.DataFrame(entry)], ignore_index=True
        #     )
        #     return stock_price_records

        # if self._ticker is None:
        #     return None

        # req_params = MarketStackAdapter._REQ_PARAMS.copy()
        # req_params["symbols"] = self._ticker
        # req_params["date_from"] = date_range[0]
        # req_params["date_to"] = date_range[1]
        # stock_price_records = {}

        # # Iterated over response and fetch records to populate a custom dataframe
        # for records in MarketStackAdapter._PaginatedRecords(
        #     api_url=MarketStackAdapter._EOD_API_URL, req_params=req_params
        # ):
        #     for page in records:
        #         stock_price_records = _extract_stock_price_details(
        #             stock_price_records, page
        #         )

        # return stock_price_records

        if self._ticker is None:
            return None

        params = {
            "access_key": self._api_key,
            "limit": self._PAGE_LIMIT,
            "symbols": self._ticker,
            "date_from": date_range[0],
            "date_to": date_range[1],
        }

        frames: list[pd.DataFrame] = []

        for page in MarketStackAdapter._Paginated(self._EOD_API_URL, params):
            df = pd.DataFrame(page)[["date", "close"]]
            df = df.rename(columns={"close": "stock price"})
            df["time"] = df["date"].str.split("T").str[0]
            frames.append(df[["time", "stock price"]])
        return pd.concat(frames, ignore_index=True) if frames else None

    @classmethod
    def get_samples_of_available_tickers(
        cls, api_key: str | None = None
    ) -> list[str]:
        """
        Function to get a collection of available symbols from MarketStack.
        Pagination support can be added as enhancement as well.

        """
        # api_response = req.get(
        #     MarketStackAdapter._TICKER_API_URL, MarketStackAdapter._REQ_PARAMS
        # ).json()
        # return [record["symbol"] for record in api_response["data"]]

        key = RequiresAPIKeyMixin._require_api_key(cls, api_key)  # type: ignore[arg-type]
        params = {"access_key": key, "limit": 500}
        data = req.get(cls._TICKER_API_URL, params).json()["data"]
        return [d["symbol"] for d in data]


# Async adapters (mixins not required here)
class AsyncStockPriceAdapter(ABC):
    @abstractmethod
    async def _fetch_symbol(self, *args: Any, **kwargs: Any): ...

    @abstractmethod
    def get_stock_price_data(self, *args: Any, **kwargs: Any): ...


class FinancialModelingPrepAdaptor(RequiresAPIKeyMixin, AsyncStockPriceAdapter):
    """
    FMP API Client Adaptor.

    Fetch market and financial data from `Financial Modeling Prep`_
    ... _ Financial Modeling Prep: https:/financialmodelingprep.com/developer/docs/
    """

    # url: str = "https://financialmodelingprep.com/api"
    # key: str = field(default_factory=lambda: os.environ.get("FMP_KEY", ""))

    _API_KEY_ENV = "FMP_API_KEY"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://financialmodelingprep.com/api/v3",
        timeout: int = 30,
    ):
        self._api_key = self._require_api_key(api_key)
        self.base_url, self.timeout = base_url, timeout

    # async helpers
    async def _fetch_symbol(self, session, symbol: str, start: str, end: str):
        url = f"{self.base_url}/historical-price-full/{symbol.upper()}"
        params = {"from": start, "to": end, "apikey": self._api_key}
        async with session.get(url, params=params, timeout=self.timeout) as r:
            r.raise_for_status()
            payload = await r.json(content_type=None)
        if "historical" not in payload:
            raise ValueError(f"No data for {symbol}: {payload}")
        return pd.DataFrame(payload["historical"]).assign(symbol=symbol.upper())

    async def _gather(self, symbols: list[str], start: str, end: str):
        async with aio.ClientSession() as sess:
            tasks = [self._fetch_symbol(sess, s, start, end) for s in symbols]
            return await asyncio.gather(*tasks)

    def get_stock_price_data(self, symbols: list[str], start_date, end_date):
        s = (
            start_date.strftime("%Y-%m-%d")
            if isinstance(start_date, datetime)
            else start_date
        )
        e = (
            end_date.strftime("%Y-%m-%d")
            if isinstance(end_date, datetime)
            else end_date
        )
        dfs = asyncio.run(self._gather(symbols, s, e))
        df = pd.concat(dfs, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.sort_values(["symbol", "date"]).reset_index(drop=True)


class YFinanceAdapter(AsyncStockPriceAdapter):
    _DEFAULT_INTERVAL = "1d"

    async def _fetch_symbol(
        self, symbol: str, start: str, end: str, interval: str
    ):
        def blocking_download():
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                threads=False,
            )
            """
            NOTE:
            Why shouldnt blocking_download() be using threads?
            asnycio.to_thread(...) already lifts the whole yf.download() call into **one worker-thread** of Pythons default ThreadPoolExecutor.
            If you pass `threads = True` to *yfinance* you'd create a second layer of per-ticker threads inside *that* worker-thread. Basically,
            you would end up with something like:
            ::mermaid
            event-loop --> worker-thread \#1 -> yfinance spins up N-internal threads

            """
