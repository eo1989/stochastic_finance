import asyncio
import enum
import os
import sys
from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime
from typing import Any

# EO contributions
import aiohttp
import pandas as pd
import requests as req
import yfinance as yf
from dotenv import load_dotenv
from yahoofinancials import YahooFinancials

# NOTE: YahooFinancials has a concurrent argument which I dont think gets set
# by default.

load_dotenv()
# dotenv.load_dotenv("~/.config/zsh/.env").
# env_path = f"/Users/eo/.config/zsh/.env"
# _keys = dotenv.dotenv_values(dotenv_path=env_path)
# _key = dotenv.get_key("/Users/eo/.config/zsh/.env", "FINANCIAL_MODELING_PREP_KEY")
# FMP_KEY = _keys["FINANCIAL_MODELING_PREP_KEY"]

__all__ = [
    # ------------------- OG Book API -----------------------
    "Frequency",
    "StockPriceDatasetAdapter",
    "BaseStockPriceDatasetAdapter",
    "YahooFinancialsAdapter",
    "MarketStackAdapter",
    # ------------------- new async API by EO -------------------
    "AsyncStockPriceAdapter",
    # "FMPConfig",
    "FinancialModelingPrepAdapter",
    "YFinanceAdapter",
]


# Helpers
class RequiresAPIKeyMixin:
    """Mixin that lazily fetches & validates an API Key.

    Sub-classes **declare** the env var name via class attribute
    `_API_KEY_ENV` and call `self._require_api_key(explicit_key)` whenever
    they actually need the key.
    """

    # overridden by subclasses
    _API_KEY_ENV: str

    def _require_api_key(self, api_key: str | None) -> str:
        key = api_key or os.getenv(self._API_KEY_ENV, "")
        if not key:
            raise RuntimeError(
                f"{self.__class__.__name__}: supply `api_key` or set env var {self._API_KEY_ENV}"
            )
        return key


# ================================================================
# OG Synchronous adapters (book code slightly patched)
# ================================================================


class Frequency(enum.Enum):
    """Sampling intervals supported by *yahoofinancials*."""

    # NOTE: the intraday enums are specifically for YFinance & FMP
    MINUTE = "1-min"
    FIVE_MINUTE = "5-min"
    # -------------------------------------------------------------------------
    HOURLY = "hourly"
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

    DEFAULT_TICKER: str = "NVDA"

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
      Function to get validation dataset for a given stock symbol (ticker).
      This dataset can be used to train a stock price model.
      Although there is no such restrictions on using it elsewhere.

      Returns
      ----
      A dataframe. Each dataframe has two columns: stock price & time
    """


class BaseStockPriceDatasetAdapter(StockPriceDatasetAdapter, ABC):
    """Caches training/validation dataframes and exposes read-only props."""

    def __init__(self, ticker: str):
        self._ticker = ticker
        self._training_set: pd.DataFrame | None = None
        self._validation_set: pd.DataFrame | None = None

    @abstractmethod
    def _connect_and_prepare(self, date_range: tuple): ...

    """
    This function should be overridden by implementing data source adapter. It
    should connect to the stock price data source and return records within the
    specified date range.
    """

    # ---------------- public proxies ------------------
    @property
    def training_set(self):
        # return self._training_set.copy()
        return (
            self._training_set.copy()
            if self._training_set is not None
            else None
        )

    @property
    def validation_set(self):
        # return self._validation_set.copy()
        return (
            self._validation_set.copy()
            if self._validation_set is not None
            else None
        )


class YahooFinancialsAdapter(BaseStockPriceDatasetAdapter):
    """
    Synchronous Dataset adapter using the *yahoofinancials* package.
    """

    def __init__(
        self,
        ticker: str = StockPriceDatasetAdapter.DEFAULT_TICKER,
        frequency: Frequency = Frequency.DAILY,
        training_set_date_range: tuple[str, str] = ("2020-01-01", "2025-04-31"),
        validation_set_date_range: tuple[str, str] = (
            "2024-11-01",
            "2024-12-01",
        ),
    ):
        super().__init__(ticker)
        self._frequency = frequency
        # self._yf = YahooFinancials(self._ticker,)
        self._yf = YahooFinancials(self._ticker, concurrent=True)

        self._training_set = self._connect_and_prepare(training_set_date_range)
        self._validation_set = self._connect_and_prepare(
            validation_set_date_range
        )

    def _connect_and_prepare(self, date_range: tuple):
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

        df = pd.DataFrame(records["prices"])[["formatted_date", "close"]]
        return df.rename(
            columns={"formatted_date": "time", "close": "stock_price"}
        )


class MarketStackAdapter(RequiresAPIKeyMixin, BaseStockPriceDatasetAdapter):
    """
    Dataset adapter for Market Stack (https://marketstack.com/).
    It can be used for symbols not supported by Yahoo Fiancials.
    """

    # eo
    _API_KEY_ENV = "MARKETSTACK_API_KEY"
    _PAGE_LIMIT = 100

    # dictionary of requests parameters
    _REQ_PARAMS = {
        "access_key": "ce72d47022d573ffb1c47820c7e98f15",
        "limit": 100,
    }

    # REST API url to get EOD quotes
    _EOD_API_URL = "http://api.marketstack.com/v1/eod"

    # REST API url to get list of all stock symbols
    _TICKER_API_URL = "http://api.marketstack.com/v1/tickers"

    # Internal iterator to walk paginated responses ---------------------------
    class _Paginated:
        """
        Market stack API sends paginated response with offset,
        limit & total records.Inner class _PaginatedRecords
        provides a stateful page navigation mechanism to
        iterated over records.
        """

        def __init__(self, url: str, params: dict[str, Any]):
            self.url, self.params = url, params
            ## maybe url = _api_url && params = _req_params.keys() ??
            # self._req_params = req_params
            # self._api_url = api_url
            self.offset = 0
            self.total = sys.maxsize  # unknown until first fetch

        def __iter__(self):
            return self

        def __next__(self):
            if self.offset >= self.total:
                raise StopIteration
            self.params["offset"] = self.offset
            resp = req.get(self.url, self.params).json()
            self.total = resp["pagination"]["total"]
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
        ticker: str,
        training_set_date_range: tuple = ("2020-01-01", "2025-04-30"),
        validation_set_date_range: tuple = (
            "2024-11-01",
            "2024-12-01",
        ),
        api_key: str | None = None,
        # _API_KEY_ENV: str,
    ):
        super().__init__(ticker)
        self.api_key = self._require_api_key(api_key)
        self._training_set = self._connect_and_prepare(training_set_date_range)
        self._validation_set = self._connect_and_prepare(
            validation_set_date_range
        )

    def _connect_and_prepare(self, date_range: tuple):
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
            "access_key": self.api_key,
            "limit": self._PAGE_LIMIT,
            "symbols": self._ticker,
            "date_from": date_range[0],
            "date_to": date_range[1],
        }

        frames: list[pd.DataFrame] = []

        for page in MarketStackAdapter._Paginated(self._EOD_API_URL, params):
            df = pd.DataFrame(page)[["date", "close"]]
            df.rename(columns={"close": "stock price"}, inplace=True)
            df["time"] = df["date"].str.split("T").str[0]
            frames.append(df[["time", "stock price"]])
        return pd.concat(frames, ignore_index=True) if frames else None

    # Convenience lookup ------------------------------------------------------
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

        # key = RequiresAPIKeyMixin._require_api_key(cls, api_key)  # type: ignore[arg-type]
        key = cls._require_api_key(cls, api_key)  # type: ignore[arg-type]
        # params = {"access_key": key, "limit": 500}
        data = req.get(
            cls._TICKER_API_URL, {"access_key": key, "limit": 100}
        ).json()["data"]
        return [d["symbol"] for d in data]


# Async adapter interface
class AsyncStockPriceAdapter(ABC):
    """Minimal contract every *async* price adapter must fulfill."""

    # Internal coroutine - fetch one symbol (subclasses implemented) ----------
    @abstractmethod
    async def _fetch_symbol(self, *args: Any, **kwargs: Any): ...

    # public helper - gather & return tidy DataFrame --------------------------
    @abstractmethod
    def get_stock_price_data(
        self,
        symbols: list[str],
        start_date: datetime | str,
        end_date: datetime | str,
        **kwargs: Any,
    ): ...


# FinancialModelingPrepAdapter async adapter ----------------------------------
class FinancialModelingPrepAdapter(RequiresAPIKeyMixin, AsyncStockPriceAdapter):
    """
    Fetch OHLCV bars from Financial Modeling Prep's `/historical-price-full` endpoint
    .. Financial Modeling Prep: https:/financialmodelingprep.com/developer/docs/
    """

    # url: str = "https://financialmodelingprep.com/api"
    # key: str = field(default_factory=lambda: os.environ.get("FMP_KEY", ""))

    _API_KEY_ENV = "FINANCIAL_MODELING_PREP_KEY"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://financialmodelingprep.com/api/v3",
        timeout: int = 30,
    ):
        self._api_key = self._require_api_key(api_key)
        self.base_url, self.timeout = base_url, timeout

    # -------------------------------------------------------------------------
    async def _fetch_symbol(
        self, session: aiohttp.ClientSession, symbol: str, s: str, e: str
    ):
        url = f"{self.base_url}/historical-price-full/{symbol.upper()}"
        params = {"from": s, "to": e, "apikey": self._api_key}
        async with session.get(url, params=params, timeout=self.timeout) as r:
            r.raise_for_status()
            payload = await r.json(content_type=None)
        if "historical" not in payload:
            raise ValueError(f"FMP returned no data for {symbol}")
        return pd.DataFrame(payload["historical"]).assign(symbol=symbol.upper())

    async def _gather(self, symbols: list[str], s: str, e: str):
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_symbol(session, sym, s, e) for sym in symbols]
            return await asyncio.gather(*tasks)

    def get_stock_price_data(
        self,
        symbols: list[str],
        start_date: datetime | str,
        end_date: datetime | str,
    ):
        # canonicalise dates to YYYY-MM-DD strings understood by FMP
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
        frames = asyncio.run(self._gather(symbols, s, e))
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.sort_values(["symbol", "date"]).reset_index(drop=True)


# YFinance async adapter ------------------------------------------------------
class YFinanceAdapter(AsyncStockPriceAdapter):
    """
    Concurrent downloads from Yahoo Finance via *yfinance*.
    off-load the blocking `yf.download` call to a thread (one per symbol) using
    `asyncio.to_thread`, then gather the results in parallel.
    """

    _DEFAULT_INTERVAL = "1d"

    async def _fetch_symbol(
        self,
        symbol: str,
        s: str,
        e: str,
        interval: str,
    ):
        # Inner blocking function executed in thread-pool ---------------------
        def _blocking_download():
            df = yf.download(
                symbol,
                start=s,
                end=e,
                interval=interval,
                progress=False,
                threads=False,
            )
            if df.empty:
                raise ValueError(f"No Yahoo data for {symbol}")
            df.reset_index(inplace=True)
            df.rename(axis={"Date": "date"}, inplace=True)
            return df.assign(symbol=symbol.upper())

            """
            NOTE:
            Why shouldnt blocking_download() be using threads?
            asnycio.to_thread(...) already lifts the whole yf.download() call into **one worker-thread** of Pythons default ThreadPoolExecutor.
            If you pass `threads = True` to *yfinance* you'd create a second layer of per-ticker threads inside *that* worker-thread. Basically,
            you would end up with something like:

            .. mermaid
            event-loop --> worker-thread \\#1 -> yfinance spins up N-internal threads
            """

        return await asyncio.to_thread(_blocking_download)

    async def _gather(
        self,
        symbols: list[str],
        s: str,
        e: str,
        interval: str,
    ):
        tasks = [self._fetch_symbol(sym, s, e, interval) for sym in symbols]
        return await asyncio.gather(*tasks)

    def get_stock_price_data(
        self,
        symbols: list[str],
        start_date: datetime | str,
        end_date: datetime | str,
        *,
        interval: str | None = None,
    ):
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
        interval = interval or self._DEFAULT_INTERVAL
        frames = asyncio.run(self._gather(symbols, s, e, interval))
        df = pd.concat(frames, ignore_index=True)
        return df.sort_values(["symbol", "date"]).reset_index(drop=True)
