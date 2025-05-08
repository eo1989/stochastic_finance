from typing import Any

import matplotlib.pyplot as plt


def plot_security_prices(all_records: dict[str, Any], security_type) -> None:
    plt.style.use("seaborn")
    n = len(all_records)
    rows = int(n / 2)
    cols = 2
    if n == 1:
        rows, cols = 1, 1

    fig, ax = plt.subplots(rows, cols, figsize=(15, 10))
    i, r = 0, 0
    security_names = list(all_records.keys())

    def _axis_plot_security_prices(records, col, name) -> None:
        match n:
            case 1:
                ax.set_title(name)
                records.plot(ax=ax, x="time", y=security_type)
            case 2:
                ax[col].set_title(name)
                records.plot(ax=ax[col], x="time", y=security_type)
            case _:
                ax[r, col].set_title(name)
                records.plot(ax=ax[r, col], x="time", y=security_type)

    while i < n:
        _axis_plot_security_prices(
            all_records[security_names[i]], 0, security_names[i]
        )
        i += 1
        if n > 1:
            _axis_plot_security_prices(
                all_records[security_names[i]], 1, security_names[i]
            )
        # i = i + 1
        i += 1
        # r = r + 1
        r += 1

    fig.tight_layout()
    plt.show()


def plot_returns_for_different_periods(
    ticker, periodic_returns: list[tuple]
) -> None:
    plt.style.use("seaborn")
    fig, ax = plt.subplots(len(periodic_returns), 1, figsize=(15, 10))

    for index, t in enumerate(periodic_returns):
        t[1].plot(ax=ax[index], x="time", y="Return")
        ax[index].set_title(f"{ticker}-{t[0]} Returns")

    fig.tight_layout()
    plt.show()
