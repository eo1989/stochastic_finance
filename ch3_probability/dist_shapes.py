# type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import beta, binom

plt.style.use("seaborn-v0_8")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))


x_pmf = np.arange(0, 15)
# sns.barplot(ax=ax[0], x=x_pmf, y=binom(15, 0.3).pmf(x_pmf), width=0.2)
sns.barplot(ax=ax[0], x=x_pmf, y=binom(15, 0.3).pmf(x_pmf), width=0.2)
ax[0].set(title="PMF", xlabel="x", ylabel="f(x)")

x_pdf = np.linspace(start=0, stop=1, num=100)
beta_rv = beta(a=2, b=5)
sns.lineplot(ax=ax[1], x=x_pdf, y=beta_rv.pdf(x=x_pdf), lw=2)
x_fill = [0.2, 0.24]

ax[1].fill_between(x=x_fill, y1=beta_rv.pdf(x=x_fill))
ax[1].set(title="PDF", xlabel="x", ylabel="f(x)")

fig.tight_layout()
plt.show()