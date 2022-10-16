# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Filling
# ## Fill forward

# %%
import pandas as pd

import tstoolbox.tstoolbox as ts

# %%
dr = pd.date_range(periods=16, start="2010-01-01", freq="H")
df = pd.DataFrame([1.1, 2.3, 2.1] + 10 * [None] + [5.4, 6.5, 7.2], dr)

# %%
df

# %%
ts.plot(input_ts=df, ofilename="base.png", style="r-*")

# %% [markdown]
# ![](base.png)

# %%
ndf = ts.fill(input_ts=df)
ndf

# %%
ts.plot(input_ts=ndf, ofilename="ffill.png", style="r-*")

# %% [markdown]
# ![](ffill.png)

# %%
ndf = ts.fill(input_ts=df, method="bfill")

# %%
ts.plot(input_ts=ndf, ofilename="bfill.png", style="r-*")

# %% [markdown]
# ![](bfill.png)

# %%
