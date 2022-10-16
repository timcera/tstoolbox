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
# # tstoolbox: Statistics
# I will use the water level record for USGS site '02240000 OCKLAWAHA RIVER NEAR CONNER, FL'.  Use tsgettoolbox to download the data we will be using.

# %%
from tsgettoolbox import tsgettoolbox

from tstoolbox import tstoolbox

# %%
ock_flow = tsgettoolbox.nwis(
    sites="02240000", startDT="2008-01-01", endDT="2016-01-01", parameterCd="00060"
)

# %%
ock_flow.head()

# %%
ock_stage = tsgettoolbox.nwis(
    sites="02240000", startDT="2008-01-01", endDT="2016-01-01", parameterCd="00065"
)

# %%
ock_stage.head()

# %% [markdown]
# The tstoolbox.rolling_window calculates the average stage over the spans listed in the 'span' argument.

# %%
r_ock_stage = tstoolbox.rolling_window(
    input_ts=ock_stage, span="1,14,30,60,90,120,180,274,365", statistic="mean"
)
a_ock_stage = tstoolbox.aggregate(
    input_ts=r_ock_stage, agg_interval="A", statistic="max"
)

# %%
a_ock_stage

# %%
q_max_avg_obs = tstoolbox.calculate_fdc(input_ts=a_ock_stage, sort_order="descending")

# %%
q_max_avg_obs

# %%
tstoolbox.plot(
    input_ts=q_max_avg_obs,
    ofilename="q_max_avg.png",
    type="norm_xaxis",
    xtitle="Annual Exceedence Probability (%)",
    ytitle="Stage (feet)",
    title="N-day Maximum Rolling Window Average Stage for 0224000",
    legend_names="1,14,30,60,90,120,180,274,365",
)

# %% [markdown]
# ![](q_max_avg.png)

# %%
combined = ock_flow.join(ock_stage)
tstoolbox.plot(
    input_ts=combined,
    type="xy",
    ofilename="stage_flow.png",
    legend=False,
    title="Plot of Daily Stage vs Flow for 02240000",
    xtitle="Flow (cfs)",
    ytitle="Stage (ft)",
)

# %% [markdown]
# ![](stage_flow.png)

# %% [markdown]
# Note the signficant effect of tail-water elevation.

# %%
