# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # tstoolbox: Statistics
# I will use the water level record for USGS site '02240000 OCKLAWAHA RIVER NEAR CONNER, FL'.  Use tsgettoolbox to download the data we will be using.

# %%
# %matplotlib inline

# %%
# Third party imports
from plottoolbox import plottoolbox
from tsgettoolbox import tsgettoolbox

# First party imports
from tstoolbox import tstoolbox
from tstoolbox.toolbox_utils.src.toolbox_utils import tsutils

# %%
ock_flow = tsgettoolbox.wdfn_daily(
    monitoring_location_id="USGS-02240000",
    time="2008-01-01/2016-01-01",
    parameter_code="00060",
)

# %%
ock_flow.head()

# %%
ock_stage = tsgettoolbox.wdfn_daily(
    monitoring_location_id="USGS-02240000",
    time="2008-01-01/2016-01-01",
    parameter_code="00065",
)

# %%
ock_stage.head()

# %% [markdown]
# The tstoolbox.rolling_window calculates the average stage over the spans listed in the 'span' argument.

# %%
r_ock_stage = tstoolbox.rolling_window(
    input_ts=ock_stage, span=[1, 14, 30, 60, 90, 120, 180, 274, 365], statistic="mean"
)
a_ock_stage = tstoolbox.aggregate(
    input_ts=r_ock_stage,
    agg_interval=tsutils.pandas_offset_by_version("YE"),
    statistic="max",
)

# %%
a_ock_stage

# %%
q_max_avg_obs = tstoolbox.calculate_fdc(input_ts=a_ock_stage, sort_values="descending")

# %%
q_max_avg_obs

# %%
plottoolbox.norm_xaxis(
    input_ts=q_max_avg_obs,
    ofilename="q_max_avg.png",
    xtitle="Annual Exceedence Probability (%)",
    ytitle="Stage (feet)",
    title="N-day Maximum Rolling Window Average Stage for 0224000",
    legend_names=["1", "14", "30", "60", "90", "120", "180", "274", "365"],
)

# %%
combined = ock_flow.join(ock_stage)
combined.head()

# %%
plottoolbox.xy(
    input_ts=combined,
    ofilename="stage_flow.png",
    legend=False,
    title="Plot of Daily Stage vs Flow for 02240000",
    xtitle="Flow (cfs)",
    ytitle="Stage (ft)",
    linestyles="",
    markerstyles=".",
)

# %% [markdown] jupyter={"outputs_hidden": true}
# Note the signficant impact of tail-water elevation.

# %%
