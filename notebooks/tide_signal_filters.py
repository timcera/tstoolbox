# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Tide Signal Filters
# ===================
# There are three filters for tidal water elevations or current speeds.  The "tide_doodson" and "tide_usgs" are kernel convolutions against 30 hour kernels.  The "tide_fft" filter is a high pass filter completely damping periods less than 30 hours with a smooth transition to accepting all periods greater than 40 hours.

# %%
# %matplotlib inline
import tstoolbox

# %% jupyter={"outputs_hidden": true}
help(tstoolbox.filter)

# %%
tdf = tstoolbox.filter(
    ["tide_doodson", "tide_usgs", "tide_fft"],
    "lowpass",
    input_ts="data_mayport_8720220_water_level.csv,1",
    print_input=True,
)

# %%
tdf.plot()
