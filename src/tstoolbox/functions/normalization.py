"""Collection of functions for the manipulation of time series."""

from typing import Literal

import pandas as pd

from ..toolbox_utils.src.toolbox_utils import tsutils

try:
    from pydantic import validate_arguments
except ImportError:
    from pydantic import validate_call as validate_arguments


@validate_arguments
@tsutils.doc(tsutils.docstrings)
def normalization(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    mode: Literal[
        "minmax", "zscore", "pct_rank", "maxabs", "normal", "robust"
    ] = "minmax",
    min_limit: float = 0,
    max_limit: float = 1,
    pct_rank_method: Literal["average", "min", "max", "first", "dense"] = "average",
    print_input=False,
    round_index=None,
    source_units=None,
    target_units=None,
    with_centering=True,
    with_scaling=True,
    quantile_range=(0.25, 0.75),
):
    """Return the normalization of the time series.

    This scales the time-series.

    Parameters
    ----------
    mode : str
        [optional, default is 'minmax']

        minmax
            min_limit +
            (X-Xmin)/(Xmax-Xmin)*(max_limit-min_limit)

        zscore
            (X-mean(X))/stddev(X)

        pct_rank
            rank(X)*100/N

        maxabs
            Scale by absolute value between -1 and 1.

        normal
            Scale to unit normal.

        robust
            Robust scale to ranked quantile ranges.

    min_limit : float
        [optional, defaults to 0, used for mode=minmax]

        Defines the minimum limit of the minmax normalization.

    max_limit : float
        [optional, defaults to 1, used for mode=minmax]

        Defines the maximum limit of the minmax normalization.

    pct_rank_method : str
        [optional, defaults to 'average']

        Defines how tied ranks are broken.  Can be 'average', 'min', 'max',
        'first', 'dense'.

    with_centering : bool
        [optional, defaults to True, used when mode=robust]

        If True, center the data before scaling.

    with_scaling : bool
        [optional, defaults to True, used when mode=robust]

        If True, scale the data to interquartile range.

    quantile_range : tuple
        [optional, defaults to (0.25, 0.75)
        (q_min, q_max), 0.0 < q_min < q_max < 100.0]

        Quantile range used to calculate scale.

    ${input_ts}

    ${columns}

    ${start_date}

    ${end_date}

    ${dropna}

    ${skiprows}

    ${index_type}

    ${names}

    ${clean}

    ${print_input}

    ${float_format}

    ${source_units}

    ${target_units}

    ${round_index}

    ${tablefmt}
    """
    tsd = tsutils.common_kwds(
        input_ts,
        skiprows=skiprows,
        names=names,
        index_type=index_type,
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        source_units=source_units,
        target_units=target_units,
        clean=clean,
    )

    # Trying to save some memory
    otsd = tsd.copy() if print_input else pd.DataFrame()
    if mode == "minmax":
        tsd = min_limit + (tsd - tsd.min()) / (tsd.max() - tsd.min()) * (
            max_limit - min_limit
        )
    elif mode == "zscore":
        tsd = (tsd - tsd.mean()) / tsd.std()
    elif mode == "pct_rank":
        tsd = tsd.rank(method=pct_rank_method, pct=True)
    elif mode == "maxabs":
        from sklearn.preprocessing import MaxAbsScaler

        tsd.loc[:, :] = MaxAbsScaler().fit_transform(tsd)
    elif mode == "normal":
        from sklearn.preprocessing import Normalizer

        tsd.loc[:, :] = Normalizer().fit_transform(tsd)
    elif mode == "robust":
        from sklearn.preprocessing import RobustScaler

        tsd.loc[:, :] = RobustScaler(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
        ).fit_transform(tsd)
    tsd = tsutils.memory_optimize(tsd)
    return tsutils.return_input(print_input, otsd, tsd, f"{mode}")
