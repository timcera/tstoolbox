"""A lag routine."""

from contextlib import suppress
from typing import List

import pandas as pd
from pydantic import PositiveInt, validate_arguments
from toolbox_utils import tsutils


@tsutils.transform_args(lags=tsutils.make_list)
@validate_arguments
@tsutils.doc(tsutils.docstrings)
def lag(
    lags: List[PositiveInt],
    input_ts="-",
    print_input=False,
    start_date=None,
    end_date=None,
    columns=None,
    clean=False,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    skiprows=None,
):
    """Create a series of lagged time-series.

    Parameters
    ----------
    lags : str, int, or list
        If an integer will calculate all lags up to and including the
        lag number.  If a list will calculate each lag in the list.  If
        a string must be a comma separated list of integers.
    ${print_input}
    ${input_ts}
    ${start_date}
    ${end_date}
    ${clean}
    ${skiprows}
    ${index_type}
    ${names}
    ${source_units}
    ${target_units}
    ${columns}
    ${tablefmt}
    """
    tsd = tsutils.common_kwds(
        input_ts,
        dropna="all",
        skiprows=skiprows,
        names=names,
        index_type=index_type,
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        source_units=source_units,
        target_units=target_units,
        clean=clean,
    )
    if len(lags) == 1:
        lags = lags[0]
    with suppress(TypeError):
        lags = list(range(1, lags + 1))

    if lags == 0:
        return tsd

    ntsd = tsd.copy() if print_input is True else tsd
    ntsd = tsutils.asbestfreq(ntsd)

    cols = {}
    nlags = []
    for i in lags:
        for x in list(ntsd.columns):
            parts = x.split(":")
            parts[0] = f"{parts[0]}_{i}"
            cols.setdefault(x, []).append(":".join(parts))
            nlags.append(i)
    for k, v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=ntsd.index)
        for c, i in zip(columns, lags):
            dfn[c] = ntsd[k].shift(periods=i)
        ntsd = pd.concat([ntsd, dfn], axis=1).reindex(ntsd.index)
    return tsutils.return_input(print_input, tsd, ntsd, "lag")
