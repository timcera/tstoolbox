# -*- coding: utf-8 -*-
"""A lag routine."""

from __future__ import absolute_import, print_function

from typing import List

import mando
import pandas as pd
import typic
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils


@mando.command("lag", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def lag_cli(
    lags,
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
    tablefmt="csv",
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
    tsutils.printiso(
        lag(
            lags,
            input_ts=input_ts,
            print_input=print_input,
            start_date=start_date,
            end_date=end_date,
            columns=columns,
            clean=clean,
            index_type=index_type,
            names=names,
            source_units=source_units,
            target_units=target_units,
            skiprows=skiprows,
        ),
        tablefmt=tablefmt,
    )


@typic.constrained(ge=0)
class Lags(int):
    """'lags' constraint"""


@tsutils.transform_args(lags=tsutils.make_list)
@typic.al
@tsutils.copy_doc(lag_cli)
def lag(
    lags: List[Lags],
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
    """Create a series of lagged time-series."""
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
    try:
        lags = list(range(1, lags + 1))
    except TypeError:
        pass

    if lags == 0:
        return tsd

    if print_input is True:
        ntsd = tsd.copy()
    else:
        ntsd = tsd

    ntsd = tsutils.asbestfreq(ntsd)

    cols = {}
    nlags = []
    for i in lags:
        for x in list(ntsd.columns):
            parts = x.split(":")
            parts[0] = "{}_{}".format(parts[0], i)
            cols.setdefault(x, []).append(":".join(parts))
            nlags.append(i)
    for k, v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=ntsd.index)
        for c, i in zip(columns, lags):
            dfn[c] = ntsd[k].shift(periods=i)
        ntsd = pd.concat([ntsd, dfn], axis=1).reindex(ntsd.index)
    return tsutils.return_input(print_input, tsd, ntsd, "lag")
