# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
import pandas as pd
import typic
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

warnings.filterwarnings("ignore")


@mando.command("calculate_kde", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def calculate_kde_cli(
    ascending=True,
    evaluate=False,
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    clean=False,
    skiprows=None,
    index_type="datetime",
    source_units=None,
    target_units=None,
    names=None,
    tablefmt="csv",
):
    """Return the kernel density estimation (KDE) curve.

    Returns a time-series or the KDE curve depending on the `evaluate`
    keyword.

    Parameters
    ----------
    ascending : bool
        [optional, defaults to True, input filter]

        Sort order.
    evaluate : bool
        [optional, defaults to False, transformation]

        Whether or not to return a time-series of KDE density values or
        the KDE curve.  Defaults to False, which would return the KDE
        curve.
    ${input_ts}
    ${columns}
    ${start_date}
    ${end_date}
    ${skiprows}
    ${index_type}
    ${names}
    ${source_units}
    ${target_units}
    ${clean}
    ${tablefmt}
    """
    tsutils.printiso(
        calculate_kde(
            ascending=ascending,
            evaluate=evaluate,
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            source_units=source_units,
            target_units=target_units,
            names=names,
        ),
        tablefmt=tablefmt,
    )


@typic.al
@tsutils.copy_doc(calculate_kde_cli)
def calculate_kde(
    ascending: bool = True,
    evaluate: bool = False,
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    clean=False,
    skiprows=None,
    index_type="datetime",
    source_units=None,
    target_units=None,
    names=None,
):
    """Return the kernel density estimation (KDE) curve."""
    tsd = tsutils.common_kwds(
        input_ts,
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

    if len(tsd.columns) > 1:
        raise ValueError(
            tsutils.error_wrapper(
                """
Right now "calculate_kde" only support one time-series at a time.

You gave {}.
""".format(
                    tsd.columns
                )
            )
        )

    from scipy.stats import gaussian_kde

    tmptsd = tsd.dropna()
    ndf = tmptsd.sort_values(tmptsd.columns[0], ascending=ascending)
    gkde = gaussian_kde(ndf.iloc[:, 0])

    if evaluate is True:
        y = gkde.evaluate(tmptsd.iloc[:, 0])
        ndf = pd.DataFrame(y, index=tmptsd.index)
    else:
        y = gkde.evaluate(ndf.iloc[:, 0])
        ndf = pd.DataFrame(y)

    return ndf
