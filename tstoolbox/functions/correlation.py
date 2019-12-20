#!/sjr/beodata/local/python_linux/bin/python
"""A correlation routine."""

from __future__ import absolute_import, print_function

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils
from . import lag


@mando.command("correlation", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def correlation_cli(
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
    """Develop a correlation between time-series and potentially lags.

    Parameters
    ----------
    lags : str, int, or list
        If an integer will calculate all lags up to and including the
        lag number.  If a list will calculate each lag in the list.  If
        a string must be a comma separated list of integers.  If lags ==
        0 then will only cross correlate on the input time-series.

        Python example::

            lags=[2, 5, 3]

        Command line example::

            --lags='2,5,3'
    {print_input}
    {input_ts}
    {start_date}
    {end_date}
    {clean}
    {skiprows}
    {index_type}
    {names}
    {source_units}
    {target_units}
    {columns}
    {tablefmt}

    """
    tsutils._printiso(
        correlation(
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
        showindex="always",
        tablefmt=tablefmt,
    )


@tsutils.validator(lags=[int, ["pass", []], None])
def correlation(
    lags,
    input_ts="-",
    method="ffill",
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
    """Develop a correlation between time-series and potentially lags."""
    ntsd = lag.lag(
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
    )
    return ntsd.corr()


correlation.__doc__ = correlation_cli.__doc__
