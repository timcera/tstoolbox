"""Collection of functions for the manipulation of time series."""

import warnings

import cltoolbox
from cltoolbox.rst_text_formatter import RSTHelpFormatter
from pydantic import validate_arguments
from toolbox_utils import tsutils

warnings.filterwarnings("ignore")


@cltoolbox.command("describe", formatter_class=RSTHelpFormatter)
@tsutils.doc(tsutils.docstrings)
def describe_cli(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    transpose=False,
    tablefmt="csv",
):
    """Print out statistics for the time-series.

    Parameters
    ----------
    transpose
        [optional, default is False, output format]

        If 'transpose' option is used, will transpose describe output.
    ${input_ts}
    ${columns}
    ${start_date}
    ${end_date}
    ${dropna}
    ${skiprows}
    ${index_type}
    ${names}
    ${clean}
    ${tablefmt}
    """
    tsutils.printiso(
        describe(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            clean=clean,
            transpose=transpose,
        ),
        showindex="always",
        tablefmt=tablefmt,
    )


@validate_arguments
@tsutils.copy_doc(describe_cli)
def describe(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    transpose: bool = False,
):
    """Print out statistics for the time-series."""
    tsd = tsutils.common_kwds(
        input_ts,
        skiprows=skiprows,
        names=names,
        index_type=index_type,
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        dropna=dropna,
        clean=clean,
    )
    ntsd = tsd.describe().transpose() if transpose else tsd.describe()
    ntsd.index.name = "Statistic"
    return ntsd
