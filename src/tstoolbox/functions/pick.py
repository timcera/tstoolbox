# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

warnings.filterwarnings("ignore")


@mando.command("pick", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def pick_cli(
    columns,
    input_ts="-",
    start_date=None,
    end_date=None,
    round_index=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    clean=False,
    tablefmt="csv",
):
    """Will pick a column or list of columns from input [DEPRECATED].

    DEPRECATED: Effectively replaced by the "columns" keyword available
    in all other functions.

    Will be removed in a future version of `tstoolbox`.

    Can use column names or column numbers.  If using numbers, column
    number 1 is the first data column.

    Parameters
    ----------
    ${columns}
    ${input_ts}
    ${start_date}
    ${end_date}
    ${dropna}
    ${skiprows}
    ${index_type}
    ${names}
    ${clean}
    ${source_units}
    ${target_units}
    ${round_index}
    ${tablefmt}
    """
    tsutils.printiso(
        pick(
            columns,
            input_ts=input_ts,
            start_date=start_date,
            end_date=end_date,
            round_index=round_index,
            dropna=dropna,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            source_units=source_units,
            target_units=target_units,
            clean=clean,
        ),
        tablefmt=tablefmt,
    )


@tsutils.copy_doc(pick_cli)
def pick(
    columns,
    input_ts="-",
    start_date=None,
    end_date=None,
    round_index=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    clean=False,
):
    """Will pick a column or list of columns from input."""
    warnings.warn(
        """
*
*   DEPRECATED in favor of using the "columns" keyword available in all
*   other functions.
*
*   Will be removed in a future version of `tstoolbox`.
*
"""
    )
    return tsutils.common_kwds(
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
