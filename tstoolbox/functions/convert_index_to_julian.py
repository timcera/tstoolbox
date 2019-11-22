#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils
from . import convert_index

warnings.filterwarnings("ignore")


@mando.command(formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def convert_index_to_julian(
    epoch="julian",
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    round_index=None,
    dropna="no",
    clean=False,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    skiprows=None,
):
    """DEPRECATED: Use convert_index instead.

    Will be removed in a future version of `tstoolbox`.

    Use `convert_index` in place of `convert_index_to_julian`.

    For command line::

        tstoolbox convert_index julian ...

    For Python::

        from tstoolbox import tstoolbox
        ndf = ntstoolbox.convert_index('julian', ...)
    """
    warnings.warn(
        """
*
*   DEPRECATED in favor of using `convert_index` with the 'julian'
*   option.
*
*   Will be removed in a future version of `tstoolbox`.
*
"""
    )
    return convert_index(
        "julian",
        epoch="julian",
        columns=columns,
        input_ts=input_ts,
        start_date=start_date,
        end_date=end_date,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        skiprows=skiprows,
        names=names,
        source_units=source_units,
        target_units=target_units,
        index_type=index_type,
    )
