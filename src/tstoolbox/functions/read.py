# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import os
import warnings
from argparse import RawTextHelpFormatter

import mando
import pandas as pd
import typic

from .. import tsutils

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


warnings.filterwarnings("ignore")


@mando.command("read", formatter_class=RawTextHelpFormatter)
@tsutils.doc(tsutils.docstrings)
def read_cli(
    force_freq=None,
    append="columns",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    source_units=None,
    target_units=None,
    float_format="g",
    round_index=None,
    tablefmt="csv",
    *filenames,
):
    """Combine time-series from different sources into single dataset.

    Prints the read in time-series in the tstoolbox standard format.

    WARNING: Accepts naive and timezone aware time-series by converting all to
    UTC and removing timezone information.

    Parameters
    ----------
    filenames : str
        From the command line a list of comma or space delimited filenames to
        read time series from.  Using the Python API a list or tuple of
        filenames.
    append : str
        [optional, default is 'columns']

        The type of appending to do.  For "combine" option matching column
        indices will append rows, matching row indices will append columns, and
        matching column/row indices use the value from the first dataset.  You
        can use "row" or "column" to force an append along either axis.
    ${force_freq}
        ${pandas_offset_codes}
    ${columns}
    ${start_date}
    ${end_date}
    ${dropna}
    ${skiprows}
    ${index_type}
    ${names}
    ${clean}
    ${source_units}
    ${target_units}
    ${float_format}
    ${round_index}
    ${tablefmt}
    """
    tsutils.printiso(
        read(
            *filenames,
            append=append,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            index_type=index_type,
            clean=clean,
            force_freq=force_freq,
            round_index=round_index,
            columns=columns,
            skiprows=skiprows,
            names=names,
            source_units=source_units,
            target_units=target_units,
        ),
        float_format=float_format,
        tablefmt=tablefmt,
    )


@typic.al
@tsutils.copy_doc(read_cli)
def read(
    *filenames,
    force_freq=None,
    append: Literal["columns", "rows", "combine"] = "columns",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    source_units=None,
    target_units=None,
    round_index=None,
):
    """Collect time series from a list of pickle or csv files."""
    if force_freq is not None:
        dropna = "no"

    if isinstance(filenames, (list, tuple)) and len(filenames) == 1:
        filenames = filenames[0]

    # Check for older style where comma delimited list of only files.
    # If so, rework as space delimited.
    isspacedelimited = False
    for fname in tsutils.make_list(filenames, sep=","):
        if not os.path.exists(str(fname)):
            isspacedelimited = True
            break

    if isspacedelimited is True:
        filenames = tsutils.make_list(filenames, sep=" ", flat=False)
    else:
        # All filenames are real files.  Therefore old style and just make
        # a simple list.
        filenames = tsutils.make_list(filenames, sep=",")
        warnings.warn(
            tsutils.error_wrapper(
                """
Using "," separated files is deprecated in favor of space delimited files."""
            )
        )

    tsd = tsutils.common_kwds(
        input_tsd=filenames,
        skiprows=skiprows,
        index_type=index_type,
        start_date=start_date,
        end_date=end_date,
        round_index=round_index,
        names=names,
        dropna=dropna,
        force_freq=force_freq,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
        usecols=columns,
    )

    return tsd
