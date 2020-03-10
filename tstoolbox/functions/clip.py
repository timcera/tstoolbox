#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import numpy as np
import pandas as pd

from .. import tsutils

warnings.filterwarnings("ignore")


@mando.command("clip", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def clip_cli(
    input_ts="-",
    start_date=None,
    end_date=None,
    columns=None,
    dropna="no",
    clean=False,
    skiprows=None,
    index_type="datetime",
    names=None,
    a_min=None,
    a_max=None,
    round_index=None,
    source_units=None,
    target_units=None,
    print_input=False,
    tablefmt="csv",
):
    """Return a time-series with values limited to [a_min, a_max].

    Parameters
    ---------
    a_min
        [optional, defaults to None, transformation]

        All values lower than this will be set to this value.
        Default is None.
    a_max
        [optional, defaults to None, transformation]

        All values higher than this will be set to this value.
        Default is None.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {clean}
    {skiprows}
    {index_type}
    {print_input}
    {names}
    {source_units}
    {target_units}
    {round_index}
    {tablefmt}

    """
    tsutils.printiso(
        clip(
            input_ts=input_ts,
            start_date=start_date,
            end_date=end_date,
            columns=columns,
            dropna=dropna,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            a_min=a_min,
            a_max=a_max,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


@tsutils.validator(a_min=[float, ["pass", []], 1], a_max=[float, ["pass", []], 1])
def clip(
    input_ts="-",
    start_date=None,
    end_date=None,
    columns=None,
    dropna="no",
    clean=False,
    skiprows=None,
    index_type="datetime",
    names=None,
    a_min=None,
    a_max=None,
    round_index=None,
    source_units=None,
    target_units=None,
    print_input=False,
):
    """Return a time-series with values limited to [a_min, a_max]."""
    tsd = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts, skiprows=skiprows, names=names, index_type=index_type
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        source_units=source_units,
        target_units=target_units,
        clean=clean,
    )

    ntsd = pd.DataFrame()
    for col in tsd.columns:
        if a_min is None:
            try:
                n_min = np.finfo(tsd[col].dtype).min
            except ValueError:
                n_min = np.iinfo(tsd[col].dtype).min
        else:
            n_min = float(a_min)

        if a_max is None:
            try:
                n_max = np.finfo(tsd[col].dtype).max
            except ValueError:
                n_max = np.iinfo(tsd[col].dtype).max
        else:
            n_max = float(a_max)

        ntsd = ntsd.join(tsd[col].clip(n_min, n_max), how="outer")

    return tsutils.return_input(print_input, tsd, ntsd, "clip")


clip.__doc__ = clip_cli.__doc__
