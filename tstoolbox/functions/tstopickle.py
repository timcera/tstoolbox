#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

warnings.filterwarnings("ignore")


@mando.command("tstopickle", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def tstopickle_cli(
    filename,
    input_ts="-",
    columns=None,
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
    """Pickle the data into a Python pickled file.

    Can be brought back into Python with 'pickle.load' or 'numpy.load'.
    See also 'tstoolbox read'.

    Parameters
    ----------
    filename : str
         The filename to store the pickled data.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {skiprows}
    {index_type}
    {names}
    {clean}
    {source_units}
    {target_units}
    {round_index}

    """
    tsutils.printiso(
        tstopickle(
            filename,
            input_ts=input_ts,
            columns=columns,
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
        )
    )


def tstopickle(
    filename,
    input_ts="-",
    columns=None,
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
    """Pickle the data into a Python pickled file."""
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
    tsd.to_pickle(filename)


tstopickle.__doc__ = tstopickle_cli.__doc__
