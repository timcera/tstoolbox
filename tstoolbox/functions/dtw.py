#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from builtins import range
from builtins import str

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import pandas as pd

from .. import tsutils


def _dtw(ts_a, ts_b, d=lambda x, y: abs(x - y), window=10000):
    """Return the DTW similarity distance timeseries numpy arrays.

    Arguments
    ---------
    ts_a, ts_b : array of shape [n_samples, n_timepoints]
        Two arrays containing n_samples of timeseries data
        whose DTW distance between each sample of A and B
        will be compared

    d : DistanceMetric object (default = abs(x-y))
        the distance measure used for A_i - B_j in the
        DTW dynamic programming function

    Returns
    -------
    DTW distance between A and B

    """
    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = pd.np.array(ts_a), pd.np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = sys.maxsize * pd.np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - window),
                       min(N, i + window)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window
    return cost[-1, -1]


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def dtw(input_ts='-',
        columns=None,
        start_date=None,
        end_date=None,
        round_index=None,
        dropna='no',
        skiprows=None,
        index_type='datetime',
        names=None,
        clean=False,
        source_units=None,
        target_units=None,
        window=10000):
    """Dynamic Time Warping.

    Parameters
    ----------
    window : int
         [optional, default is 10000]

         Window length.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {round_index}
    {dropna}
    {skiprows}
    {index_type}
    {source_units}
    {target_units}
    {names}
    {clean}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts,
                                                  skiprows=skiprows,
                                                  names=names,
                                                  index_type=index_type),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna='no',
                              source_units=source_units,
                              target_units=target_units,
                              clean=clean)

    process = {}
    for i in tsd.columns:
        for j in tsd.columns:
            if (i, j) not in process and (j, i) not in process and i != j:
                process[(i, j)] = _dtw(tsd[i], tsd[j], window=window)

    ntsd = pd.DataFrame(list(process.items()))
    ncols = ntsd.columns
    ncols = ['Variables'] + [str(i) + 'DTW_score' for i in ncols[1:]]
    ntsd.columns = ncols
    return tsutils.printiso(ntsd)
