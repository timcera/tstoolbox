#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def pca(input_ts='-',
        columns=None,
        start_date=None,
        end_date=None,
        clean=False,
        skiprows=None,
        index_type='datetime',
        names=None,
        n_components=None,
        source_units=None,
        target_units=None,
        round_index=None):
    """Return the principal components analysis of the time series.

    Does not return a time-series.

    Parameters
    ----------
    n_components : int
        [optional, default is None]

        The number of groups to separate the time series into.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {clean}
    {skiprows}
    {index_type}
    {names}
    {source_units}
    {target_units}
    {round_index}

    """
    from sklearn.decomposition import PCA

    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts,
                                                  skiprows=skiprows,
                                                  names=names,
                                                  index_type=index_type),
                              start_date=start_date,
                              end_date=end_date,
                              round_index=round_index,
                              pick=columns,
                              source_units=source_units,
                              target_units=target_units,
                              clean=clean)

    pca = PCA(n_components)
    pca.fit(tsd.dropna(how='any'))
    print(pca.components_)
