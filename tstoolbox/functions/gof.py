#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

warnings.filterwarnings("ignore")


@mando.command("gof", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def gof_cli(
    input_ts="-",
    stats="all",
    columns=None,
    start_date=None,
    end_date=None,
    round_index=None,
    clean=False,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    skiprows=None,
    tablefmt="plain",
    float_format=".3f",
):
    """Will calculate goodness of fit statistics between two time-series.

    The first time series must be the observed, the second the predicted
    series.  You can only give two time-series.

    Parameters
    ----------
    {input_ts}
    stats : str
        [optional,  Python: list, Command line: comma separated string,
        default is 'all']

        The statistics that will be presented.

        +-----------------+--------------------------------------------------+
        | stats           | Description                                      |
        +=================+==================================================+
        | bias            | mean(s) - mean(o)                                |
        +-----------------+--------------------------------------------------+
        | pc_bias         | 100.0*sum(s-o)/sum(o)                            |
        +-----------------+--------------------------------------------------+
        | apc_bias        | 100.0*sum(abs(s-o))/sum(o)                       |
        +-----------------+--------------------------------------------------+
        | rmsd            | sum[(s - o)**2]/N                                |
        +-----------------+--------------------------------------------------+
        | crmsd           | sum[(s - mean(s))(o - mean(o))]**2/N             |
        +-----------------+--------------------------------------------------+
        | corrcoef        | Correlation coefficient                          |
        +-----------------+--------------------------------------------------+
        | murphyss        | 1 - RMSE**2/SDEV**2                              |
        +-----------------+--------------------------------------------------+
        | nse             | 1 - sum(s - o)**2 / sum (o - mean(r))**2         |
        +-----------------+--------------------------------------------------+
        | kge             | 1 - sqrt((cc-1)**2 + (alpha-1)**2 + (beta-1)**2) |
        +-----------------+--------------------------------------------------+
        |                 |                     cc = correlation coefficient |
        +-----------------+--------------------------------------------------+
        |                 |           alpha = std(simulated) / std(observed) |
        +-----------------+--------------------------------------------------+
        |                 |            beta = sum(simulated) / sum(observed) |
        +-----------------+--------------------------------------------------+
        | index_agreement | 1.0 - sum((o - s)**2) /                          |
        |                 | sum((abs(s - mean(o)) + abs(o - mean(o)))**2)    |
        +-----------------+--------------------------------------------------+
        | brierss         | sum(f - o)**2/N                                  |
        +-----------------+--------------------------------------------------+
        |                 |                       f = forecast probabilities |
        +-----------------+--------------------------------------------------+
        | mean            | observed mean, simulated mean                    |
        +-----------------+--------------------------------------------------+
        | stdev           | observed stdev, simulated stdev                  |
        +-----------------+--------------------------------------------------+

    {columns}
    {start_date}
    {end_date}
    {round_index}
    {clean}
    {index_type}
    {names}
    {source_units}
    {target_units}
    {skiprows}
    {tablefmt}
    {float_format}

    """
    tsutils._printiso(
        gof(
            input_ts=input_ts,
            stats=stats,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            round_index=round_index,
            clean=clean,
            index_type=index_type,
            names=names,
            source_units=source_units,
            target_units=target_units,
            skiprows=skiprows,
        ),
        tablefmt=tablefmt,
        headers=["Statistic", "Comparison", "Observed", "Simulated"],
        float_format=float_format,
    )


@tsutils.validator(
    stats=[
        str,
        [
            "domain",
            [
                "bias",
                "pc_bias",
                "apc_bias",
                "rmsd",
                "crmsd",
                "corrcoef",
                "murphyss",
                "nse",
                "kge",
                "index_agreement",
                "brierss",
                "mean",
                "stdev",
                "all",
            ],
        ],
        None,
    ]
)
def gof(
    input_ts="-",
    stats="all",
    columns=None,
    start_date=None,
    end_date=None,
    round_index=None,
    clean=False,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    skiprows=None,
):
    """Will calculate goodness of fit statistics between two time-series."""
    if stats == "all":
        stats = [
            "bias",
            "pc_bias",
            "apc_bias",
            "rmsd",
            "crmsd",
            "corrcoef",
            "murphyss",
            "nse",
            "kge",
            "index_agreement",
            "brierss",
            "mean",
            "stdev",
        ]
    else:
        try:
            stats = tsutils.make_list(stats)
        except AttributeError:
            pass

    # Use dropna='no' to get the lengths of both time-series.
    tsd = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts, skiprows=skiprows, names=names, index_type=index_type
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna="no",
        source_units=source_units,
        target_units=target_units,
        clean=clean,
    )
    if len(tsd.columns) != 2:
        raise ValueError(
            tsutils.error_wrapper(
                """
The gof algorithms work with two time-series only.  You gave {0}.
""".format(
                    len(tsd.columns)
                )
            )
        )
    lennao, lennas = tsd.isna().sum()

    tsd = tsd.dropna(how="any")

    from .. import skill_metrics as sm
    import numpy as np
    import pandas as pd

    statval = []

    ref = tsd.iloc[:, 0].values
    pred = tsd.iloc[:, 1].values

    if "bias" in stats:
        statval.append(["Bias", sm.bias(pred, ref)])

    if "pc_bias" in stats:
        statval.append(["Percent bias", sm.pc_bias(pred, ref)])

    if "apc_bias" in stats:
        statval.append(["Absolute percent bias", sm.apc_bias(pred, ref)])

    if "rmsd" in stats:
        statval.append(["Root-mean-square Deviation (RMSD)", sm.rmsd(pred, ref)])

    if "crmsd" in stats:
        statval.append(["Centered RMSD (CRMSD)", sm.centered_rms_dev(pred, ref)])

    if "corrcoef" in stats:
        statval.append(["Correlation coefficient (r)", np.corrcoef(pred, ref)[0, 1]])

    if "murphyss" in stats:
        statval.append(["Skill score (Murphy)", sm.skill_score_murphy(pred, ref)])

    if "nse" in stats:
        statval.append(["Nash-Sutcliffe Efficiency", sm.nse(pred, ref)])

    if "kge" in stats:
        statval.append(["Kling-Gupta Efficiency", sm.kge(pred, ref)])

    if "index_agreement" in stats:
        statval.append(["Index of agreement", sm.index_agreement(pred, ref)])

    statval.append(["Common count observed and simulated", len(tsd.index)])

    statval.append(["Count of NaNs", None, lennao, lennas])

    if "mean" in stats:
        statval.append(["Mean", None, ref.mean(), pred.mean()])

    if "stdev" in stats:
        statval.append(["Standard deviation", None, ref.std(), pred.std()])

    return statval


gof.__doc__ = gof_cli.__doc__
