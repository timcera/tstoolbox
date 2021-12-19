# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings
from typing import List, Union

import HydroErr as he
import mando
import numpy as np
import typic
from mando.rst_text_formatter import RSTHelpFormatter

from .. import skill_metrics as sm
from .. import tsutils

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

warnings.filterwarnings("ignore")


@mando.command("gof", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def gof_cli(
    input_ts="-",
    stats="default",
    replace_nan=None,
    replace_inf=None,
    remove_neg=False,
    remove_zero=False,
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
    kge_sr=1.0,
    kge09_salpha=1.0,
    kge12_sgamma=1.0,
    kge_sbeta=1.0,
):
    """Will calculate goodness of fit statistics between two time-series.

    The first time series must be the observed, the second the predicted
    series.  You can only give two time-series.

    Parameters
    ----------
    {input_ts}

    stats : str
        [optional,  Python: list, Command line: comma separated string,
        default is 'default']

        Comma separated list of statistical measures.

        You can select two groups of statistical measures.

        +------------+---------------------------------------+
        | stats      | Description                           |
        +============+=======================================+
        | default    | A subset of common statistic measures |
        +------------+---------------------------------------+
        | all        | All available statistic measures      |
        +------------+---------------------------------------+

        The 'default' set of statistics are:

        +-----------------+--------------------------------------------------+
        | stats           | Description                                      |
        +=================+==================================================+
        | bias            | Bias                                             |
        |                 | mean(s) - mean(o)                                |
        +-----------------+--------------------------------------------------+
        | pc_bias         | Percent Bias                                     |
        |                 | 100.0*sum(s-o)/sum(o)                            |
        +-----------------+--------------------------------------------------+
        | apc_bias        | Absolute Percent Bias                            |
        |                 | 100.0*sum(abs(s-o))/sum(o)                       |
        +-----------------+--------------------------------------------------+
        | rmsd            | Root Mean Square Deviation/Error                 |
        |                 | sqrt(sum[(s - o)**2]/N)                          |
        +-----------------+--------------------------------------------------+
        | crmsd           | Centered Root Mean Square Deviation/Error        |
        |                 | sum[(s - mean(s))(o - mean(o))]**2/N             |
        +-----------------+--------------------------------------------------+
        | corrcoef        | Pearson Correlation coefficient (r)              |
        +-----------------+--------------------------------------------------+
        | coefdet         | Coefficient of determination (R^2)               |
        +-----------------+--------------------------------------------------+
        | murphyss        | Murphy Skill Score                               |
        |                 | 1 - RMSE**2/SDEV**2                              |
        +-----------------+--------------------------------------------------+
        | nse             | Nash-Sutcliffe Efficiency                        |
        |                 | 1 - sum(s - o)**2 / sum (o - mean(r))**2         |
        +-----------------+--------------------------------------------------+
        | kge09           | Kling-Gupta Efficiency, 2009                     |
        |                 | 1 - sqrt((cc-1)**2 + (alpha-1)**2 + (beta-1)**2) |
        |                 |                     cc = correlation coefficient |
        |                 |           alpha = std(simulated) / std(observed) |
        |                 |            beta = sum(simulated) / sum(observed) |
        +-----------------+--------------------------------------------------+
        | kge12           | Kling-Gupta Efficiency, 2012                     |
        |                 | 1 - sqrt((cc-1)**2 + (alpha-1)**2 + (beta-1)**2) |
        |                 |                     cc = correlation coefficient |
        |                 |           alpha = std(simulated) / std(observed) |
        |                 |            beta = sum(simulated) / sum(observed) |
        +-----------------+--------------------------------------------------+
        | index_agreement | Index of Agreement                               |
        |                 | 1.0 - sum((o - s)**2) /                          |
        |                 | sum((abs(s - mean(o)) + abs(o - mean(o)))**2)    |
        +-----------------+--------------------------------------------------+
        | brierss         | Brier Skill Score                                |
        |                 | sum(f - o)**2/N                                  |
        |                 |                       f = forecast probabilities |
        +-----------------+--------------------------------------------------+
        | mae             | Mean Absolute Error                              |
        |                 | sum(abs(s - o))/N                                |
        +-----------------+--------------------------------------------------+
        | mean            | observed mean, simulated mean                    |
        +-----------------+--------------------------------------------------+
        | stdev           | observed stdev, simulated stdev                  |
        +-----------------+--------------------------------------------------+

        Additional statistics:

        +-------------+-------------------------------------------------------+
        | stats       | Description                                           |
        +=============+=======================================================+
        | acc         | Anomaly correlation coefficient (ACC)                 |
        +-------------+-------------------------------------------------------+
        | d1          | Index of agreement (d1)                               |
        +-------------+-------------------------------------------------------+
        | d1_p        | Legate-McCabe Index of Agreement                      |
        +-------------+-------------------------------------------------------+
        | d           | Index of agreement (d)                                |
        +-------------+-------------------------------------------------------+
        | dmod        | Modified index of agreement (dmod)                    |
        +-------------+-------------------------------------------------------+
        | drel        | Relative index of agreement (drel)                    |
        +-------------+-------------------------------------------------------+
        | dr          | Refined index of agreement (dr)                       |
        +-------------+-------------------------------------------------------+
        | ed          | Euclidean distance in vector space                    |
        +-------------+-------------------------------------------------------+
        | g_mean_diff | Geometric mean difference                             |
        +-------------+-------------------------------------------------------+
        | h10_mahe    | H10 mean absolute error                               |
        +-------------+-------------------------------------------------------+
        | h10_mhe     | H10 mean error                                        |
        +-------------+-------------------------------------------------------+
        | h10_rmshe   | H10 root mean square error                            |
        +-------------+-------------------------------------------------------+
        | h1_mahe     | H1 absolute error                                     |
        +-------------+-------------------------------------------------------+
        | h1_mhe      | H1 mean error                                         |
        +-------------+-------------------------------------------------------+
        | h1_rmshe    | H1 root mean square error                             |
        +-------------+-------------------------------------------------------+
        | h2_mahe     | H2 mean absolute error                                |
        +-------------+-------------------------------------------------------+
        | h2_mhe      | H2 mean error                                         |
        +-------------+-------------------------------------------------------+
        | h2_rmshe    | H2 root mean square error                             |
        +-------------+-------------------------------------------------------+
        | h3_mahe     | H3 mean absolute error                                |
        +-------------+-------------------------------------------------------+
        | h3_mhe      | H3 mean error                                         |
        +-------------+-------------------------------------------------------+
        | h3_rmshe    | H3 root mean square error                             |
        +-------------+-------------------------------------------------------+
        | h4_mahe     | H4 mean absolute error                                |
        +-------------+-------------------------------------------------------+
        | h4_mhe      | H4 mean error                                         |
        +-------------+-------------------------------------------------------+
        | h4_rmshe    | H4 root mean square error                             |
        +-------------+-------------------------------------------------------+
        | h5_mahe     | H5 mean absolute error                                |
        +-------------+-------------------------------------------------------+
        | h5_mhe      | H5 mean error                                         |
        +-------------+-------------------------------------------------------+
        | h5_rmshe    | H5 root mean square error                             |
        +-------------+-------------------------------------------------------+
        | h6_mahe     | H6 mean absolute error                                |
        +-------------+-------------------------------------------------------+
        | h6_mhe      | H6 mean error                                         |
        +-------------+-------------------------------------------------------+
        | h6_rmshe    | H6 root mean square error                             |
        +-------------+-------------------------------------------------------+
        | h7_mahe     | H7 mean absolute error                                |
        +-------------+-------------------------------------------------------+
        | h7_mhe      | H7 mean error                                         |
        +-------------+-------------------------------------------------------+
        | h7_rmshe    | H7 root mean square error                             |
        +-------------+-------------------------------------------------------+
        | h8_mahe     | H8 mean absolute error                                |
        +-------------+-------------------------------------------------------+
        | h8_mhe      | H8 mean error                                         |
        +-------------+-------------------------------------------------------+
        | h8_rmshe    | H8 root mean square error                             |
        +-------------+-------------------------------------------------------+
        | irmse       | Inertial root mean square error (IRMSE)               |
        +-------------+-------------------------------------------------------+
        | lm_index    | Legate-McCabe Efficiency Index                        |
        +-------------+-------------------------------------------------------+
        | maape       | Mean Arctangent Absolute Percentage Error (MAAPE)     |
        +-------------+-------------------------------------------------------+
        | male        | Mean absolute log error                               |
        +-------------+-------------------------------------------------------+
        | mapd        | Mean absolute percentage deviation (MAPD)             |
        +-------------+-------------------------------------------------------+
        | mape        | Mean absolute percentage error (MAPE)                 |
        +-------------+-------------------------------------------------------+
        | mase        | Mean absolute scaled error                            |
        +-------------+-------------------------------------------------------+
        | mb_r        | Mielke-Berry R value (MB R)                           |
        +-------------+-------------------------------------------------------+
        | mdae        | Median absolute error (MdAE)                          |
        +-------------+-------------------------------------------------------+
        | mde         | Median error (MdE)                                    |
        +-------------+-------------------------------------------------------+
        | mdse        | Median squared error (MdSE)                           |
        +-------------+-------------------------------------------------------+
        | mean_var    | Mean variance                                         |
        +-------------+-------------------------------------------------------+
        | me          | Mean error                                            |
        +-------------+-------------------------------------------------------+
        | mle         | Mean log error                                        |
        +-------------+-------------------------------------------------------+
        | mse         | Mean squared error                                    |
        +-------------+-------------------------------------------------------+
        | msle        | Mean squared log error                                |
        +-------------+-------------------------------------------------------+
        | ned         | Normalized Euclidian distance in vector space         |
        +-------------+-------------------------------------------------------+
        | nrmse_iqr   | IQR normalized root mean square error                 |
        +-------------+-------------------------------------------------------+
        | nrmse_mean  | Mean normalized root mean square error                |
        +-------------+-------------------------------------------------------+
        | nrmse_range | Range normalized root mean square error               |
        +-------------+-------------------------------------------------------+
        | nse_mod     | Modified Nash-Sutcliffe efficiency (NSE mod)          |
        +-------------+-------------------------------------------------------+
        | nse_rel     | Relative Nash-Sutcliffe efficiency (NSE rel)          |
        +-------------+-------------------------------------------------------+
        | rmse        | Root mean square error                                |
        +-------------+-------------------------------------------------------+
        | rmsle       | Root mean square log error                            |
        +-------------+-------------------------------------------------------+
        | sa          | Spectral Angle (SA)                                   |
        +-------------+-------------------------------------------------------+
        | sc          | Spectral Correlation (SC)                             |
        +-------------+-------------------------------------------------------+
        | sga         | Spectral Gradient Angle (SGA)                         |
        +-------------+-------------------------------------------------------+
        | sid         | Spectral Information Divergence (SID)                 |
        +-------------+-------------------------------------------------------+
        | smape1      | Symmetric Mean Absolute Percentage Error (1) (SMAPE1) |
        +-------------+-------------------------------------------------------+
        | smape2      | Symmetric Mean Absolute Percentage Error (2) (SMAPE2) |
        +-------------+-------------------------------------------------------+
        | spearman_r  | Spearman rank correlation coefficient                 |
        +-------------+-------------------------------------------------------+
        | ve          | Volumetric Efficiency (VE)                            |
        +-------------+-------------------------------------------------------+
        | watt_m      | Watterson’s M (M)                                     |
        +-------------+-------------------------------------------------------+

    replace_nan: (float, optional)
        If given, indicates which value to replace NaN values with in the two
        arrays. If None, when a NaN value is found at the i-th position in the
        observed OR simulated array, the i-th value of the observed and
        simulated array are removed before the computation.

    replace_inf: (float, optional)
        If given, indicates which value to replace Inf values with in the two
        arrays. If None, when an inf value is found at the i-th position in the
        observed OR simulated array, the i-th value of the observed and
        simulated array are removed before the computation.

    remove_neg: (boolean, optional)
        If True, when a negative value is found at the i-th position in the
        observed OR simulated array, the i-th value of the observed AND
        simulated array are removed before the computation.

    remove_zero: (boolean, optional)
        If true, when a zero value is found at the i-th position in the
        observed OR simulated array, the i-th value of the observed AND
        simulated array are removed before the computation.

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

    kge_sr: float
        [optional, defaults to 1.0]

        Scaling factor for `kge09` and `kge12` correlation.

    kge09_salpha: float
        [optional, defaults to 1.0]

        Scaling factor for `kge09` alpha.

    kge12_sgamma: float
        [optional, defaults to 1.0]

        Scaling factor for `kge12` beta.

    kge_sbeta: float
        [optional, defaults to 1.0]

        Scaling factor for `kge09` and `kge12` beta."""
    tsutils.printiso(
        gof(
            input_ts=input_ts,
            stats=stats,
            replace_nan=replace_nan,
            replace_inf=replace_inf,
            remove_neg=remove_neg,
            remove_zero=remove_zero,
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
            kge_sr=kge_sr,
            kge09_salpha=kge09_salpha,
            kge12_sgamma=kge12_sgamma,
            kge_sbeta=kge_sbeta,
        ),
        tablefmt=tablefmt,
        headers=["Statistic", "Comparison", "Observed", "Simulated"],
        float_format=float_format,
    )


stats_dict = {
    "bias": ["Bias", sm.bias],
    "pc_bias": ["Percent bias", sm.pc_bias],
    "apc_bias": ["Absolute percent bias", sm.apc_bias],
    "rmsd": ["Root-mean-square Deviation/Error (RMSD)", he.rmse],
    "crmsd": ["Centered RMSD (CRMSD)", sm.centered_rms_dev],
    "corrcoef": ["Pearson coefficient of correlation (r)", he.pearson_r],
    "coefdet": ["Coefficient of determination (r^2)", he.r_squared],
    "murphyss": ["Skill score (Murphy)", sm.skill_score_murphy],
    "nse": ["Nash-Sutcliffe Efficiency", he.nse],
    "kge09": ["Kling-Gupta efficiency (2009)", he.kge_2009],
    "kge12": ["Kling-Gupta efficiency (2012)", he.kge_2012],
    "index_agreement": ["Index of agreement", he.d],
    "brierss": ["Brier's Score", lambda pred, ref: np.sum(pred - ref) ** 2 / len(pred)],
    "mae": ["Mean Absolute Error", he.mae],
    "mean": ["Mean", lambda x: x],
    "stdev": ["Standard deviation", lambda x: x],
    "acc": ["Anomaly correlation coefficient (ACC)", he.acc],
    "d1": ["Index of agreement (d1)", he.d1],
    "d1_p": ["Legate-McCabe Index of Agreement", he.d1_p],
    "d": ["Index of agreement (d)", he.d],
    "dmod": ["Modified index of agreement (dmod)", he.dmod],
    "drel": ["Relative index of agreement (drel)", he.drel],
    "dr": ["Refined index of agreement (dr)", he.dr],
    "ed": ["Euclidean distance in vector space", he.ed],
    "g_mean_diff": ["Geometric mean difference", he.g_mean_diff],
    "h10_mahe": ["H10 mean absolute error", he.h10_mahe],
    "h10_mhe": ["H10 mean error", he.h10_mhe],
    "h10_rmshe": ["H10 root mean square error", he.h10_rmshe],
    "h1_mahe": ["H1 absolute error", he.h1_mahe],
    "h1_mhe": ["H1 mean error", he.h1_mhe],
    "h1_rmshe": ["H1 root mean square error", he.h1_rmshe],
    "h2_mahe": ["H2 mean absolute error", he.h2_mahe],
    "h2_mhe": ["H2 mean error", he.h2_mhe],
    "h2_rmshe": ["H2 root mean square error", he.h2_rmshe],
    "h3_mahe": ["H3 mean absolute error", he.h3_mahe],
    "h3_mhe": ["H3 mean error", he.h3_mhe],
    "h3_rmshe": ["H3 root mean square error", he.h3_rmshe],
    "h4_mahe": ["H4 mean absolute error", he.h4_mahe],
    "h4_mhe": ["H4 mean error", he.h4_mhe],
    "h4_rmshe": ["H4 mean error", he.h4_rmshe],
    "h5_mahe": ["H5 mean absolute error", he.h5_mahe],
    "h5_mhe": ["H5 mean error", he.h5_mhe],
    "h5_rmshe": ["H5 root mean square error", he.h5_rmshe],
    "h6_mahe": ["H6 mean absolute error", he.h6_mahe],
    "h6_mhe": ["H6 mean error", he.h6_mhe],
    "h6_rmshe": ["H6 root mean square error", he.h6_rmshe],
    "h7_mahe": ["H7 mean absolute error", he.h7_mahe],
    "h7_mhe": ["H7 mean error", he.h7_mhe],
    "h7_rmshe": ["H7 root mean square error", he.h7_rmshe],
    "h8_mahe": ["H8 mean absolute error", he.h8_mahe],
    "h8_mhe": ["H8 mean error", he.h8_mhe],
    "h8_rmshe": ["H8 root mean square error", he.h8_rmshe],
    "irmse": ["Inertial root mean square error (IRMSE)", he.irmse],
    "lm_index": ["Legate-McCabe Efficiency Index", he.lm_index],
    "maape": ["Mean Arctangent Absolute Percentage Error (MAAPE)", he.maape],
    "male": ["Mean absolute log error", he.male],
    "mapd": ["Mean absolute percentage deviation (MAPD)", he.mapd],
    "mape": ["Mean absolute percentage error (MAPE)", he.mape],
    "mase": ["Mean absolute scaled error", he.mase],
    "mb_r": ["Mielke-Berry R value (MB R)", he.mb_r],
    "mdae": ["Median absolute error (MdAE)", he.mdae],
    "mde": ["Median error (MdE)", he.mde],
    "mdse": ["Median squared error (MdSE)", he.mdse],
    "mean_var": ["Mean variance", he.mean_var],
    "me": ["Mean error", he.me],
    "mle": ["Mean log error", he.mle],
    "mse": ["Mean squared error", he.mse],
    "msle": ["Mean squared log error", he.msle],
    "ned": ["Normalized Euclidian distance in vector space", he.ned],
    "nrmse_iqr": ["IQR normalized root mean square error", he.nrmse_iqr],
    "nrmse_mean": ["Mean normalized root mean square error", he.nrmse_mean],
    "nrmse_range": ["Range normalized root mean square error", he.nrmse_range],
    "nse_mod": ["Modified Nash-Sutcliffe efficiency (NSE mod)", he.nse_mod],
    "nse_rel": ["Relative Nash-Sutcliffe efficiency (NSE rel)", he.nse_rel],
    "rmse": ["Root mean square error", he.rmse],
    "rmsle": ["Root mean square log error", he.rmsle],
    "sa": ["Spectral Angle (SA)", he.sa],
    "sc": ["Spectral Correlation (SC)", he.sc],
    "sga": ["Spectral Gradient Angle (SGA)", he.sga],
    "sid": ["Spectral Information Divergence (SID)", he.sid],
    "smape1": ["Symmetric Mean Absolute Percentage Error (1) (SMAPE1)", he.smape1],
    "smape2": ["Symmetric Mean Absolute Percentage Error (2) (SMAPE2)", he.smape2],
    "spearman_r": ["Spearman rank correlation coefficient", he.spearman_r],
    "ve": ["Volumetric Efficiency (VE)", he.ve],
    "watt_m": ["Watterson’s M (M)", he.watt_m],
}


@tsutils.transform_args(stats=tsutils.make_list)
@typic.al
def gof(
    input_ts="-",
    stats: List[
        Literal[
            "default",
            "all",
            "bias",
            "pc_bias",
            "apc_bias",
            "rmsd",
            "crmsd",
            "corrcoef",
            "coefdet",
            "murphyss",
            "nse",
            "kge09",
            "kge12",
            "index_agreement",
            "brierss",
            "mae",
            "mean",
            "stdev",
            "acc",
            "d1",
            "d1_p",
            "d",
            "dmod",
            "drel",
            "dr",
            "ed",
            "g_mean_diff",
            "h10_mahe",
            "h10_mhe",
            "h10_rmshe",
            "h1_mahe",
            "h1_mhe",
            "h1_rmshe",
            "h2_mahe",
            "h2_mhe",
            "h2_rmshe",
            "h3_mahe",
            "h3_mhe",
            "h3_rmshe",
            "h4_mahe",
            "h4_mhe",
            "h4_rmshe",
            "h5_mahe",
            "h5_mhe",
            "h5_rmshe",
            "h6_mahe",
            "h6_mhe",
            "h6_rmshe",
            "h7_mahe",
            "h7_mhe",
            "h7_rmshe",
            "h8_mahe",
            "h8_mhe",
            "h8_rmshe",
            "irmse",
            "lm_index",
            "maape",
            "male",
            "mapd",
            "mape",
            "mase",
            "mb_r",
            "mdae",
            "mde",
            "mdse",
            "mean_var",
            "me",
            "mle",
            "mse",
            "msle",
            "ned",
            "nrmse_iqr",
            "nrmse_mean",
            "nrmse_range",
            "nse_mod",
            "nse_rel",
            "rmse",
            "rmsle",
            "sa",
            "sc",
            "sga",
            "sid",
            "smape1",
            "smape2",
            "spearman_r",
            "ve",
            "watt_m",
        ],
    ] = "default",
    replace_nan=None,
    replace_inf=None,
    remove_neg=False,
    remove_zero=False,
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
    kge_sr: float = 1.0,
    kge09_salpha: float = 1.0,
    kge12_sgamma: float = 1.0,
    kge_sbeta: float = 1.0,
):
    """Will calculate goodness of fit statistics between two time-series."""
    if "default" in stats:
        stats = [
            "bias",
            "pc_bias",
            "apc_bias",
            "rmsd",
            "crmsd",
            "corrcoef",
            "coefdet",
            "murphyss",
            "nse",
            "kge09",
            "kge12",
            "index_agreement",
            "brierss",
            "mae",
            "mean",
            "stdev",
        ]
    elif "all" in stats:
        stats = stats_dict.keys()

    # Use dropna='no' to get the lengths of both time-series.
    tsd = tsutils.common_kwds(
        input_ts,
        skiprows=skiprows,
        names=names,
        index_type=index_type,
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
The gof algorithms work with two time-series only.  You gave {}.
""".format(
                    len(tsd.columns)
                )
            )
        )
    lennao, lennas = tsd.isna().sum()

    tsd = tsd.dropna(how="any")

    statval = []

    ref = tsd.iloc[:, 0].astype("float64")
    pred = tsd.iloc[:, 1].astype("float64")

    nstats = [i for i in stats if i not in ["mean", "stdev"]]

    for stat in nstats:
        extra_args = {
            "replace_nan": replace_nan,
            "replace_inf": replace_inf,
            "remove_neg": remove_neg,
            "remove_zero": remove_zero,
        }
        if stat in ["bias", "crmsd", "murphyss", "brierss", "pc_bias", "apc_bias"]:
            extra_args = {}
        proc = stats_dict[stat]
        if "kge09" == stat:
            extra_args = {"s": (kge_sr, kge09_salpha, kge_sbeta)}
        if "kge12" == stat:
            extra_args = {"s": (kge_sr, kge12_sgamma, kge_sbeta)}
        statval.append(
            [
                proc[0],
                proc[1](pred, ref, **extra_args),
            ]
        )

    statval.append(["Common count observed and simulated", len(tsd.index)])

    statval.append(["Count of NaNs", None, lennao, lennas])

    if "mean" in stats:
        statval.append(["Mean", None, ref.mean(), pred.mean()])

    if "stdev" in stats:
        statval.append(["Standard deviation", None, ref.std(), pred.std()])

    return statval


gof.__doc__ = gof_cli.__doc__
