# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings
from typing import List, Union

import HydroErr as he
import mando
import numpy as np
import pandas as pd
import typic
from mando.rst_text_formatter import RSTHelpFormatter

from .. import skill_metrics as sm
from .. import tsutils
from .read import read

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

warnings.filterwarnings("ignore")


@mando.command("gof", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def gof_cli(
    obs_col=None,
    sim_col=None,
    stats="default",
    replace_nan=None,
    replace_inf=None,
    remove_neg=False,
    remove_zero=False,
    start_date=None,
    end_date=None,
    round_index=None,
    clean=False,
    index_type="datetime",
    source_units=None,
    target_units=None,
    tablefmt="plain",
    float_format=".3f",
    kge_sr=1.0,
    kge09_salpha=1.0,
    kge12_sgamma=1.0,
    kge_sbeta=1.0,
):
    """Will calculate goodness of fit statistics between two time-series.

    The first time series must be the observed, the second the simulated
    series.  You can only give two time-series.

    Parameters
    ----------
    obs_col
        If integer represents the column number of standard input. Can be
        If integer represents the column number of standard input. Can be
        a csv, wdm, hdf or xlsx file following format specified in
        'tstoolbox read ...'.
    sim_col
        If integer represents the column number of standard input. Can be
        a csv, wdm, hdf or xlsx file following format specified in
        'tstoolbox read ...'.
    stats: str
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
        | me              | Mean error or bias                               |
        |                 | -inf < ME < inf, close to 0 is better            |
        +-----------------+--------------------------------------------------+
        | pc_bias         | Percent Bias                                     |
        |                 | -inf < PC_BIAS < inf, close to 0 is better       |
        +-----------------+--------------------------------------------------+
        | apc_bias        | Absolute Percent Bias                            |
        |                 | 0 <= APC_BIAS < inf, close to 0 is better        |
        +-----------------+--------------------------------------------------+
        | rmsd            | Root Mean Square Deviation/Error                 |
        |                 | 0 <= RMSD < inf, smaller is better               |
        +-----------------+--------------------------------------------------+
        | crmsd           | Centered Root Mean Square Deviation/Error        |
        +-----------------+--------------------------------------------------+
        | corrcoef        | Pearson Correlation coefficient (r)              |
        |                 | -1 <= r <= 1                                     |
        |                 | 1 perfect positive correlation                   |
        |                 | 0 complete randomness                            |
        |                 | -1 perfect negative correlation                  |
        +-----------------+--------------------------------------------------+
        | coefdet         | Coefficient of determination (r^2)               |
        |                 | 0 <= r^2 <= 1                                    |
        |                 | 1 perfect correlation                            |
        |                 | 0 complete randomness                            |
        +-----------------+--------------------------------------------------+
        | murphyss        | Murphy Skill Score                               |
        +-----------------+--------------------------------------------------+
        | nse             | Nash-Sutcliffe Efficiency                        |
        |                 | -inf < NSE < 1, larger is better                 |
        +-----------------+--------------------------------------------------+
        | kge09           | Kling-Gupta Efficiency, 2009                     |
        |                 | -inf < KGE09 < 1, larger is better               |
        +-----------------+--------------------------------------------------+
        | kge12           | Kling-Gupta Efficiency, 2012                     |
        |                 | -inf < KGE12 < 1, larger is better               |
        +-----------------+--------------------------------------------------+
        | index_agreement | Index of agreement (d)                           |
        |                 | 0 <= d < 1, larger is better                     |
        +-----------------+--------------------------------------------------+
        | brierss         | Brier Skill Score                                |
        +-----------------+--------------------------------------------------+
        | mae             | Mean Absolute Error                              |
        |                 | 0 <= MAE < 1, larger is better                   |
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
        |             | -1 <= r <= 1                                          |
        |             | 1 positive correlation of variation in anomalies      |
        |             | 0 complete randomness of variation in anomalies       |
        |             | -1 negative correlation of variation in anomalies     |
        +-------------+-------------------------------------------------------+
        | d1          | Index of agreement (d1)                               |
        |             | 0 <= d1 < 1, larger is better                         |
        +-------------+-------------------------------------------------------+
        | d1_p        | Legate-McCabe Index of Agreement                      |
        |             | 0 <= d1_p < 1, larger is better                       |
        +-------------+-------------------------------------------------------+
        | d           | Index of agreement (d)                                |
        |             | 0 <= d < 1, larger is better                          |
        +-------------+-------------------------------------------------------+
        | dmod        | Modified index of agreement (dmod)                    |
        |             | 0 <= dmod < 1, larger is better                       |
        +-------------+-------------------------------------------------------+
        | drel        | Relative index of agreement (drel)                    |
        |             | 0 <= drel < 1, larger is better                       |
        +-------------+-------------------------------------------------------+
        | dr          | Refined index of agreement (dr)                       |
        |             | -1 <= dr < 1, larger is better                        |
        +-------------+-------------------------------------------------------+
        | ed          | Euclidean distance in vector space                    |
        |             | 0 <= ed < inf, smaller is better                      |
        +-------------+-------------------------------------------------------+
        | g_mean_diff | Geometric mean difference                             |
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
        | h10_mahe    | H10 mean absolute error                               |
        +-------------+-------------------------------------------------------+
        | h10_mhe     | H10 mean error                                        |
        +-------------+-------------------------------------------------------+
        | h10_rmshe   | H10 root mean square error                            |
        +-------------+-------------------------------------------------------+
        | irmse       | Inertial root mean square error (IRMSE)               |
        |             | 0 <= irmse < inf, smaller is better                   |
        +-------------+-------------------------------------------------------+
        | lm_index    | Legate-McCabe Efficiency Index                        |
        |             | 0 <= lm_index < 1, larger is better                   |
        +-------------+-------------------------------------------------------+
        | maape       | Mean Arctangent Absolute Percentage Error (MAAPE)     |
        |             | 0 <= maape < pi/2, smaller is better                  |
        +-------------+-------------------------------------------------------+
        | male        | Mean absolute log error                               |
        |             | 0 <= male < inf, smaller is better                    |
        +-------------+-------------------------------------------------------+
        | mapd        | Mean absolute percentage deviation (MAPD)             |
        +-------------+-------------------------------------------------------+
        | mape        | Mean absolute percentage error (MAPE)                 |
        |             | 0 <= mape < inf, 0 indicates perfect correlation      |
        +-------------+-------------------------------------------------------+
        | mase        | Mean absolute scaled error                            |
        +-------------+-------------------------------------------------------+
        | mb_r        | Mielke-Berry R value (MB R)                           |
        |             | 0 <= mb_r < 1, larger is better                       |
        +-------------+-------------------------------------------------------+
        | mdae        | Median absolute error (MdAE)                          |
        |             | 0 <= mdae < inf, smaller is better                    |
        +-------------+-------------------------------------------------------+
        | mde         | Median error (MdE)                                    |
        |             | -inf < mde < inf, closer to zero is better            |
        +-------------+-------------------------------------------------------+
        | mdse        | Median squared error (MdSE)                           |
        |             | 0 < mde < inf, closer to zero is better               |
        +-------------+-------------------------------------------------------+
        | mean_var    | Mean variance                                         |
        +-------------+-------------------------------------------------------+
        | me          | Mean error                                            |
        |             | -inf < me < inf, closer to zero is better             |
        +-------------+-------------------------------------------------------+
        | mle         | Mean log error                                        |
        |             | -inf < mle < inf, closer to zero is better            |
        +-------------+-------------------------------------------------------+
        | mse         | Mean squared error                                    |
        |             | 0 <= mse < inf, smaller is better                     |
        +-------------+-------------------------------------------------------+
        | msle        | Mean squared log error                                |
        |             | 0 <= msle < inf, smaller is better                    |
        +-------------+-------------------------------------------------------+
        | ned         | Normalized Euclidian distance in vector space         |
        |             | 0 <= ned < inf, smaller is better                     |
        +-------------+-------------------------------------------------------+
        | nrmse_iqr   | IQR normalized root mean square error                 |
        |             | 0 <= nrmse_iqr < inf, smaller is better               |
        +-------------+-------------------------------------------------------+
        | nrmse_mean  | Mean normalized root mean square error                |
        |             | 0 <= nrmse_mean < inf, smaller is better              |
        +-------------+-------------------------------------------------------+
        | nrmse_range | Range normalized root mean square error               |
        |             | 0 <= nrmse_range < inf, smaller is better             |
        +-------------+-------------------------------------------------------+
        | nse_mod     | Modified Nash-Sutcliffe efficiency (NSE mod)          |
        |             | -inf < nse_mod < 1, larger is better                  |
        +-------------+-------------------------------------------------------+
        | nse_rel     | Relative Nash-Sutcliffe efficiency (NSE rel)          |
        |             | -inf < nse_mod < 1, larger is better                  |
        +-------------+-------------------------------------------------------+
        | rmse        | Root mean square error                                |
        |             | 0 <= rmse < inf, smaller is better                    |
        +-------------+-------------------------------------------------------+
        | rmsle       | Root mean square log error                            |
        |             | 0 <= rmsle < inf, smaller is better                   |
        +-------------+-------------------------------------------------------+
        | sa          | Spectral Angle (SA)                                   |
        |             | -pi/2 <= sa < pi/2, closer to 0 is better             |
        +-------------+-------------------------------------------------------+
        | sc          | Spectral Correlation (SC)                             |
        |             | -pi/2 <= sc < pi/2, closer to 0 is better             |
        +-------------+-------------------------------------------------------+
        | sga         | Spectral Gradient Angle (SGA)                         |
        |             | -pi/2 <= sga < pi/2, closer to 0 is better            |
        +-------------+-------------------------------------------------------+
        | sid         | Spectral Information Divergence (SID)                 |
        |             | -pi/2 <= sid < pi/2, closer to 0 is better            |
        +-------------+-------------------------------------------------------+
        | smape1      | Symmetric Mean Absolute Percentage Error (1) (SMAPE1) |
        |             | 0 <= smape1 < 100, smaller is better                  |
        +-------------+-------------------------------------------------------+
        | smape2      | Symmetric Mean Absolute Percentage Error (2) (SMAPE2) |
        |             | 0 <= smape2 < 100, smaller is better                  |
        +-------------+-------------------------------------------------------+
        | spearman_r  | Spearman rank correlation coefficient                 |
        |             | -1 <= spearman_r <= 1                                 |
        |             | 1 perfect positive correlation                        |
        |             | 0 complete randomness                                 |
        |             | -1 perfect negative correlation                       |
        +-------------+-------------------------------------------------------+
        | ve          | Volumetric Efficiency (VE)                            |
        |             | 0 <= ve < 1, smaller is better                        |
        +-------------+-------------------------------------------------------+
        | watt_m      | Watterson's M (M)                                     |
        |             | -1 <= watt_m < 1, larger is better                    |
        +-------------+-------------------------------------------------------+
    replace_nan: float
        If given, indicates which value to replace NaN values with in the two
        arrays. If None, when a NaN value is found at the i-th position in the
        observed OR simulated array, the i-th value of the observed and
        simulated array are removed before the computation.
    replace_inf: float
        If given, indicates which value to replace Inf values with in the two
        arrays. If None, when an inf value is found at the i-th position in the
        observed OR simulated array, the i-th value of the observed and
        simulated array are removed before the computation.
    remove_neg: boolean
        If True, when a negative value is found at the i-th position in the
        observed OR simulated array, the i-th value of the observed AND
        simulated array are removed before the computation.
    remove_zero: boolean
        If true, when a zero value is found at the i-th position in the
        observed OR simulated array, the i-th value of the observed AND
        simulated array are removed before the computation.
    ${start_date}
    ${end_date}
    ${round_index}
    ${clean}
    ${index_type}
    ${source_units}
    ${target_units}
    ${tablefmt}
    ${float_format}
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

        Scaling factor for `kge09` and `kge12` beta.
    """
    obs_col = obs_col or 1
    sim_col = sim_col or 2
    tsutils.printiso(
        gof(
            obs_col=obs_col,
            sim_col=sim_col,
            stats=stats,
            replace_nan=replace_nan,
            replace_inf=replace_inf,
            remove_neg=remove_neg,
            remove_zero=remove_zero,
            start_date=start_date,
            end_date=end_date,
            round_index=round_index,
            clean=clean,
            index_type=index_type,
            source_units=source_units,
            target_units=target_units,
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
    "bias": ["Mean error or bias", he.me],
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
    "brierss": ["Brier's Score", lambda sim, obs: np.sum(sim - obs) ** 2 / len(sim)],
    "mae": ["Mean Absolute Error", he.mae],
    "mean": ["Mean", lambda x: x],
    "stdev": ["Standard deviation", lambda x: x],
    "acc": ["Anomaly correlation coefficient (ACC)", he.acc],
    "d1": ["Index of agreement (d1)", he.d1],
    "d1_p": ["Legate-McCabe Index of Agreement", he.d1_p],
    "d": ["Index of agreement (d)", he.d],
    "dmod": ["Modified index of agreement (dmod)", he.dmod],
    "drel": ["Relative index of agreement (drel)", he.drel],
    "dr": ["refined index of agreement (dr)", he.dr],
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
    "me": ["Mean error or bias", he.me],
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
@tsutils.copy_doc(gof_cli)
def gof(
    obs_col=1,
    sim_col=2,
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
    start_date=None,
    end_date=None,
    round_index=None,
    clean=False,
    index_type="datetime",
    source_units=None,
    target_units=None,
    kge_sr: float = 1.0,
    kge09_salpha: float = 1.0,
    kge12_sgamma: float = 1.0,
    kge_sbeta: float = 1.0,
):
    """Will calculate goodness of fit statistics between two time-series."""
    if "default" in stats:
        stats = [
            "me",
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
    tsd = read(
        obs_col,
        sim_col,
        index_type=index_type,
        start_date=start_date,
        end_date=end_date,
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
The "gof" requires only two time-series, the first one is the observed values
and the second is the simulated.  """
            )
        )
    lennao, lennas = tsd.isna().sum()

    tsd = tsd.dropna(how="any")

    statval = []

    obs = tsd.iloc[:, 0].astype("float64")
    sim = tsd.iloc[:, 1].astype("float64")

    nstats = [i for i in stats if i not in ["mean", "stdev"]]

    for stat in nstats:
        extra_args = {
            "replace_nan": replace_nan,
            "replace_inf": replace_inf,
            "remove_neg": remove_neg,
            "remove_zero": remove_zero,
        }
        if stat in ["crmsd", "murphyss", "brierss", "pc_bias", "apc_bias"]:
            extra_args = {}
        proc = stats_dict[stat]
        if "kge09" == stat:
            extra_args = {"s": (kge_sr, kge09_salpha, kge_sbeta)}
        if "kge12" == stat:
            extra_args = {"s": (kge_sr, kge12_sgamma, kge_sbeta)}
        statval.append(
            [
                proc[0],
                proc[1](sim, obs, **extra_args),
            ]
        )

    statval.append(["Common count observed and simulated", len(tsd.index)])

    statval.append(["Count of NaNs", None, lennao, lennas])

    if "mean" in stats:
        statval.append(["Mean", None, obs.mean(), sim.mean()])

    if "stdev" in stats:
        statval.append(["Standard deviation", None, obs.std(), sim.std()])

    return statval
