# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import mando
import pandas as pd
import pyaf.ForecastEngine as autof
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

# Placeholder for future forecast subcommand.
# mando.main.add_subprog("forecast", help="Forecast algorithms")


@mando.command("forecast", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def forecast_cli(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    print_input=False,
    tablefmt="csv",
    horizon=2,
    print_cols="all",
):
    """Machine learning automatic forecasting

    Machine learning forecast using PyAF (Python Automatic Forecasting)

    Uses a machine learning approach (The signal is cut into estimation
    and validation parts, respectively, 80% and 20% of the signal).  A
    time-series cross-validation can also be used.

    Forecasting a time series model on a given horizon (forecast result
    is also pandas data-frame) and providing prediction/confidence
    intervals for the forecasts.

    Generic training features

    * Signal decomposition as the sum of a trend, periodic and AR
      component
    * Works as a competition between a comprehensive set of possible
      signal transformations and linear decompositions. For each
      transformed signal , a set of possible trends, periodic components
      and AR models is generated and all the possible combinations are
      estimated.  The best decomposition in term of performance is kept
      to forecast the signal (the performance is computed on a part of
      the signal that was not used for the estimation).
    * Signal transformation is supported before signal decompositions.
      Four transformations are supported by default. Other
      transformation are available (Box-Cox etc).
    * All Models are estimated using standard procedures and
      state-of-the-art time series modeling. For example, trend
      regressions and AR/ARX models are estimated using scikit-learn
      linear regression models.
    * Standard performance measures are used (L1, RMSE, MAPE, etc)

    Exogenous Data Support

    * Exogenous data can be provided to improve the forecasts. These are
      expected to be stored in an external data-frame (this data-frame
      will be merged with the training data-frame).
    * Exogenous data are integrated in the modeling process through
      their past values (ARX models).
    * Exogenous variables can be of any type (numeric, string , date, or
      object).
    * Exogenous variables are dummified for the non-numeric types, and
      standardized for the numeric types.

    Hierarchical Forecasting

    * Bottom-Up, Top-Down (using proportions), Middle-Out and Optimal
      Combinations are implemented.
    * The modeling process is customizable and has a huge set of
      options. The default values of these options should however be OK
      to produce a reasonable quality model in a limited amount of time
      (a few minutes).

    Parameters
    ----------
    ${input_ts}
    ${start_date}
    ${end_date}
    ${skiprows}
    ${names}
    ${columns}
    ${dropna}
    ${clean}
    ${source_units}
    ${target_units}
    ${round_index}
    ${index_type}
    ${print_input}
    ${tablefmt}
    horizon: int
        Number of intervals to forecast.
    print_cols: str
        Identifies what columns to return.  One of "all" or "forecast"

    """
    tsutils.printiso(
        forecast(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            source_units=source_units,
            target_units=target_units,
            print_input=print_input,
            horizon=horizon,
            print_cols=print_cols,
        ),
        tablefmt=tablefmt,
    )


@tsutils.copy_doc(forecast_cli)
def forecast(
    input_ts="-",
    start_date=None,
    end_date=None,
    columns=None,
    dropna="no",
    round_index=None,
    clean=False,
    target_units=None,
    source_units=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    horizon=2,
    print_input=False,
    print_cols="all",
):
    tsd = tsutils.common_kwds(
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

    index_name = tsd.index.name

    # create a forecast engine. This is the main object handling all the
    # operations
    lEngine = autof.cForecastEngine()

    multiple_cols = len(tsd.columns) > 1

    if multiple_cols is True and print_cols != "forecast":
        raise ValueError(
            tsutils.error_wrapper(
                f"""
To forecast multiple columns requires `print_cols` to be "forecast", not
{print_cols}.  """
            )
        )

    rtsd = pd.DataFrame()
    for col in tsd.columns:
        ntsd = pd.DataFrame(
            {
                index_name: tsd.index.astype("M8[ns]").values,
                col: tsd[col].values.astype("float64"),
            }
        )
        print(ntsd)
        # get the best time series model for predicting
        lEngine.train(iInputDS=ntsd, iTime=index_name, iSignal=col, iHorizon=horizon)
        lEngine.getModelInfo()

        df_forecast = lEngine.forecast(iInputDS=ntsd, iHorizon=horizon)

        df_forecast = df_forecast.set_index(index_name)

        if print_cols == "forecast":
            rtsd = rtsd.join(df_forecast[col + "_Forecast"], how="outer")

    if print_cols == "forecast":
        return tsutils.return_input(print_input, tsd, rtsd)
    return tsutils.return_input(print_input, tsd, df_forecast)


# placeholder for future arima...

# @mando.main.forecast.command("arima", formatter_class=RSTHelpFormatter, doctype="numpy")
# @tsutils.doc(tsutils.docstrings)
# def arima_cli(
#     input_ts="-",
#     columns=None,
#     start_date=None,
#     end_date=None,
#     dropna="no",
#     clean=False,
#     round_index=None,
#     skiprows=None,
#     index_type="datetime",
#     names=None,
#     source_units=None,
#     target_units=None,
#     print_input=False,
#     tablefmt="csv",
#     horizon=2,
#     print_cols="all",
# ):
#     """ARIMA
#
#     ARIMA Model
#
#     Parameters
#     ----------
#     ${input_ts}
#     ${start_date}
#     ${end_date}
#     ${skiprows}
#     ${names}
#     ${columns}
#     ${dropna}
#     ${clean}
#     ${source_units}
#     ${target_units}
#     ${round_index}
#     ${index_type}
#     ${print_input}
#     ${tablefmt}
#     print_cols: str
#         Identifies what columns to return.  One of "all" or "forecast"
#
#     """
#     tsutils.printiso(
#         arima(
#             input_ts=input_ts,
#             columns=columns,
#             start_date=start_date,
#             end_date=end_date,
#             dropna=dropna,
#             clean=clean,
#             round_index=round_index,
#             skiprows=skiprows,
#             index_type=index_type,
#             names=names,
#             source_units=source_units,
#             target_units=target_units,
#             print_input=print_input,
#             horizon=horizon,
#             print_cols=print_cols,
#         ),
#         tablefmt=tablefmt,
#     )
#
#
# def arima(
#     input_ts="-",
#     start_date=None,
#     end_date=None,
#     columns=None,
#     dropna="no",
#     round_index=None,
#     clean=False,
#     target_units=None,
#     source_units=None,
#     skiprows=None,
#     index_type="datetime",
#     names=None,
#     horizon=2,
#     print_input=False,
#     print_cols="all",
# ):
#     tsd = tsutils.common_kwds(
#         input_ts,
#         skiprows=skiprows,
#         names=names,
#         index_type=index_type,
#         start_date=start_date,
#         end_date=end_date,
#         pick=columns,
#         round_index=round_index,
#         dropna=dropna,
#         source_units=source_units,
#         target_units=target_units,
#         clean=clean,
#     )
