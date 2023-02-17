"""Collection of functions for the manipulation of time series."""

import os.path as _os_path
import sys as _sys
import warnings as _warnings

from toolbox_utils import tsutils as _tsutils

__all__ = [
    "unstack",
    "accumulate",
    "add_trend",
    "aggregate",
    "calculate_fdc",
    "calculate_kde",
    "clip",
    "convert",
    "convert_index",
    "convert_index_to_julian",
    "converttz",
    "correlation",
    "createts",
    "date_offset",
    "date_slice",
    "describe",
    "dtw",
    "equation",
    "ewm_window",
    "expanding_window",
    "fill",
    "filter",
    "fit",
    "forecast",
    "gof",
    "lag",
    "normalization",
    "pca",
    "pct_change",
    "peak_detection",
    "pick",
    "plot",
    "rank",
    "read",
    "regression",
    "remove_trend",
    "replace",
    "rolling_window",
    "stack",
    "stdtozrxp",
    "tstopickle",
    "unstack",
]

from tstoolbox.functions.accumulate import accumulate
from tstoolbox.functions.add_trend import add_trend
from tstoolbox.functions.aggregate import aggregate
from tstoolbox.functions.calculate_fdc import calculate_fdc
from tstoolbox.functions.calculate_kde import calculate_kde
from tstoolbox.functions.clip import clip
from tstoolbox.functions.convert import convert
from tstoolbox.functions.convert_index import convert_index
from tstoolbox.functions.convert_index_to_julian import convert_index_to_julian
from tstoolbox.functions.converttz import converttz
from tstoolbox.functions.correlation import correlation
from tstoolbox.functions.createts import createts
from tstoolbox.functions.date_offset import date_offset
from tstoolbox.functions.date_slice import date_slice
from tstoolbox.functions.describe import describe
from tstoolbox.functions.dtw import dtw
from tstoolbox.functions.equation import equation
from tstoolbox.functions.ewm_window import ewm_window
from tstoolbox.functions.expanding_window import expanding_window
from tstoolbox.functions.fill import fill
from tstoolbox.functions.filter import filter
from tstoolbox.functions.fit import fit
from tstoolbox.functions.forecast import forecast
from tstoolbox.functions.gof import gof
from tstoolbox.functions.lag import lag
from tstoolbox.functions.normalization import normalization
from tstoolbox.functions.pca import pca
from tstoolbox.functions.pct_change import pct_change
from tstoolbox.functions.peak_detection import peak_detection
from tstoolbox.functions.pick import pick
from tstoolbox.functions.plot import plot
from tstoolbox.functions.rank import rank
from tstoolbox.functions.read import read
from tstoolbox.functions.regression import regression
from tstoolbox.functions.remove_trend import remove_trend
from tstoolbox.functions.replace import replace
from tstoolbox.functions.rolling_window import rolling_window
from tstoolbox.functions.stack import stack
from tstoolbox.functions.stdtozrxp import stdtozrxp
from tstoolbox.functions.tstopickle import tstopickle
from tstoolbox.functions.unstack import unstack

_warnings.filterwarnings("ignore")


def main():
    """Set debug and run cltoolbox.main function."""
    if not _os_path.exists("debug_tstoolbox"):
        _sys.tracebacklimit = 0

    import cltoolbox
    from cltoolbox.rst_text_formatter import RSTHelpFormatter

    @cltoolbox.command()
    def about():
        """Display version number and system information."""
        _tsutils.about(__name__)

    @cltoolbox.command("accumulate", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(accumulate)
    def accumulate_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        dropna="no",
        clean=False,
        statistic="sum",
        round_index=None,
        skiprows=None,
        index_type="datetime",
        names=None,
        source_units=None,
        target_units=None,
        print_input=False,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            accumulate(
                input_ts=input_ts,
                skiprows=skiprows,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                clean=clean,
                statistic=statistic,
                round_index=round_index,
                index_type=index_type,
                names=names,
                source_units=source_units,
                target_units=target_units,
                print_input=print_input,
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("add_trend", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(add_trend)
    def add_trend_cli(
        start_offset,
        end_offset,
        start_index=0,
        end_index=-1,
        input_ts="-",
        start_date=None,
        end_date=None,
        skiprows=None,
        columns=None,
        clean=False,
        dropna="no",
        names=None,
        source_units=None,
        target_units=None,
        round_index=None,
        index_type="datetime",
        print_input=False,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            add_trend(
                start_offset,
                end_offset,
                start_index=start_index,
                end_index=end_index,
                input_ts=input_ts,
                columns=columns,
                clean=clean,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                round_index=round_index,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                source_units=source_units,
                target_units=target_units,
                print_input=print_input,
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("aggregate", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(aggregate)
    def aggregate_cli(
        input_ts="-",
        groupby=None,
        statistic="mean",
        columns=None,
        start_date=None,
        end_date=None,
        dropna="no",
        clean=False,
        agg_interval=None,
        ninterval=None,
        round_index=None,
        skiprows=None,
        index_type="datetime",
        names=None,
        source_units=None,
        target_units=None,
        print_input=False,
        tablefmt="csv",
        min_count=0,
    ):
        _tsutils.printiso(
            aggregate(
                input_ts=input_ts,
                groupby=groupby,
                statistic=statistic,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                clean=clean,
                agg_interval=agg_interval,
                ninterval=ninterval,
                round_index=round_index,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                source_units=source_units,
                target_units=target_units,
                print_input=print_input,
                min_count=min_count,
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("calculate_fdc", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(calculate_fdc)
    def calculate_fdc_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        clean=False,
        skiprows=None,
        index_type="datetime",
        names=None,
        percent_point_function=None,
        plotting_position="weibull",
        source_units=None,
        target_units=None,
        sort_values="ascending",
        sort_index="ascending",
        tablefmt="csv",
        add_index=False,
        include_ri=False,
        include_sd=False,
        include_cl=False,
        ci=0.9,
    ):
        _tsutils.printiso(
            calculate_fdc(
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                clean=clean,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                percent_point_function=percent_point_function,
                plotting_position=plotting_position,
                source_units=source_units,
                target_units=target_units,
                sort_values=sort_values,
                sort_index=sort_index,
                add_index=add_index,
                include_ri=include_ri,
                include_sd=include_sd,
                include_cl=include_cl,
                ci=ci,
            ),
            showindex="always",
            tablefmt=tablefmt,
        )

    @cltoolbox.command("calculate_kde", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(calculate_kde)
    def calculate_kde_cli(
        ascending=True,
        evaluate=False,
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        clean=False,
        skiprows=None,
        index_type="datetime",
        source_units=None,
        target_units=None,
        names=None,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            calculate_kde(
                ascending=ascending,
                evaluate=evaluate,
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                clean=clean,
                skiprows=skiprows,
                index_type=index_type,
                source_units=source_units,
                target_units=target_units,
                names=names,
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("clip", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(clip)
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
        _tsutils.printiso(
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

    @cltoolbox.command("convert", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(convert)
    def convert_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        dropna="no",
        clean=False,
        skiprows=None,
        index_type="datetime",
        names=None,
        factor=1.0,
        offset=0.0,
        print_input=False,
        round_index=None,
        source_units=None,
        target_units=None,
        float_format="g",
        tablefmt="csv",
    ):
        _tsutils.printiso(
            convert(
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                clean=clean,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                factor=factor,
                offset=offset,
                print_input=print_input,
                round_index=round_index,
                source_units=source_units,
                target_units=target_units,
            ),
            float_format=float_format,
            tablefmt=tablefmt,
        )

    @cltoolbox.command("convert_index", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(convert_index)
    def convert_index_cli(
        to,
        interval=None,
        epoch="julian",
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        round_index=None,
        dropna="no",
        clean=False,
        names=None,
        source_units=None,
        target_units=None,
        skiprows=None,
        tablefmt="csv",
    ):
        tsd = convert_index(
            to,
            interval=interval,
            epoch=epoch,
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            round_index=round_index,
            dropna=dropna,
            clean=clean,
            names=names,
            source_units=source_units,
            target_units=target_units,
            skiprows=skiprows,
        )
        _tsutils.printiso(tsd, tablefmt=tablefmt)

    @cltoolbox.command("convert_index_to_julian", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(convert_index_to_julian)
    def convert_index_to_julian_cli(
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
        )

    @cltoolbox.command("converttz", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(converttz)
    def converttz_cli(
        fromtz,
        totz,
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
        tablefmt="csv",
    ):
        _tsutils.printiso(
            converttz(
                fromtz,
                totz,
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                round_index=round_index,
                dropna=dropna,
                clean=clean,
                index_type=index_type,
                names=names,
                source_units=source_units,
                target_units=target_units,
                skiprows=skiprows,
            ),
            showindex="always",
            tablefmt=tablefmt,
        )

    @cltoolbox.command("correlation", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(correlation)
    def correlation_cli(
        lags,
        method="pearson",
        input_ts="-",
        start_date=None,
        end_date=None,
        columns=None,
        clean=False,
        index_type="datetime",
        names=None,
        source_units=None,
        target_units=None,
        skiprows=None,
        tablefmt="csv",
        round_index=None,
        dropna="no",
    ):
        _tsutils.printiso(
            correlation(
                lags,
                method=method,
                input_ts=input_ts,
                start_date=start_date,
                end_date=end_date,
                columns=columns,
                clean=clean,
                index_type=index_type,
                names=names,
                source_units=source_units,
                target_units=target_units,
                skiprows=skiprows,
                round_index=round_index,
                dropna=dropna,
            ),
            showindex="always",
            tablefmt=tablefmt,
        )

    @cltoolbox.command("createts", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(createts)
    def createts_cli(
        freq=None,
        fillvalue=None,
        input_ts=None,
        index_type="datetime",
        start_date=None,
        end_date=None,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            createts(
                freq=freq,
                fillvalue=fillvalue,
                input_ts=input_ts,
                index_type=index_type,
                start_date=start_date,
                end_date=end_date,
            ),
            showindex="always",
            tablefmt=tablefmt,
        )

    @cltoolbox.command("date_offset", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(date_offset)
    def date_offset_cli(
        intervals,
        offset,
        columns=None,
        dropna="no",
        clean=False,
        skiprows=None,
        index_type="datetime",
        names=None,
        input_ts="-",
        start_date=None,
        end_date=None,
        source_units=None,
        target_units=None,
        round_index=None,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            date_offset(
                intervals,
                offset,
                columns=columns,
                dropna=dropna,
                clean=clean,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                input_ts=input_ts,
                start_date=start_date,
                end_date=end_date,
                source_units=source_units,
                target_units=target_units,
                round_index=round_index,
            ),
            showindex="always",
            tablefmt=tablefmt,
        )

    @cltoolbox.command("date_slice", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(date_slice)
    def date_slice_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        dropna="no",
        clean=False,
        skiprows=None,
        index_type="datetime",
        names=None,
        round_index=None,
        source_units=None,
        target_units=None,
        float_format="g",
        tablefmt="csv",
    ):
        _tsutils.printiso(
            date_slice(
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                clean=clean,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                round_index=round_index,
                source_units=source_units,
                target_units=target_units,
            ),
            float_format=float_format,
            tablefmt=tablefmt,
        )

    @cltoolbox.command("describe", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(describe)
    def describe_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        dropna="no",
        skiprows=None,
        index_type="datetime",
        names=None,
        clean=False,
        transpose=False,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            describe(
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                clean=clean,
                transpose=transpose,
            ),
            showindex="always",
            tablefmt=tablefmt,
        )

    @cltoolbox.command("dtw", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(dtw)
    def dtw_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        round_index=None,
        dropna="no",
        skiprows=None,
        index_type="datetime",
        names=None,
        clean=False,
        window=10000,
        source_units=None,
        target_units=None,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            dtw(
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                round_index=round_index,
                dropna=dropna,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                clean=clean,
                window=window,
                source_units=source_units,
                target_units=target_units,
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("equation", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(equation)
    def equation_cli(
        equation_str,
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        dropna="no",
        skiprows=None,
        index_type="datetime",
        names=None,
        clean=False,
        print_input="",
        round_index=None,
        source_units=None,
        target_units=None,
        float_format="g",
        tablefmt="csv",
        output_names="",
    ):
        _tsutils.printiso(
            equation(
                equation_str,
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                clean=clean,
                print_input=print_input,
                round_index=round_index,
                source_units=source_units,
                target_units=target_units,
                output_names=output_names,
            ),
            float_format=float_format,
            tablefmt=tablefmt,
        )

    @cltoolbox.command("ewm_window", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(ewm_window)
    def ewm_window_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        dropna="no",
        skiprows=None,
        index_type="datetime",
        names=None,
        clean=False,
        statistic="",
        alpha_com=None,
        alpha_span=None,
        alpha_halflife=None,
        alpha=None,
        min_periods=0,
        adjust=True,
        ignore_na=False,
        source_units=None,
        target_units=None,
        print_input=False,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            ewm_window(
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                clean=clean,
                statistic=statistic,
                alpha_com=alpha_com,
                alpha_span=alpha_span,
                alpha_halflife=alpha_halflife,
                alpha=alpha,
                min_periods=min_periods,
                adjust=adjust,
                ignore_na=ignore_na,
                source_units=source_units,
                target_units=target_units,
                print_input=print_input,
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("expanding_window", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(expanding_window)
    def expanding_window_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        dropna="no",
        skiprows=None,
        index_type="datetime",
        names=None,
        clean=False,
        statistic="",
        min_periods=1,
        center=False,
        source_units=None,
        target_units=None,
        print_input=False,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            expanding_window(
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                clean=clean,
                statistic=statistic,
                min_periods=min_periods,
                center=center,
                source_units=source_units,
                target_units=target_units,
                print_input=print_input,
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("fill", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(fill)
    def fill_cli(
        input_ts="-",
        method="ffill",
        print_input=False,
        start_date=None,
        end_date=None,
        columns=None,
        clean=False,
        index_type="datetime",
        names=None,
        source_units=None,
        target_units=None,
        skiprows=None,
        from_columns=None,
        to_columns=None,
        limit=None,
        order=None,
        tablefmt="csv",
        force_freq=None,
    ):
        _tsutils.printiso(
            fill(
                input_ts=input_ts,
                method=method,
                print_input=print_input,
                start_date=start_date,
                end_date=end_date,
                columns=columns,
                clean=clean,
                index_type=index_type,
                names=names,
                source_units=source_units,
                target_units=target_units,
                skiprows=skiprows,
                from_columns=from_columns,
                to_columns=to_columns,
                limit=limit,
                order=order,
                force_freq=force_freq,
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("filter", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(filter)
    def filter_cli(
        filter_type,
        filter_pass,
        lowpass_cutoff=None,
        highpass_cutoff=None,
        window_len=3,
        butterworth_stages=1,
        reverse_second_stage=True,
        input_ts="-",
        start_date=None,
        end_date=None,
        columns=None,
        float_format="g",
        dropna="no",
        skiprows=None,
        index_type="datetime",
        names=None,
        clean=False,
        round_index=None,
        source_units=None,
        target_units=None,
        print_input=False,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            filter(
                filter_type,
                filter_pass,
                butterworth_stages=butterworth_stages,
                reverse_second_stage=reverse_second_stage,
                lowpass_cutoff=lowpass_cutoff,
                highpass_cutoff=highpass_cutoff,
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                clean=clean,
                print_input=print_input,
                window_len=window_len,
                source_units=source_units,
                target_units=target_units,
                round_index=round_index,
            ),
            float_format=float_format,
            tablefmt=tablefmt,
        )

    @cltoolbox.command("fit", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(fit)
    def fit_cli(
        method,
        lowess_frac=0.01,
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
    ):
        _tsutils.printiso(
            fit(
                method,
                lowess_frac=lowess_frac,
                input_ts=input_ts,
                skiprows=skiprows,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                clean=clean,
                round_index=round_index,
                index_type=index_type,
                names=names,
                source_units=source_units,
                target_units=target_units,
                print_input=print_input,
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("forecast", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(forecast)
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
        _tsutils.printiso(
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

    @cltoolbox.command("gof", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(gof)
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
        obs_col = obs_col or 1
        sim_col = sim_col or 2
        _tsutils.printiso(
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

    @cltoolbox.command("lag", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(lag)
    def lag_cli(
        lags,
        input_ts="-",
        print_input=False,
        start_date=None,
        end_date=None,
        columns=None,
        clean=False,
        index_type="datetime",
        names=None,
        source_units=None,
        target_units=None,
        skiprows=None,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            lag(
                lags,
                input_ts=input_ts,
                print_input=print_input,
                start_date=start_date,
                end_date=end_date,
                columns=columns,
                clean=clean,
                index_type=index_type,
                names=names,
                source_units=source_units,
                target_units=target_units,
                skiprows=skiprows,
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("normalization", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(normalization)
    def normalization_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        dropna="no",
        skiprows=None,
        index_type="datetime",
        names=None,
        clean=False,
        mode="minmax",
        min_limit=0,
        max_limit=1,
        pct_rank_method="average",
        print_input=False,
        round_index=None,
        source_units=None,
        target_units=None,
        float_format="g",
        tablefmt="csv",
        with_centering=True,
        with_scaling=True,
        quantile_range=(0.25, 0.75),
    ):
        _tsutils.printiso(
            normalization(
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                clean=clean,
                mode=mode,
                min_limit=min_limit,
                max_limit=max_limit,
                pct_rank_method=pct_rank_method,
                print_input=print_input,
                round_index=round_index,
                source_units=source_units,
                target_units=target_units,
                with_centering=with_centering,
                with_scaling=with_scaling,
                quantile_range=quantile_range,
            ),
            float_format=float_format,
            tablefmt=tablefmt,
        )

    @cltoolbox.command("pca", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(pca)
    def pca_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        clean=False,
        skiprows=None,
        index_type="datetime",
        names=None,
        n_components=None,
        source_units=None,
        target_units=None,
        round_index=None,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            pca(
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                clean=clean,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                n_components=n_components,
                source_units=source_units,
                target_units=target_units,
                round_index=round_index,
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("pct_change", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(pct_change)
    def pct_change_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        dropna="no",
        skiprows=None,
        index_type="datetime",
        names=None,
        clean=False,
        periods=1,
        fill_method="pad",
        limit=None,
        freq=None,
        print_input=False,
        round_index=None,
        source_units=None,
        target_units=None,
        float_format="g",
        tablefmt="csv",
    ):
        _tsutils.printiso(
            pct_change(
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                clean=clean,
                periods=periods,
                fill_method=fill_method,
                limit=limit,
                freq=freq,
                print_input=print_input,
                round_index=round_index,
                source_units=source_units,
                target_units=target_units,
            ),
            float_format=float_format,
            tablefmt=tablefmt,
        )

    @cltoolbox.command("peak_detection", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(peak_detection)
    def peak_detection_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        dropna="no",
        skiprows=None,
        index_type="datetime",
        names=None,
        clean=False,
        method="rel",
        extrema="peak",
        window=24,
        pad_len=5,
        points=9,
        lock_frequency=False,
        float_format="g",
        round_index=None,
        source_units=None,
        target_units=None,
        print_input="",
        tablefmt="csv",
    ):
        _tsutils.printiso(
            peak_detection(
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                clean=clean,
                method=method,
                extrema=extrema,
                window=window,
                pad_len=pad_len,
                points=points,
                lock_frequency=lock_frequency,
                round_index=round_index,
                source_units=source_units,
                target_units=target_units,
                print_input=print_input,
            ),
            float_format=float_format,
            tablefmt=tablefmt,
        )

    @cltoolbox.command("pick", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(pick)
    def pick_cli(
        columns,
        input_ts="-",
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
        tablefmt="csv",
    ):
        _tsutils.printiso(
            pick(
                columns,
                input_ts=input_ts,
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
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("plot", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(plot)
    def plot_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        clean=False,
        skiprows=None,
        dropna="no",
        index_type="datetime",
        names=None,
        ofilename="plot.png",
        type="time",
        xtitle="",
        ytitle="",
        title="",
        figsize="10,6.0",
        legend=None,
        legend_names=None,
        subplots=False,
        sharex=True,
        sharey=False,
        colors="auto",
        linestyles="auto",
        markerstyles=" ",
        bar_hatchstyles="auto",
        style="auto",
        logx=False,
        logy=False,
        xaxis="arithmetic",
        yaxis="arithmetic",
        xlim=None,
        ylim=None,
        secondary_y=False,
        secondary_x=False,
        mark_right=True,
        scatter_matrix_diagonal="kde",
        bootstrap_size=50,
        bootstrap_samples=500,
        xy_match_line="",
        grid=False,
        label_rotation=None,
        label_skip=1,
        force_freq=None,
        drawstyle="default",
        por=False,
        invert_xaxis=False,
        invert_yaxis=False,
        round_index=None,
        plotting_position="weibull",
        prob_plot_sort_values="descending",
        source_units=None,
        target_units=None,
        lag_plot_lag=1,
        plot_styles="bright",
        hlines_y=None,
        hlines_xmin=None,
        hlines_xmax=None,
        hlines_colors="auto",
        hlines_linestyles="auto",
        vlines_x=None,
        vlines_ymin=None,
        vlines_ymax=None,
        vlines_colors="auto",
        vlines_linestyles="auto",
    ):
        pltr = plot(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            dropna=dropna,
            index_type=index_type,
            names=names,
            ofilename=ofilename,
            type=type,
            xtitle=xtitle,
            ytitle=ytitle,
            title=title,
            figsize=figsize,
            legend=legend,
            legend_names=legend_names,
            subplots=subplots,
            sharex=sharex,
            sharey=sharey,
            colors=colors,
            linestyles=linestyles,
            markerstyles=markerstyles,
            bar_hatchstyles=bar_hatchstyles,
            style=style,
            logx=logx,
            logy=logy,
            xaxis=xaxis,
            yaxis=yaxis,
            xlim=xlim,
            ylim=ylim,
            secondary_y=secondary_y,
            secondary_x=secondary_x,
            mark_right=mark_right,
            scatter_matrix_diagonal=scatter_matrix_diagonal,
            bootstrap_size=bootstrap_size,
            bootstrap_samples=bootstrap_samples,
            xy_match_line=xy_match_line,
            grid=grid,
            label_rotation=label_rotation,
            label_skip=label_skip,
            force_freq=force_freq,
            drawstyle=drawstyle,
            por=por,
            invert_xaxis=invert_xaxis,
            invert_yaxis=invert_yaxis,
            round_index=round_index,
            plotting_position=plotting_position,
            prob_plot_sort_values=prob_plot_sort_values,
            source_units=source_units,
            target_units=target_units,
            lag_plot_lag=lag_plot_lag,
            plot_styles=plot_styles,
            hlines_y=hlines_y,
            hlines_xmin=hlines_xmin,
            hlines_xmax=hlines_xmax,
            hlines_colors=hlines_colors,
            hlines_linestyles=hlines_linestyles,
            vlines_x=vlines_x,
            vlines_ymin=vlines_ymin,
            vlines_ymax=vlines_ymax,
            vlines_colors=vlines_colors,
            vlines_linestyles=vlines_linestyles,
        )
        return pltr

    @cltoolbox.command("rank", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(rank)
    def rank_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        dropna="no",
        skiprows=None,
        index_type="datetime",
        names=None,
        clean=False,
        axis=0,
        method="average",
        numeric_only=None,
        na_option="keep",
        ascending=True,
        pct=False,
        print_input=False,
        float_format="g",
        source_units=None,
        target_units=None,
        round_index=None,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            rank(
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                clean=clean,
                axis=axis,
                method=method,
                numeric_only=numeric_only,
                na_option=na_option,
                ascending=ascending,
                pct=pct,
                print_input=print_input,
                source_units=source_units,
                target_units=target_units,
                round_index=round_index,
            ),
            float_format=float_format,
            tablefmt=tablefmt,
        )

    @cltoolbox.command("read", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(read)
    def read_cli(
        force_freq=None,
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
        _tsutils.printiso(
            read(
                *filenames,
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

    @cltoolbox.command("regression", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(regression)
    def regression_cli(
        method,
        x_train_cols,
        y_train_col,
        x_pred_cols=None,
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
        print_input=False,
        tablefmt="csv",
        por=False,
    ):
        _tsutils.printiso(
            regression(
                method,
                x_train_cols,
                y_train_col,
                x_pred_cols=x_pred_cols,
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
                print_input=print_input,
                por=por,
            ),
            tablefmt=tablefmt,
            headers=[],
        )

    @cltoolbox.command("remove_trend", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(remove_trend)
    def remove_trend_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        dropna="no",
        skiprows=None,
        index_type="datetime",
        names=None,
        clean=False,
        round_index=None,
        source_units=None,
        target_units=None,
        print_input=False,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            remove_trend(
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                clean=clean,
                round_index=round_index,
                source_units=source_units,
                target_units=target_units,
                print_input=print_input,
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("replace", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(replace)
    def replace_cli(
        from_values,
        to_values,
        round_index=None,
        input_ts="-",
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
        print_input=False,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            replace(
                from_values,
                to_values,
                round_index=round_index,
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                clean=clean,
                source_units=source_units,
                target_units=target_units,
                print_input=print_input,
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("rolling_window", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(rolling_window)
    def rolling_window_cli(
        statistic,
        groupby=None,
        window=None,
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        dropna="no",
        skiprows=None,
        index_type="datetime",
        names=None,
        clean=False,
        span=None,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        closed=None,
        source_units=None,
        target_units=None,
        print_input=False,
        tablefmt="csv",
    ):
        _tsutils.printiso(
            rolling_window(
                statistic,
                groupby=groupby,
                window=window,
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                clean=clean,
                span=span,
                min_periods=min_periods,
                center=center,
                win_type=win_type,
                on=on,
                closed=closed,
                source_units=source_units,
                target_units=target_units,
                print_input=print_input,
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("stack", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(stack)
    def stack_cli(
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
        tablefmt="csv",
    ):
        _tsutils.printiso(
            stack(
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
            ),
            tablefmt=tablefmt,
        )

    @cltoolbox.command("stdtozrxp", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(stdtozrxp)
    def stdtozrxp_cli(
        input_ts="-",
        columns=None,
        start_date=None,
        end_date=None,
        dropna="no",
        skiprows=None,
        index_type="datetime",
        names=None,
        clean=False,
        round_index=None,
        source_units=None,
        target_units=None,
        rexchange=None,
    ):
        _tsutils.printiso(
            stdtozrxp(
                input_ts=input_ts,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                dropna=dropna,
                skiprows=skiprows,
                index_type=index_type,
                names=names,
                clean=clean,
                round_index=round_index,
                source_units=source_units,
                target_units=target_units,
                rexchange=rexchange,
            )
        )

    @cltoolbox.command("tstopickle", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(tstopickle)
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
        _tsutils.printiso(
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

    @cltoolbox.command("unstack", formatter_class=RSTHelpFormatter)
    @_tsutils.copy_doc(unstack)
    def unstack_cli(
        column_names,
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
        tablefmt="csv",
    ):
        _tsutils.printiso(
            unstack(
                column_names,
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
            ),
            tablefmt=tablefmt,
        )

    cltoolbox.main()


if __name__ == "__main__":
    from tstoolbox import tstoolbox

    df = tstoolbox.read("../../tests/02325000_flow.csv")
    filt_fft_high = filter(
        "fft", "highpass", print_input=True, input_ts=df, highpass_cutoff=10
    )
    filt_fft_low = filter(
        "fft", "lowpass", print_input=True, input_ts=df, lowpass_cutoff=10
    )
    filt_butter_high = filter(
        "butterworth", "highpass", print_input=True, input_ts=df, highpass_cutoff=0.4
    )
    filt_butter_low = filter(
        "butterworth", "lowpass", print_input=True, input_ts=df, lowpass_cutoff=0.4
    )


if __name__ == "__main__":
    main()
