# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from tstoolbox import tstoolbox

# Pull this in once.
df = tstoolbox.aggregate(
    agg_interval="D", clean=True, input_ts="tests/02234500_65_65.csv", skipna=False
)
# Pull this in once.
dfa = tstoolbox.aggregate(
    agg_interval="A", clean=True, input_ts="tests/02234500_65_65.csv", skipna=False
)


@pytest.mark.mpl_image_compare(tolerance=6)
def test_histogram():
    plt.close("all")
    return tstoolbox.plot(
        type="histogram",
        clean=True,
        input_ts="tests/02234500_65_65.csv",
        ofilename=None,
        sharex=False,
        plot_styles="classic",
    )


# @pytest.mark.mpl_image_compare(tolerance=6)
# def test_heatmap():
#     plt.close("all")
#     return tstoolbox.plot(
#         type="heatmap",
#         columns=2,
#         clean=True,
#         input_ts=df,
#         ofilename=None,
#         plot_styles="classic",
#     )


@pytest.mark.mpl_image_compare(tolerance=6)
def test_norm_xaxis():
    plt.close("all")
    return tstoolbox.plot(
        type="norm_xaxis",
        columns=2,
        clean=True,
        input_ts="tests/02234500_65_65.csv",
        ofilename=None,
        plot_styles="classic",
    )


@pytest.mark.mpl_image_compare(tolerance=6)
def test_norm_yaxis():
    plt.close("all")
    return tstoolbox.plot(
        type="norm_yaxis",
        columns=2,
        clean=True,
        input_ts="tests/02234500_65_65.csv",
        ofilename=None,
        plot_styles="classic",
    )


@pytest.mark.mpl_image_compare(tolerance=6)
def test_lognorm_xaxis():
    plt.close("all")
    return tstoolbox.plot(
        type="lognorm_xaxis",
        columns=2,
        clean=True,
        input_ts="tests/02234500_65_65.csv",
        ofilename=None,
        plot_styles="classic",
    )


@pytest.mark.mpl_image_compare(tolerance=6)
def test_lognorm_yaxis():
    plt.close("all")
    return tstoolbox.plot(
        type="lognorm_yaxis",
        columns=2,
        clean=True,
        input_ts="tests/02234500_65_65.csv",
        ofilename=None,
        plot_styles="classic",
    )


@pytest.mark.mpl_image_compare(tolerance=6)
def test_weibull_xaxis():
    plt.close("all")
    return tstoolbox.plot(
        type="weibull_xaxis",
        columns=2,
        clean=True,
        input_ts="tests/02234500_65_65.csv",
        ofilename=None,
        plot_styles="classic",
    )


@pytest.mark.mpl_image_compare(tolerance=6)
def test_weibull_yaxis():
    plt.close("all")
    return tstoolbox.plot(
        type="weibull_yaxis",
        columns=2,
        clean=True,
        input_ts="tests/02234500_65_65.csv",
        ofilename=None,
        plot_styles="classic",
    )


# @pytest.mark.mpl_image_compare(tolerance=6)
# def test_kde_time():
#     plt.close("all")
#     return tstoolbox.plot(
#         type="kde_time",
#         columns=2,
#         clean=True,
#         input_ts="tests/02234500_65_65.csv",
#         ofilename=None,
#         plot_styles="classic",
#     )


# @pytest.mark.mpl_image_compare(tolerance=6)
# def test_kde_time_multiple_traces():
#     plt.close("all")
#     ndf = tstoolbox.read(["tests/daily.csv", "tests/02325000_flow.csv"])
#     return tstoolbox.plot(
#         type="kde_time",
#         columns=[2, 3],
#         clean=True,
#         input_ts=ndf,
#         ytitle="Flow",
#         ofilename=None,
#         plot_styles="classic",
#     )


@pytest.mark.mpl_image_compare(tolerance=6)
def test_autocorrelation():
    plt.close("all")
    return tstoolbox.plot(
        type="autocorrelation",
        columns=2,
        input_ts=df,
        ofilename=None,
        plot_styles="classic",
    )
