import matplotlib
matplotlib.use('Agg')

import pytest

from tstoolbox import tstoolbox

# Pull this in once.
df = tstoolbox.aggregate(agg_interval='D',
                         clean=True,
                         input_ts='tests/02234500_65_65.csv')

@pytest.mark.mpl_image_compare
def test_time_plot():
    return tstoolbox.plot(type='time',
                          columns=1,
                          clean=True,
                          input_ts='tests/02234500_65_65.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_time_multiple_traces_plot():
    return tstoolbox.plot(type='time',
                          columns=[2,3],
                          style='b-,r*',
                          input_ts='tests/data_daily_sample.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_time_multiple_traces_style_plot():
    return tstoolbox.plot(type='time',
                          columns=[2,3],
                          style='b-,r  ',
                          input_ts='tests/data_daily_sample.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_time_multiple_traces_new_style_plot():
    return tstoolbox.plot(type='time',
                          columns=[2,3],
                          markerstyles=' ,*',
                          linestyles='-, ',
                          input_ts='tests/data_daily_sample.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_time_markers():
    return tstoolbox.plot(type='time',
                          columns=[2, 3],
                          linestyles=' ',
                          markerstyles='auto',
                          input_ts='tests/data_daily_sample.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_xy():
    return tstoolbox.plot(type='xy',
                          clean=True,
                          input_ts='tests/02234500_65_65.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_xy_multiple_traces():
    return tstoolbox.plot(type='xy',
                          columns=[2,3,3,2],
                          input_ts='tests/data_daily_sample.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_xy_multiple_traces_logy():
    return tstoolbox.plot(type='xy',
                          columns=[2,3,3,2],
                          yaxis='log',
                          input_ts='tests/data_daily_sample.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_xy_multiple_traces_logx():
    return tstoolbox.plot(type='xy',
                          columns=[2,3,3,2],
                          xaxis='log',
                          input_ts='tests/data_daily_sample.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_xy_multiple_traces_markers():
    return tstoolbox.plot(type='xy',
                          columns=[2,3,3,2],
                          linestyles=' ',
                          markerstyles='auto',
                          input_ts='tests/data_daily_sample.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_double_mass():
    return tstoolbox.plot(type='double_mass',
                          clean=True,
                          input_ts='tests/02234500_65_65.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_double_mass_mult():
    return tstoolbox.plot(type='double_mass',
                          columns=[2,3,3,2],
                          input_ts='tests/data_daily_sample.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_double_mass_marker():
    return tstoolbox.plot(type='double_mass',
                          columns=[2, 3, 3, 2],
                          linestyles=' ',
                          markerstyles='auto',
                          input_ts='tests/data_daily_sample.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_boxplot():
    ndf = tstoolbox.read(['tests/02234500_65_65.csv',
                          'tests/02325000_flow.csv'],
                         clean=True,
                         append='combine')
    return tstoolbox.plot(input_ts=ndf,
                          clean=True,
                          columns=[2, 3],
                          type='boxplot',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_scatter_matrix():
    return tstoolbox.plot(type='scatter_matrix',
                          clean=True,
                          input_ts='tests/02234500_65_65.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_lag_plot():
    return tstoolbox.plot(columns=1,
                          type='lag_plot',
                          input_ts=df,
                          ofilename=None)

# Can't have a bootstrap test since random selections are made.
# @image_comparison(baseline_images=['bootstrap'],
#                   tol=0.019, extensions=['png'])
# def test_bootstrap():
#     return tstoolbox.plot(type='bootstrap',
#                    clean=True,
#                    columns=2,
#                    input_ts='tests/02234500_65_65.csv')

@pytest.mark.mpl_image_compare
def test_probability_density():
    return tstoolbox.plot(type='probability_density',
                          clean=True,
                          input_ts='tests/02234500_65_65.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_bar():
    return tstoolbox.plot(type='bar',
                          input_ts=df,
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_barh():
    return tstoolbox.plot(type='barh',
                          input_ts=df,
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_bar_stacked():
    return tstoolbox.plot(type='bar_stacked',
                          input_ts=df,
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_barh_stacked():
    return tstoolbox.plot(type='barh_stacked',
                          input_ts=df,
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_histogram():
    return tstoolbox.plot(type='histogram',
                          clean=True,
                          input_ts='tests/02234500_65_65.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_heatmap():
    return tstoolbox.plot(type='heatmap',
                          columns=2,
                          clean=True,
                          input_ts=df,
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_norm_xaxis():
    return tstoolbox.plot(type='norm_xaxis',
                          columns=2,
                          clean=True,
                          input_ts='tests/02234500_65_65.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_norm_yaxis():
    return tstoolbox.plot(type='norm_yaxis',
                          columns=2,
                          clean=True,
                          input_ts='tests/02234500_65_65.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_lognorm_xaxis():
    return tstoolbox.plot(type='lognorm_xaxis',
                          columns=2,
                          clean=True,
                          input_ts='tests/02234500_65_65.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_lognorm_yaxis():
    return tstoolbox.plot(type='lognorm_yaxis',
                          columns=2,
                          clean=True,
                          input_ts='tests/02234500_65_65.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_weibull_xaxis():
    return tstoolbox.plot(type='weibull_xaxis',
                          columns=2,
                          clean=True,
                          input_ts='tests/02234500_65_65.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_weibull_yaxis():
    return tstoolbox.plot(type='weibull_yaxis',
                          columns=2,
                          clean=True,
                          input_ts='tests/02234500_65_65.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_kde_time():
    return tstoolbox.plot(type='kde_time',
                          columns=2,
                          clean=True,
                          input_ts='tests/02234500_65_65.csv',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_kde_time_multiple_traces():
    ndf = tstoolbox.read(['tests/daily.csv',
                          'tests/02325000_flow.csv'])
    return tstoolbox.plot(type='kde_time',
                          columns=[2, 3],
                          clean=True,
                          input_ts=ndf,
                          ytitle='Flow',
                          ofilename=None)

@pytest.mark.mpl_image_compare
def test_autocorrelation():
    return tstoolbox.plot(type='autocorrelation',
                          columns=2,
                          input_ts=df,
                          ofilename=None)
