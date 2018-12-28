import matplotlib
matplotlib.use('Agg')

from matplotlib.testing.decorators import image_comparison

from tstoolbox import tstoolbox

# Pull this in once.
df = tstoolbox.aggregate(agg_interval='D',
                         clean=True,
                         input_ts='tests/02234500_65_65.csv')

@image_comparison(baseline_images=['time_plot'],
                  tol=0.019, extensions=['png'])
def test_time_plot():
    tstoolbox.plot(type='time',
                   columns=1,
                   clean=True,
                   input_ts='tests/02234500_65_65.csv')

@image_comparison(baseline_images=['time_multiple_traces_plot'],
                  tol=0.019, extensions=['png'])
def test_time_multiple_traces_plot():
    tstoolbox.plot(type='time',
                   columns=[2,3],
                   style='b-,r*',
                   input_ts='tests/data_daily_sample.csv')

@image_comparison(baseline_images=['time_multiple_traces_style_plot'],
                  tol=0.019, extensions=['png'])
def test_time_multiple_traces_style_plot():
    tstoolbox.plot(type='time',
                   columns=[2,3],
                   style='b-,r  ',
                   input_ts='tests/data_daily_sample.csv')

@image_comparison(baseline_images=['time_multiple_traces_new_style_plot'],
                  tol=0.019, extensions=['png'])
def test_time_multiple_traces_new_style_plot():
    tstoolbox.plot(type='time',
                   columns=[2,3],
                   markerstyles=' ,*',
                   linestyles='-, ',
                   input_ts='tests/data_daily_sample.csv')

@image_comparison(baseline_images=['time_markers_plot'],
                  tol=0.019, extensions=['png'])
def test_time_markers():
    tstoolbox.plot(type='time',
                   columns=[2, 3],
                   linestyles=' ',
                   markerstyles='auto',
                   input_ts='tests/data_daily_sample.csv')

@image_comparison(baseline_images=['xy_plot'],
                  tol=0.019, extensions=['png'])
def test_xy():
    tstoolbox.plot(type='xy',
                   clean=True,
                   input_ts='tests/02234500_65_65.csv')

@image_comparison(baseline_images=['xy_multiple_traces_plot'],
                  tol=0.019, extensions=['png'])
def test_xy_multiple_traces():
    tstoolbox.plot(type='xy',
                   columns=[2,3,3,2],
                   input_ts='tests/data_daily_sample.csv')

@image_comparison(baseline_images=['xy_multiple_traces_logy_plot'],
                  tol=0.019, extensions=['png'])
def test_xy_multiple_traces_logy():
    tstoolbox.plot(type='xy',
                   columns=[2,3,3,2],
                   yaxis='log',
                   input_ts='tests/data_daily_sample.csv')

@image_comparison(baseline_images=['xy_multiple_traces_logx_plot'],
                  tol=0.019, extensions=['png'])
def test_xy_multiple_traces_logx():
    tstoolbox.plot(type='xy',
                   columns=[2,3,3,2],
                   xaxis='log',
                   input_ts='tests/data_daily_sample.csv')

@image_comparison(baseline_images=['xy_multiple_traces_markers_plot'],
                  tol=0.019, extensions=['png'])
def test_xy_multiple_traces_markers():
    tstoolbox.plot(type='xy',
                   columns=[2,3,3,2],
                   linestyles=' ',
                   markerstyles='auto',
                   input_ts='tests/data_daily_sample.csv')

@image_comparison(baseline_images=['double_mass_plot'],
                  tol=0.019, extensions=['png'])
def test_double_mass():
    tstoolbox.plot(type='double_mass',
                   clean=True,
                   input_ts='tests/02234500_65_65.csv')

@image_comparison(baseline_images=['double_mass_mult_plot'],
                  tol=0.019, extensions=['png'])
def test_double_mass_mult():
    tstoolbox.plot(type='double_mass',
                   columns=[2,3,3,2],
                   input_ts='tests/data_daily_sample.csv')

@image_comparison(baseline_images=['double_mass_marker_plot'],
                  tol=0.019, extensions=['png'])
def test_double_mass_marker():
    tstoolbox.plot(type='double_mass',
                   columns=[2, 3, 3, 2],
                   linestyles=' ',
                   markerstyles='auto',
                   input_ts='tests/data_daily_sample.csv')

@image_comparison(baseline_images=['boxplot'],
                  tol=0.019, extensions=['png'])
def test_boxplot():
    ndf = tstoolbox.read(['tests/02234500_65_65.csv',
                          'tests/02325000_flow.csv'],
                         clean=True,
                         append='combine')
    tstoolbox.plot(input_ts=ndf,
                   clean=True,
                   columns=[2, 3],
                   type='boxplot')

@image_comparison(baseline_images=['scatter_matrix'],
                  tol=0.019, extensions=['png'])
def test_scatter_matrix():
    tstoolbox.plot(type='scatter_matrix',
                   clean=True,
                   input_ts='tests/02234500_65_65.csv')

@image_comparison(baseline_images=['lag_plot'],
                  tol=0.019, extensions=['png'])
def test_lag_plot():
    tstoolbox.plot(columns=1,
                   type='lag_plot',
                   input_ts=df)

# Can't have a bootstrap test since random selections are made.
# @image_comparison(baseline_images=['bootstrap'],
#                   tol=0.019, extensions=['png'])
# def test_bootstrap():
#     tstoolbox.plot(type='bootstrap',
#                    clean=True,
#                    columns=2,
#                    input_ts='tests/02234500_65_65.csv')

@image_comparison(baseline_images=['probability_density'],
                  tol=0.019, extensions=['png'])
def test_probability_density():
    tstoolbox.plot(type='probability_density',
                   clean=True,
                   input_ts='tests/02234500_65_65.csv')

@image_comparison(baseline_images=['bar'],
                  tol=0.019, extensions=['png'])
def test_bar():
    tstoolbox.plot(type='bar', input_ts=df)

@image_comparison(baseline_images=['barh'],
                  tol=0.019, extensions=['png'])
def test_barh():
    tstoolbox.plot(type='barh', input_ts=df)

@image_comparison(baseline_images=['bar_stacked'],
                  tol=0.019, extensions=['png'])
def test_bar_stacked():
    tstoolbox.plot(type='bar_stacked', input_ts=df)

@image_comparison(baseline_images=['barh_stacked'],
                  tol=0.019, extensions=['png'])
def test_barh_stacked():
    tstoolbox.plot(type='barh_stacked', input_ts=df)

@image_comparison(baseline_images=['histogram'],
                  tol=0.019, extensions=['png'])
def test_histogram():
    tstoolbox.plot(type='histogram',
                   clean=True,
                   input_ts='tests/02234500_65_65.csv')

@image_comparison(baseline_images=['heatmap'],
                  tol=0.019, extensions=['png'])
def test_heatmap():
    tstoolbox.plot(type='heatmap',
                   columns=2,
                   clean=True,
                   input_ts=df)

@image_comparison(baseline_images=['norm_xaxis'],
                  tol=0.019, extensions=['png'])
def test_norm_xaxis():
    tstoolbox.plot(type='norm_xaxis',
                   columns=2,
                   clean=True,
                   input_ts='tests/02234500_65_65.csv')

@image_comparison(baseline_images=['norm_yaxis'],
                  tol=0.019, extensions=['png'])
def test_norm_yaxis():
    tstoolbox.plot(type='norm_yaxis',
                   columns=2,
                   clean=True,
                   input_ts='tests/02234500_65_65.csv')

@image_comparison(baseline_images=['lognorm_xaxis'],
                  tol=0.019, extensions=['png'])
def test_lognorm_xaxis():
    tstoolbox.plot(type='lognorm_xaxis',
                   columns=2,
                   clean=True,
                   input_ts='tests/02234500_65_65.csv')

@image_comparison(baseline_images=['lognorm_yaxis'],
                  tol=0.019, extensions=['png'])
def test_lognorm_yaxis():
    tstoolbox.plot(type='lognorm_yaxis',
                   columns=2,
                   clean=True,
                   input_ts='tests/02234500_65_65.csv')

@image_comparison(baseline_images=['weibull_xaxis'],
                  tol=0.019, extensions=['png'])
def test_weibull_xaxis():
    tstoolbox.plot(type='weibull_xaxis',
                   columns=2,
                   clean=True,
                   input_ts='tests/02234500_65_65.csv')

@image_comparison(baseline_images=['weibull_yaxis'],
                  tol=0.019, extensions=['png'])
def test_weibull_yaxis():
    tstoolbox.plot(type='weibull_yaxis',
                   columns=2,
                   clean=True,
                   input_ts='tests/02234500_65_65.csv')

@image_comparison(baseline_images=['kde_time'],
                  tol=0.019, extensions=['png'])
def test_kde_time():
    tstoolbox.plot(type='kde_time',
                   columns=2,
                   clean=True,
                   input_ts='tests/02234500_65_65.csv')

@image_comparison(baseline_images=['kde_time_multiple_traces'],
                  tol=0.019, extensions=['png'])
def test_kde_time_multiple_traces():
    ndf = tstoolbox.read(['tests/daily.csv',
                          'tests/02325000_flow.csv'])
    tstoolbox.plot(type='kde_time',
                   columns=[2, 3],
                   clean=True,
                   input_ts=ndf,
                   ytitle='Flow')

@image_comparison(baseline_images=['autocorrelation'],
                  tol=0.019, extensions=['png'])
def test_autocorrelation():
    tstoolbox.plot(type='autocorrelation',
                   columns=2,
                   input_ts=df)
