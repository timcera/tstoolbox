import matplotlib
matplotlib.use('Agg')

import pytest

from tstoolbox import tstoolbox

# Pull this in once.
df = tstoolbox.aggregate(agg_interval='D',
                         clean=True,
                         input_ts='tests/02234500_65_65.csv')
# Pull this in once.
dfa = tstoolbox.aggregate(agg_interval='A',
                          clean=True,
                          input_ts='tests/02234500_65_65.csv')


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
