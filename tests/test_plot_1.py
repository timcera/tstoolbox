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
