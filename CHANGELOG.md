## v106.0.0 (2023-02-17)

### Refactor

- finally fixed the cltoolbox functions showing up in descendent projects

## 105.0.0 (2023-02-05)

### Refactor

- removed unused utility code

## 104.0.4 (2023-01-16)

### Fix

- removed skipna and append="columns" that actually weren't doing anything

## 104.0.3 (2023-01-16)

### Refactor

- refactor with refurb and pylint

## 104.0.2 (2023-01-08)

## 104.0.1 (2022-12-16)

### Fix

- **fill.py**: force type to be float so that linear (and other mathematical methods) work in fill

## 104.0.0 (2022-10-18)

### Fix

- fixed type hint for plot
- another try to get plot tests to pass on github
- plot tests are failing on github that I can't reproduce locally
- **plot**: reinstated plot tests which are almost identical to plottoolbox tests but use tstoolbox.plot which uses plottoolbox in the background
- moved to the breaking change move in toolbox_utils from typic to pydantic and misc. modernizations

## 103.18.7 (2022-09-28)

## 103.18.6 (2022-09-28)

### Refactor

- moved from tstoolbox.tsutils to toolbox_utils.tsutils and finished pyproject.toml

## 103.18.5 (2022-08-10)

### Fix

- fixed skiprows option along with updates to read documentation

## 103.18.4 (2022-07-18)

### Refactor

- cltoolbox rather than mando and plottoolbox.* instead of "tstoolbox plot ..."

## 103.18.3 (2022-06-05)

### Fix

- missed that had to import pint_pandas even though not used directly

## 103.18.2 (2022-06-04)

### Fix

- reversed start of using numpy.typing because needs a too recent numpy

### Refactor

- changed .format() string to f strings

## 103.18.1 (2022-05-30)

### Fix

- allow for a period Datetime index in printiso

## 103.18.0 (2022-03-16)

### Feat

- **calculate_fdc**: added recurrence (return) interval

## 103.17.1 (2022-02-14)

### Fix

- **_date_slice**: fixed some situations where needed tz_localize rather than tz_convert

## 103.17.0 (2022-02-13)

### Feat

- copy docs using decorator instead of simple assignment

## 103.16.1 (2022-02-07)

### Fix

- shifted to subprocess because os.system wasn't working

## 103.16.0 (2022-02-06)

### Feat

- **input_ts**: changed gof to not need input_ts instead can specify obs_col and sim_col
- **gof**: shifted to using HydroErr library

### Fix

- **rolling_window.py**: window keyword was transformed to wrong type if default
- **replace.py**: support for None

### Refactor

- **clip.py**: use pandas clip

## 103.15.2 (2021-10-10)

### Fix

- typing versions greater than 2.6.4 adds a subclass test that made many tests fail

## 103.15.1 (2021-10-09)

### Refactor

- minor refactors

## 103.15.0 (2021-09-14)

### Fix

- minor fixes reverting changes from autofix

## None (2021-09-14)

## v103.15.0 (2021-09-12)

### Fix

- **replace.py**: for type checking needed additional types
- **replace.py**: None needed to be included in type for typical
- pip_requirements.txt to reduce vulnerabilities

### Feat

- add upload of coverage results to coveralls

## vpor (2021-08-15)

## v103.14.10 (2021-08-15)

### Fix

- fixed the way `start_date` and `end_date` were handled by adding working `por` keyword
- **tsutils.py**: print_iso changed to correctly print out no spaces table formats *_nos
- **plot.py**: plottoolbox needed additional keywords to correctly plot time-series

## v103.14.9 (2021-08-01)

## v103.14.8 (2021-07-30)

### Fix

- plot input_ts, cli values

## v103.13.8 (2021-07-28)

### Fix

- units, Float64, flexible input

## v103.12.8 (2021-07-25)

### Fix

- Correctly include units.
- Correctly including units in name.

## v103.11.8 (2021-07-21)

### Fix

- Can work with "A" and "M" frequencies.
- remove pint units for calculations.

## v103.10.8 (2021-07-20)

### Fix

- Correct column names to allow for empty units.
- Added "--multi-line 3" to isort

### Refactor

- rearrange to allow "import tstoolbox"
- Reorganized to allow for "import tstoolbox"

## v103.9.8 (2021-07-07)

### Fix

- work to add units to column headers

## v103.8.8 (2021-06-29)

### Refactor

- .pre-commit-config.yaml

### Feat

- Added the beginnings of a forecast feature
style:  Added several features to .pre-commit-config.yaml
- Returned the ability to use comma separated file names.

### Fix


## v103.6.8 (2021-05-25)

### Feat

- Added the capability of using the index (usually datetimeindex) as
a `x_train_cols` in the regression.
fix: Now detect is DataFrame or Series and skip further processing in
'read_iso_ts'
- Added "min_count" keyword to `aggregate`
docs: Misc.
build: Added "extra_requires" to setup.py for development

## v103.5.8 (2021-05-07)

### Feat

- New read from xlsx, wdm files.
- rework new multiple sources read feature.
- Continue type hint tests.

### Refactor

- read_iso_ts is now part of common_kwds.

## v102.5.8 (2021-03-06)

### Fix

- Minor fixes.
- Docs, fixes because of changed memory_optimize
- memory_optimize is less aggressive
- read --columns filters output.

### Feat

- "Typical" to coerce args/kwargs.
- added butterworth filter

## v101.5.8 (2020-10-21)

### Fix

- Fixed months_across_years.

## v101.4.8 (2020-09-04)

### Fix

- Matched Paul Tor's color order.
- Needed matplotlib styles in MANIFEST.in.

## v101.4.7 (2020-09-04)

### Fix

- Incorporated SciencePlots mpl styles.

## v101.4.6 (2020-09-04)

### Fix

- Pandas implemented new tolerance approach.
- Better working with time zones.

### Feat

- Lossless compression of dataframes.
- Added plot styles, new default is "bright".

## v100.4.6 (2020-07-08)

### Fix

- Correctly uses multiple input to 'regression'

## v100.4.5 (2020-07-03)

### Feat

- Added regression sub-command.
- Added 'output_names' keyword to rename columns on output.
- Added hatch patterns to bar plots.

### Fix

- Allow for index,y to by used for "xy" type.

## v100.3.4 (2020-06-24)

### Feat

- Fix x,y plots to allow for index,y plotting
- Added autocorrelation for each column.

## v100.3.3 (2020-06-10)

### Fix

- Corrected columns selection when 0
- xy plot incorrectly inserted a column.
- Allow fit to work across multiple columns.

### Feat

- Added autocorrelation if lags==0.

## v100.2.0 (2020-06-04)

### Refactor

- Just added some vertical space.
- Improved description of r statistic.

### Feat

- Added scikit-learn scaling functions.
- Added method keyword.
- lowess and linear fit
- Added coefficient of determination to gof.

### Fix

- join_axes is deprecated.
- Added force_freq throughout.

## v100.1.0 (2020-04-25)

### Feat

- Added "groupby='all' to aggregate.

## v47.97.50.35 (2020-04-01)

## v46.96.49.34 (2020-03-06)

## v45.94.49.34 (2020-03-04)

## v43.94.47.34 (2020-02-20)

## v43.91.45.33 (2019-11-21)

## v43.89.43.31 (2019-11-01)

## v40.87.42.28 (2019-09-18)

## v40.86.42.28 (2019-09-17)

## v36.86.39.27 (2019-08-28)

## v35.86.39.27 (2019-06-21)

## v35.85.39.27 (2019-06-21)

## v34.82.39.27 (2019-05-30)
