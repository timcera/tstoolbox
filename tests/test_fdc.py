# -*- coding: utf-8 -*-

from unittest import TestCase

from tstoolbox import tstoolbox

from . import capture


class TestFDC(TestCase):
    def linebyline(self, out, teststr):
        for test1, test2 in zip(out.decode().split("\n"), teststr.split("\n")):
            if not test1:
                continue
            if "Exceed" in test1:
                self.assertEqual(test1, test2)
                continue
            test1_words = test1.split(",")
            test2_words = test2.split(",")
            test1_words = [float(i) for i in test1_words]
            test2_words = [float(i) for i in test2_words]
            for t1, t2 in zip(test1_words, test2_words):
                self.assertAlmostEqual(t1, t2)

    def test_flat_norm(self):
        """Test linear ramp CLI calculation of the FDC."""
        out = capture.capture(
            tstoolbox.calculate_fdc, input_ts="tests/data_flat_01.csv"
        )
        teststr = """Exceedance, Value, Exceedance_Label
-1.78615556126, 2, 0.037037037037
-1.44610359292, 2, 0.0740740740741
-1.22064034885, 2, 0.111111111111
-1.04440879487, 2, 0.148148148148
-0.895779818884, 2, 0.185185185185
-0.764709673786, 2, 0.222222222222
-0.645630749276, 2, 0.259259259259
-0.535082815086, 2, 0.296296296296
-0.430727299295, 2, 0.333333333333
-0.330872571726, 2, 0.37037037037
-0.234219193915, 2, 0.407407407407
-0.139710298882, 2, 0.444444444444
-0.0464357247705, 2, 0.481481481481
0.0464357247705, 2, 0.518518518519
0.139710298882, 2, 0.555555555556
0.234219193915, 2, 0.592592592593
0.330872571726, 2, 0.62962962963
0.430727299295, 2, 0.666666666667
0.535082815086, 2, 0.703703703704
0.645630749276, 2, 0.740740740741
0.764709673786, 2, 0.777777777778
0.895779818884, 2, 0.814814814815
1.04440879487, 2, 0.851851851852
1.22064034885, 2, 0.888888888889
1.44610359292, 2, 0.925925925926
1.78615556126, 2, 0.962962962963
"""
        self.linebyline(out, teststr)

    def test_flat_linear(self):
        """Test FDC API with linear plotting position."""
        out = capture.capture(
            tstoolbox.calculate_fdc,
            plotting_position="california",
            input_ts="tests/data_flat_01.csv",
        )
        teststr = """Exceedance, Value, Exceedance_Label
0.037037037037, 2, 0.037037037037
0.0740740740741, 2, 0.0740740740741
0.111111111111, 2, 0.111111111111
0.148148148148, 2, 0.148148148148
0.185185185185, 2, 0.185185185185
0.222222222222, 2, 0.222222222222
0.259259259259, 2, 0.259259259259
0.296296296296, 2, 0.296296296296
0.333333333333, 2, 0.333333333333
0.37037037037, 2, 0.37037037037
0.407407407407, 2, 0.407407407407
0.444444444444, 2, 0.444444444444
0.481481481481, 2, 0.481481481481
0.518518518519, 2, 0.518518518519
0.555555555556, 2, 0.555555555556
0.592592592593, 2, 0.592592592593
0.62962962963, 2, 0.62962962963
0.666666666667, 2, 0.666666666667
0.703703703704, 2, 0.703703703704
0.740740740741, 2, 0.740740740741
0.777777777778, 2, 0.777777777778
0.814814814815, 2, 0.814814814815
0.851851851852, 2, 0.851851851852
0.888888888889, 2, 0.888888888889
0.925925925926, 2, 0.925925925926
0.962962962963, 2, 0.962962962963
"""
        self.linebyline(out, teststr)

    def test_sunspot(self):
        """Test normal plotting position FDC API."""
        out = capture.capture(
            tstoolbox.calculate_fdc,
            plotting_position="weibull",
            input_ts="tests/data_sunspot.csv",
        )
        fp = open("tests/sunspot_area_fdc_compare.txt", "r")
        teststr = "".join(fp.readlines())
        self.linebyline(out, teststr)
