# -*- coding: utf-8 -*-

from unittest import TestCase

from tstoolbox import tstoolbox


class Testgof(TestCase):
    def setUp(self):
        self.df = tstoolbox.read(["tests/data_sunspot.csv", "tests/data_sunspot.csv"])
        self.gof = [
            ["Mean error or bias", 0.0],
            ["Root-mean-square Deviation/Error (RMSD)", 0.0],
            ["Centered RMSD (CRMSD)", 0.0],
            ["Pearson coefficient of correlation (r)", 1.0],
            ["Skill score (Murphy)", 1.0],
            ["Nash-Sutcliffe Efficiency", 1.0],
            ["Brier's Score", 0.0],
            ["Common count observed and simulated", 10],
            ["Count of NaNs", None, 0, 0],
        ]

    def test_gof(self):
        """Test of gof API."""
        out = tstoolbox.gof(
            obs_col=self.df.iloc[10:20, 0],
            sim_col=self.df.iloc[10:20, 1],
            stats=["me", "rmsd", "crmsd", "corrcoef", "murphyss", "nse", "brierss"],
        )
        assert out == self.gof
