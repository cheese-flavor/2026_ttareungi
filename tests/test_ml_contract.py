"""Check timing and data-quality failures independently of the real dataset."""
import unittest

import numpy as np
import pandas as pd

from src.ml.features import FEATURES, make_features, validate_daily
from src.ml.splits import split_data, time_cv


class TimingContractTests(unittest.TestCase):
    def setUp(self):
        self.daily = pd.DataFrame({"date": pd.date_range("2025-01-01", "2026-06-30"),
                                   "rentals": np.arange(546) + 100})

    def test_unavailable_values_cannot_change_features(self):
        original = make_features(self.daily)
        changed = self.daily.copy()
        date = pd.Timestamp("2025-06-01")
        changed.loc[changed.date >= date - pd.Timedelta(days=1), "rentals"] += 1_000_000
        altered = make_features(changed)
        pd.testing.assert_frame_equal(original.loc[original.date <= date, FEATURES],
                                      altered.loc[altered.date <= date, FEATURES])
        row = original.loc[original.date == date].iloc[0]
        self.assertEqual(row.lag_2, self.daily.loc[self.daily.date == date - pd.Timedelta(days=2), "rentals"].iloc[0])
        expected = self.daily.loc[self.daily.date.between(date - pd.Timedelta(days=8), date - pd.Timedelta(days=2)), "rentals"].mean()
        self.assertEqual(row.rolling_7, expected)

    def test_missing_duplicate_and_negative_fail(self):
        for broken in (self.daily.drop(index=100), pd.concat([self.daily, self.daily.iloc[-1:]]),
                       self.daily.assign(rentals=-1)):
            with self.assertRaises(ValueError):
                validate_daily(broken)

    def test_cv_target_availability(self):
        train, validation, final_train, test = split_data(make_features(self.daily))
        self.assertEqual(len(validation), 92)
        self.assertEqual(len(test), 181)
        self.assertLessEqual(final_train.date.max(), test.date.min() - pd.Timedelta(days=2))
        for fit, holdout in time_cv().split(train):
            self.assertLessEqual(train.iloc[fit].date.max(), train.iloc[holdout].date.min() - pd.Timedelta(days=2))


if __name__ == "__main__":
    unittest.main()
