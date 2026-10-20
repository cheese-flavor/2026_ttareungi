"""Check timing and data-quality failures independently of the real dataset."""
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.ml.calendar_features import add_holiday
from src.ml.features import FEATURES, make_features, validate_daily
from src.ml.predict import forecast
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


class ForecastTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.daily = pd.DataFrame({"date": pd.date_range("2025-01-01", "2026-06-30"),
                                   "rentals": rng.integers(10_000, 150_000, 546)})
        frame = make_features(self.daily)
        self.model = LinearRegression().fit(frame[FEATURES], frame.rentals)

    def test_forecast_uses_only_targets_through_d_minus_2(self):
        date = pd.Timestamp("2026-03-01")
        result = forecast(self.daily, date, self.model)
        expected = make_features(self.daily).loc[lambda f: f.date == date, FEATURES].reset_index(drop=True)
        pd.testing.assert_frame_equal(result[FEATURES], expected)
        changed = self.daily.copy()
        changed.loc[changed.date >= date - pd.Timedelta(days=1), "rentals"] += 1_000_000
        self.assertEqual(forecast(changed, date, self.model).prediction.iloc[0], result.prediction.iloc[0])

    def test_forecast_after_data_end(self):
        self.assertEqual(len(forecast(self.daily, "2026-07-02", self.model)), 1)
        with self.assertRaises(ValueError):
            forecast(self.daily, "2026-07-03", self.model)

    def test_holiday_flag_uses_only_the_calendar(self):
        frame = add_holiday(make_features(self.daily))
        flags = frame.set_index("date").holiday
        for day, expected in (("2025-10-06", 1), ("2025-10-08", 1), ("2025-10-10", 0),
                              ("2025-05-01", 0), ("2026-05-01", 1), ("2026-06-03", 1)):
            self.assertEqual(flags[pd.Timestamp(day)], expected, day)
        changed = add_holiday(make_features(self.daily.assign(rentals=self.daily.rentals * 3)))
        pd.testing.assert_series_equal(changed.holiday, frame.holiday)

    def test_forecast_needs_28_days_of_history(self):
        history = self.daily.iloc[:28]
        target = history.date.max() + pd.Timedelta(days=2)
        self.assertEqual(len(forecast(history, target, self.model)), 1)
        with self.assertRaises(ValueError):
            forecast(history.iloc[1:], target, self.model)


if __name__ == "__main__":
    unittest.main()
