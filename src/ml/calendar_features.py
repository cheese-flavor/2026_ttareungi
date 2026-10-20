import holidays
import pandas as pd

from src.ml.features import FEATURES

HOLIDAY_FEATURES = FEATURES + ["holiday"]


def holiday_table(start, end):
    # Official public holidays, including substitute, temporary and election days.
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    calendar = holidays.KR(years=range(start.year, end.year + 1), language="ko")
    rows = [(pd.Timestamp(day), name) for day, name in calendar.items() if start <= pd.Timestamp(day) <= end]
    return pd.DataFrame(rows, columns=["date", "name"]).sort_values("date").reset_index(drop=True)


def add_holiday(frame):
    # Holidays are announced in advance, so D's flag is available at D-1 00:00.
    table = holiday_table(frame.date.min(), frame.date.max())
    return frame.assign(holiday=frame.date.isin(table.date).astype(int))
