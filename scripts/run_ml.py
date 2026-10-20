"""Build and execute requirements 1..8, adding short result-based Markdown notes."""
import argparse
import json
import sys
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ml"
# 4회차 선행 분석 결과(docs/ml/EDA_SUMMARY.md)
PRIOR_EDA = {"max_temp_corr": 0.7188, "rain_gap_pct": -27.52}

SETUP = '''from pathlib import Path
import sys, json, time, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "requirements.txt").exists():
    ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from src.ml.features import FEATURES, load_daily, make_features
from src.ml.splits import split_data, time_cv
from src.ml.models import model_candidates
from src.ml.evaluation import metrics, compare, grouped_metrics
from src.ml.tuning import tune_models
from src.ml.predict import forecast
from sklearn.base import clone
from sklearn.inspection import permutation_importance

plt.rcParams.update({"figure.figsize": (10, 4), "font.family": "Malgun Gothic", "axes.unicode_minus": False})
OUT = ROOT / "outputs/ml"
OUT.mkdir(parents=True, exist_ok=True)
LOCAL = ROOT / "data/processed"
LOCAL.mkdir(parents=True, exist_ok=True)
daily = load_daily(ROOT)
frame = make_features(daily)
train, validation, final_train, test = split_data(frame)
'''

SECTIONS = {
    1: ("문제 정의", [
        ('''contract = pd.DataFrame([
    ["예측 대상", "서울시 전체의 D일 총 대여건수 (회귀)"],
    ["예측 시점", "D-1일 00:00"],
    ["과거 정보", "D-2일까지의 확정 대여건수, D일 달력"],
    ["활용", "도시 전체 운영 규모 계획의 참고 지표"],
    ["데이터 범위", f"{daily.date.min():%Y-%m-%d}~{daily.date.max():%Y-%m-%d}, {len(daily)}일"],
    ["제약", "반기 공개 자료를 이용한 시뮬레이션, 실제 운영에는 일 집계 공급 필요"],
], columns=["항목", "내용"])
display(contract)
''', "problem")]),
    2: ("전처리·피처 구성 및 누수 점검", [
        ('''quality = pd.DataFrame([{
    "raw_rows": len(daily), "feature_rows": len(frame), "warmup_excluded": len(daily) - len(frame),
    "duplicate_dates": int(daily.date.duplicated().sum()),
    "missing_days": int(len(pd.date_range(daily.date.min(), daily.date.max())) - len(daily)),
    "negative_targets": int((daily.rentals < 0).sum()),
}])
quality.to_csv(OUT / "02_quality.csv", index=False)
display(quality)
display(pd.DataFrame({"feature": FEATURES, "available": ["D-2", "D-7", "D-14", "D-8~D-2", "D-29~D-2"] + ["D일 달력"] * 5}))
changed = daily.copy()
cutoff = pd.Timestamp("2025-08-01")
changed.loc[changed.date >= cutoff - pd.Timedelta(days=1), "rentals"] += 1_000_000
altered = make_features(changed)
pd.testing.assert_frame_equal(frame.loc[frame.date <= cutoff, FEATURES], altered.loc[altered.date <= cutoff, FEATURES])
print("누수 점검 통과: D-1 이후의 대여건수 변경이 D일 피처에 영향을 주지 않음")
''', "quality")]),
    3: ("데이터 분할 및 기준선", [
        ('''split_summary = pd.DataFrame([
    {"split": name, "start": f"{part.date.min():%Y-%m-%d}", "end": f"{part.date.max():%Y-%m-%d}", "rows": len(part)}
    for name, part in [("train", train), ("validation", validation), ("final_train", final_train), ("test", test)]
])
split_summary.to_csv(OUT / "03_splits.csv", index=False)
display(split_summary)
baselines = {"train_mean": np.full(len(validation), train.rentals.mean()), "lag_7": validation.lag_7.to_numpy()}
baseline_metrics = compare(validation, baselines)
baseline_metrics.to_csv(OUT / "03_baselines.csv")
display(baseline_metrics.round(4))
''', "baseline")]),
    4: ("모델 학습 및 성능 비교", [
        ('''predictions = {"train_mean": np.full(len(validation), train.rentals.mean()), "lag_7": validation.lag_7.to_numpy()}
elapsed = {}
for name, model in model_candidates().items():
    started = time.perf_counter()
    model.fit(train[FEATURES], train.rentals)
    elapsed[name] = time.perf_counter() - started
    predictions[name] = model.predict(validation[FEATURES])
comparison = compare(validation, predictions)
comparison["fit_seconds"] = pd.Series(elapsed)
comparison.to_csv(OUT / "04_models.csv")
display(comparison.round(4))
comparison.MAE.plot.bar(title="검증 구간 모델별 MAE", ylabel="건/일", rot=0)
plt.tight_layout()
plt.savefig(OUT / "04_comparison.png", dpi=120)
plt.show()
''', "models")]),
    5: ("교차검증 및 하이퍼파라미터 조정", [
        ('''searches, cv_results, settings, selected = tune_models(train[FEATURES], train.rentals)
cv_results.to_csv(OUT / "05_cv_candidates.csv", index=False)
settings.to_csv(OUT / "05_cv_settings.csv", index=False)
folds = pd.DataFrame([{
    "fold": i + 1, "train_end": f"{train.iloc[fit].date.max():%Y-%m-%d}",
    "validation_start": f"{train.iloc[val].date.min():%Y-%m-%d}", "validation_end": f"{train.iloc[val].date.max():%Y-%m-%d}",
    "train_rows": len(fit), "validation_rows": len(val)
} for i, (fit, val) in enumerate(time_cv().split(train))])
folds.to_csv(OUT / "05_cv_folds.csv", index=False)
display(folds)
display(cv_results.round(3))
display(settings.round(3))
''', "cv"),
        ('''tuned_validation = compare(validation, {name: search.predict(validation[FEATURES]) for name, search in searches.items()})
tuned_validation.to_csv(OUT / "05_validation.csv")
chosen = searches[selected]
selection = {"model": selected, "params": chosen.best_params_, "cv_MAE": -chosen.best_score_,
             "selection_rule": "minimum mean CV MAE on training period; validation is diagnostic only",
             "features": FEATURES, "random_state": 42, "prediction_time": "D-1 00:00; targets through D-2",
             "test_previously_seen_in_EDA": True}
(OUT / "05_selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
joblib.dump(chosen.best_estimator_, LOCAL / "ml_selected_pipeline.joblib")
display(tuned_validation.round(4))
display(selection)
''', "selection")]),
    6: ("최종 평가 및 결과 해석", [
        ('''selection = json.loads((OUT / "05_selection.json").read_text(encoding="utf-8"))
model = clone(joblib.load(LOCAL / "ml_selected_pipeline.joblib"))
model.fit(final_train[FEATURES], final_train.rentals)
prediction = model.predict(test[FEATURES])
joblib.dump(model, LOCAL / "ml_final_pipeline.joblib")
scores = compare(test, {"train_mean": np.full(len(test), final_train.rentals.mean()),
                        "lag_7": test.lag_7.to_numpy(), selection["model"]: prediction})
scores.to_csv(OUT / "06_test_metrics.csv")
display(scores.round(4))
detail = test[["date", "rentals"]].copy()
detail["prediction"] = prediction
detail["residual"] = detail.rentals - prediction
detail["absolute_error"] = detail.residual.abs()
detail["month"] = detail.date.dt.strftime("%Y-%m")
detail["weekday"] = detail.date.dt.dayofweek
q1, q2 = final_train.rentals.quantile([1/3, 2/3]).to_numpy()
detail["demand_band"] = pd.cut(detail.rentals, [-np.inf, q1, q2, np.inf], labels=["low", "middle", "high"])
detail.to_csv(LOCAL / "ml_test_predictions.csv", index=False)
for column in ("month", "weekday", "demand_band"):
    grouped = grouped_metrics(detail, column)
    grouped.to_csv(OUT / f"06_{column}.csv")
    display(grouped.round(4))
fig, axes = plt.subplots(2, 1, figsize=(11, 7))
axes[0].plot(detail.date, detail.rentals, label="실제")
axes[0].plot(detail.date, detail.prediction, label="예측")
axes[0].legend()
axes[0].set_title("일별 대여건수: 실제와 예측")
axes[1].scatter(detail.prediction, detail.residual, s=10)
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set(xlabel="예측 대여건수", ylabel="실제 - 예측", title="잔차 분석")
plt.tight_layout()
plt.savefig(OUT / "06_predictions_residuals.png", dpi=120)
plt.show()
''', "test"),
        ('''# Interpretation only: no model selection or tuning uses these test diagnostics.
importance = permutation_importance(model, test[FEATURES], test.rentals,
    scoring="neg_mean_absolute_error", n_repeats=10, random_state=42, n_jobs=1)
importance_table = pd.DataFrame({"feature": FEATURES, "MAE_increase": importance.importances_mean,
                                 "std": importance.importances_std}).sort_values("MAE_increase", ascending=False)
importance_table.to_csv(OUT / "06_importance.csv", index=False)
display(importance_table.round(3))
importance_table.set_index("feature").MAE_increase.plot.bar(title="입력 순열에 따른 MAE 증가", ylabel="건/일", rot=45)
plt.tight_layout()
plt.savefig(OUT / "06_importance.png", dpi=120)
plt.show()
worst = detail.nlargest(5, "absolute_error")[["date", "rentals", "prediction", "residual"]]
display(worst.round({"rentals": 1, "prediction": 1, "residual": 1}))
# Public notebooks show only a few error examples; full daily records remain untracked.
(LOCAL / "ml_error_examples.json").write_text(worst.to_json(orient="records", date_format="iso"), encoding="utf-8")
''', "importance"),
        ('''# 4회차 F406 결과(docs/ml/EDA_SUMMARY.md). 당일 관측 날씨를 사용한 사후 실험으로 입력 조건이 다르다.
prior = pd.DataFrame([
    {"model": "4회차 weather Ridge", "MAE": 16212.48164033676, "RMSE": 20634.831132942145, "WAPE": 0.1632160101143025},
    {"model": "4회차 weather_calendar Ridge", "MAE": 12382.940625485498, "RMSE": 16629.422335266554, "WAPE": 0.1246628497234842},
]).assign(same_day_weather=True)
current = scores[["MAE", "RMSE", "WAPE"]].rename(index=lambda name: f"5회차 {name}").reset_index().assign(same_day_weather=False)
eda_models = pd.concat([current, prior], ignore_index=True).set_index("model")
eda_models.to_csv(OUT / "06_eda_models.csv")
display(eda_models.round(4))
half = daily.assign(year=daily.date.dt.year, month=daily.date.dt.month).query("month <= 6")
eda_months = half.pivot_table(index="month", columns="year", values="rentals", aggfunc="mean")
eda_months.columns = [f"mean_{year}" for year in eda_months.columns]
eda_months["yoy_change_pct"] = (eda_months.mean_2026 / eda_months.mean_2025 - 1) * 100
month_errors = grouped_metrics(detail, "month")
month_errors.index = month_errors.index.str[-2:].astype(int)
eda_months = eda_months.join(month_errors[["MAE", "WAPE"]])
eda_months.to_csv(OUT / "06_eda_months.csv")
display(eda_months.round(2))
''', "eda")]),
    7: ("모델 활용 방안 및 한계", [
        ('''model = joblib.load(LOCAL / "ml_final_pipeline.joblib")
selection = json.loads((OUT / "05_selection.json").read_text(encoding="utf-8"))
mae = pd.read_csv(OUT / "06_test_metrics.csv", index_col=0).loc[selection["model"], "MAE"]
saved = pd.read_csv(LOCAL / "ml_test_predictions.csv", parse_dates=["date"])
# Typical case: the test day whose absolute error is closest to the median.
typical = saved.iloc[(saved.absolute_error - saved.absolute_error.median()).abs().argmin()]
example = forecast(daily, typical.date, model)
assert np.isclose(example.prediction.iloc[0], typical.prediction), "forecast() must match the 6번 evaluation path"
example["actual"] = typical.rentals
example["error"] = example.actual - example.prediction
display(example.T)
''', "example"),
        ('''future = pd.concat([forecast(daily, date, model) for date in ["2026-07-01", "2026-07-02"]], ignore_index=True)
future["lower"] = (future.prediction - mae).clip(lower=0)
future["upper"] = future.prediction + mae
display(future.T)
try:
    forecast(daily, "2026-07-03", model)
    unavailable = ""
except ValueError as error:
    unavailable = str(error)
    print(f"2026-07-03 예측 불가: {unavailable}")
examples = pd.concat([example.assign(kind="test"), future.assign(kind="future")], ignore_index=True)
examples.insert(1, "weekday", examples.date.dt.day_name())
examples.to_csv(OUT / "07_examples.csv", index=False)
context = {"model": selection["model"], "test_MAE": mae, "data_end": f"{daily.date.max():%Y-%m-%d}",
           "july_2025_mean": daily.loc[daily.date.dt.strftime("%Y-%m") == "2025-07", "rentals"].mean(),
           "unavailable_2026_07_03": unavailable}
(OUT / "07_context.json").write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
display(context)
''', "future"),
        ('''months = pd.read_csv(OUT / "06_eda_months.csv", index_col=0)
bands = pd.read_csv(OUT / "06_demand_band.csv", index_col=0)
usage = pd.DataFrame([
    ["입력", "D-2까지 확정된 일별 총 대여건수(최소 28일)와 D일 날짜"],
    ["출력", f"D일 서울시 전체 총 대여건수 1개 값, 테스트 MAE ±{mae:,.0f}건을 참고 범위로 함께 제시"],
    ["활용", "전일 기준 정비·재배치 인력과 차량 규모 계획, 수요 급증·급감 예상일 사전 점검"],
    ["적용 범위", "서울시 전체 일 단위 총량. 대여소·시간대별 배치 결정에는 적용 불가"],
    ["한계", f"날씨·공휴일·운영 변화 미반영, 월별 MAE 최대 {months.MAE.max():,.0f}건, 저수요 구간 WAPE {bands.loc['low', 'WAPE']:.1%}"],
    ["한계", "약 1년 학습, 반기 공개자료라 실시간 갱신 불가, 테스트 기간을 선행 EDA에서 이미 관찰"],
    ["개선", "일 집계 자동 공급, 공휴일 달력, 전일 발행 기상예보, 대여소별 수요·재고 자료, 정기 재학습"],
], columns=["항목", "내용"])
usage.to_csv(OUT / "07_usage.csv", index=False)
display(usage)
''', "usage")]),
    8: ("추가 실험: 공휴일 피처", [
        ('''from src.ml.calendar_features import HOLIDAY_FEATURES, add_holiday, holiday_table

holiday_frame = add_holiday(frame)
train8, validation8, final_train8, test8 = split_data(holiday_frame)
listing = holiday_table(daily.date.min(), daily.date.max())
listing["weekday"] = listing.date.dt.day_name()
listing.to_csv(OUT / "08_holidays.csv", index=False)
counts = pd.DataFrame([{"split": name, "rows": len(part), "holidays": int(part.holiday.sum()),
                        "weekday_holidays": int(((part.holiday == 1) & (part.weekend == 0)).sum())}
                       for name, part in [("train", train8), ("validation", validation8),
                                          ("final_train", final_train8), ("test", test8)]])
counts.to_csv(OUT / "08_holiday_counts.csv", index=False)
display(listing)
display(counts)
''', "holidays"),
        ('''# Pre-registered rule: adopt the holiday feature only if the selected model's mean CV MAE on the training period improves.
experiments, rows = {}, []
for name, columns in [("base", FEATURES), ("holiday", HOLIDAY_FEATURES)]:
    searches, _, _, selected = tune_models(train8[columns], train8.rentals)
    experiments[name] = {"columns": columns, "search": searches[selected], "model": selected}
    for model_name, search in searches.items():
        rows.append({"features": name, "model": model_name, "selected": model_name == selected,
                     "params": str(search.best_params_), "cv_MAE": -search.best_score_,
                     "cv_std": search.cv_results_["std_test_score"][search.best_index_]})
cv8 = pd.DataFrame(rows)
cv8.to_csv(OUT / "08_cv.csv", index=False)
display(cv8.round(3))
base_cv, holiday_cv = (-experiments[name]["search"].best_score_ for name in ("base", "holiday"))
decision = {"rule": "adopt holiday feature if selected-model mean CV MAE decreases; test is report-only",
            "base_model": experiments["base"]["model"], "base_cv_MAE": base_cv,
            "holiday_model": experiments["holiday"]["model"], "holiday_cv_MAE": holiday_cv,
            "cv_change_pct": (holiday_cv / base_cv - 1) * 100, "adopt_holiday": bool(holiday_cv < base_cv)}
display(decision)
''', "cv8"),
        ('''# Report-only: test results never change the decision above.
predictions, models8 = {"lag_7": test8.lag_7.to_numpy()}, {}
for name, experiment in experiments.items():
    model8 = clone(experiment["search"].best_estimator_).fit(final_train8[experiment["columns"]], final_train8.rentals)
    models8[name] = model8
    predictions[name] = model8.predict(test8[experiment["columns"]])
test_scores8 = compare(test8, predictions)
test_scores8.to_csv(OUT / "08_test.csv")
display(test_scores8.round(4))
day_type = np.select([(test8.holiday == 1) & (test8.weekend == 0), test8.weekend == 1], ["holiday", "weekend"], "workday")
day_rows = []
for name in experiments:
    detail8 = test8[["date", "rentals"]].assign(prediction=predictions[name], day_type=day_type)
    day_rows.append(grouped_metrics(detail8, "day_type").assign(features=name).reset_index())
by_day = pd.concat(day_rows, ignore_index=True)
by_day.to_csv(OUT / "08_day_type.csv", index=False)
display(by_day.round(4))
by_day.pivot(index="group", columns="features", values="MAE").loc[["workday", "weekend", "holiday"]].plot.bar(
    title="테스트 일 유형별 MAE: 기존 vs 공휴일 피처", ylabel="건/일", rot=0)
plt.tight_layout()
plt.savefig(OUT / "08_day_type.png", dpi=120)
plt.show()
importance8 = permutation_importance(models8["holiday"], test8[HOLIDAY_FEATURES], test8.rentals,
    scoring="neg_mean_absolute_error", n_repeats=10, random_state=42, n_jobs=1)
decision["holiday_importance_MAE"] = float(importance8.importances_mean[HOLIDAY_FEATURES.index("holiday")])
decision["test_MAE"] = {name: float(test_scores8.loc[name, "MAE"]) for name in experiments}
(OUT / "08_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
display(decision)
''', "test8")]),
}


def setup_for(step):
    code = SETUP
    if step == 1:
        start = code.index("from src.ml.features")
        end = code.index("plt.rcParams")
        code = code[:start] + code[end:]
        code = code[:code.index("daily = load_daily(ROOT)")] + '''paths = sorted((ROOT / "data/raw").glob("서울특별시 공공자전거 일별 대여건수_*.csv"))
daily = pd.concat([pd.read_csv(path, encoding="cp949") for path in paths], ignore_index=True)
daily = daily.rename(columns={"대여일자": "date", "대여건수": "rentals"})
daily["date"] = pd.to_datetime(daily.date)
'''
        return code
    if step < 3:
        code = code.replace("from src.ml.splits import split_data, time_cv\n", "")
        code = code.replace("train, validation, final_train, test = split_data(frame)\n", "")
    if step < 4:
        code = code.replace("from src.ml.models import model_candidates\n", "")
    if step < 3:
        code = code.replace("from src.ml.evaluation import metrics, compare, grouped_metrics\n", "")
    if step < 5:
        code = code.replace("from src.ml.tuning import tune_models\n", "")
    if step < 7:
        code = code.replace("from src.ml.predict import forecast\n", "")
    return code


def notes(key):
    def table(name):
        return pd.read_csv(OUT / name, index_col=0)
    if key == "problem":
        return "- 기존 분석의 계절·요일 패턴을 일별 수요 예측 문제로 연결함.\n- 달력과 과거 대여건수를 사용하며 당일 관측 날씨는 입력에서 제외함.\n- 도시 전체 총량 예측으로 개별 대여소 수요를 직접 판단하기 어려움.\n- 반기 공개자료의 운영 시뮬레이션이며 실서비스에는 일 집계 공급이 필요함."
    if key == "quality":
        row = pd.read_csv(OUT / "02_quality.csv").iloc[0]
        return (f"- 원본 {int(row.raw_rows)}일에서 이동통계 준비 구간 {int(row.warmup_excluded)}일을 제외해 {int(row.feature_rows)}일을 구성함.\n"
                "- 중복·누락 날짜와 음수 타깃이 없으며 날짜를 건너뛴 지연 피처 생성을 방지함.\n"
                "- D-1 이후 값 변경에도 D일 피처가 동일해 설계한 정보 시점 조건을 충족함.")
    if key == "baseline":
        scores = table("03_baselines.csv")
        winner = scores.MAE.idxmin()
        return f"- 검증 MAE는 학습 평균 {scores.loc['train_mean', 'MAE']:,.0f}건, 전주 동일 요일 {scores.loc['lag_7', 'MAE']:,.0f}건임.\n- 두 기준선 중 {winner}의 검증 MAE가 더 낮음.\n- 학습 끝과 검증 시작 사이 하루를 제외해 타깃 가용 시점을 맞춤.\n- 테스트 점수는 아직 계산하지 않음."
    if key == "models":
        scores = table("04_models.csv")
        winner = scores.MAE.idxmin()
        return f"- 동일한 검증 날짜에서 {winner}의 MAE가 {scores.loc[winner, 'MAE']:,.0f}건으로 가장 낮음.\n- Ridge는 선형 관계, Random Forest는 비선형·상호작용을 비교하기 위해 선정함.\n- 이 결과는 고정 기본 설정의 비교이며 최종 모델은 다음 단계의 시간순 CV로 선정함."
    if key == "cv":
        scores = pd.read_csv(OUT / "05_cv_candidates.csv")
        row = scores.loc[scores.mean_MAE.idxmin()]
        return f"- 시간순 3개 fold에서 가장 낮은 평균 MAE는 {row['model']}의 {row.mean_MAE:,.0f}건임.\n- 해당 설정의 fold 표준편차는 {row.std_MAE:,.0f}건으로 평가 시점에 따른 변동이 존재함.\n- fold마다 전처리를 다시 학습하고 하루 gap을 두어 미래 타깃 유입을 방지함."
    if key == "selection":
        choice = json.loads((OUT / "05_selection.json").read_text(encoding="utf-8"))
        scores = table("05_validation.csv")
        return f"- 학습 구간의 평균 CV MAE를 기준으로 {choice['model']}을 선정하고 설정을 고정함.\n- 선정 모델의 고정 검증 MAE는 {scores.loc[choice['model'], 'MAE']:,.0f}건임.\n- 검증 점수는 추가 진단이며 테스트 결과로 설정을 변경하지 않음."
    if key == "test":
        choice = json.loads((OUT / "05_selection.json").read_text(encoding="utf-8"))
        scores = table("06_test_metrics.csv")
        row = scores.loc[choice['model']]
        improvement = (1 - row.MAE / scores.loc['lag_7', 'MAE']) * 100
        direction = "감소" if improvement >= 0 else "증가"
        months = table("06_month.csv")
        return (f"- 선정 모델의 테스트 MAE {row.MAE:,.0f}건, RMSE {row.RMSE:,.0f}건, WAPE {row.WAPE:.2%}임.\n"
                f"- 전주 동일 요일 기준선 대비 MAE가 {abs(improvement):.2f}% {direction}함.\n"
                f"- 월별 MAE가 가장 큰 시기는 {months.MAE.idxmax()}이며 계절별 오차가 균일하지 않음.\n"
                "- 테스트는 기존 EDA에서 관찰한 2026년 상반기로, 완전히 새로운 기간의 성능을 보장하지 않음.")
    if key == "importance":
        imp = pd.read_csv(OUT / "06_importance.csv").set_index("feature")
        season = imp.loc[["year_sin", "year_cos"], "MAE_increase"].sum()
        weekday = imp.loc[["dow_sin", "dow_cos", "weekend"], "MAE_increase"].sum()
        return (f"- 순열 중요도에서 {imp.index[0]}의 평균 MAE 증가가 {imp.MAE_increase.iloc[0]:,.0f}건으로 가장 큼.\n"
                f"- 연중 주기 피처의 MAE 증가 합은 {season:,.0f}건으로, 4회차 EDA의 뚜렷한 계절 패턴과 같은 방향임.\n"
                f"- 4회차에서 최고기온 상관이 {PRIOR_EDA['max_temp_corr']:.2f}였으며, 날씨 입력이 없어 연중 주기가 기온 변화를 대신 반영했을 가능성이 있음.\n"
                f"- 요일 피처의 MAE 증가 합은 {weekday:,.0f}건으로 작으며, 요일 패턴을 lag_7·이동평균이 흡수했을 가능성이 있음.\n"
                "- 중요도는 모델의 입력 의존도이며 수요에 대한 인과효과를 의미하지 않음.\n"
                "- 지연값·이동평균은 서로 연관되어 개별 중요도가 분산될 수 있음.\n"
                "- 큰 오류의 원인을 날씨·공휴일로 단정할 수 없으며 후속 데이터로 확인해야 함.")
    if key == "eda":
        choice = json.loads((OUT / "05_selection.json").read_text(encoding="utf-8"))
        models = table("06_eda_models.csv")
        months = table("06_eda_months.csv")
        bands = table("06_demand_band.csv")
        ours, prior = models.loc[f"5회차 {choice['model']}"], models.loc["4회차 weather_calendar Ridge"]
        worst = months.MAE.idxmax()
        return (f"- 선정 모델의 테스트 MAE {ours.MAE:,.0f}건은 같은 기간 4회차 날씨+달력 회귀의 {prior.MAE:,.0f}건보다 큼.\n"
                "- 4회차 모델은 당일 관측 날씨를 사용한 사후 실험으로 예측 시점 조건이 달라 직접 비교할 수 없음.\n"
                "- 두 결과의 차이는 날씨 정보가 줄 수 있는 개선 폭의 상한에 가까우며 실제 효과는 기상예보로 검증해야 함.\n"
                f"- MAE가 가장 큰 {worst}월은 일평균 {months.loc[worst, 'mean_2026']:,.0f}건, 전년 동월 대비 {months.loc[worst, 'yoy_change_pct']:+.2f}%로 EDA의 봄~초여름 수요 상승 구간에 해당함.\n"
                f"- 저수요 구간 WAPE가 {bands.loc['low', 'WAPE']:.1%}로 가장 높음. 4회차에서 강수일 평균이 {abs(PRIOR_EDA['rain_gap_pct']):.2f}% 낮았으므로 강수일이 포함됐을 가능성이 있으나 강수 자료가 없어 확인하지 않음.")
    if key == "example":
        row = pd.read_csv(OUT / "07_examples.csv").query("kind == 'test'").iloc[0]
        return (f"- {row.date} 입력 10개로 {row.prediction:,.0f}건을 예측했으며 실제 {row.actual:,.0f}건, 오차 {row.error:+,.0f}건임.\n"
                "- 테스트 절대오차가 중앙값에 가장 가까운 날을 골라 전형적인 사례로 제시함.\n"
                "- 활용 함수의 예측이 6번 평가의 저장 예측과 같아 운영 입력과 평가 입력이 동일함을 확인함.\n"
                "- 입력은 D-2까지의 대여건수와 D일 달력뿐이며 D-1 이후 값은 사용하지 않음.")
    if key == "future":
        rows = pd.read_csv(OUT / "07_examples.csv").query("kind == 'future'")
        context = json.loads((OUT / "07_context.json").read_text(encoding="utf-8"))
        lines = [f"- {r.date}({r.weekday}) 예측 {r.prediction:,.0f}건, 참고 범위 {r.lower:,.0f}~{r.upper:,.0f}건임." for r in rows.itertuples()]
        gap = (rows.prediction.mean() / context["july_2025_mean"] - 1) * 100
        return "\n".join(lines + [
            "- 6번에서 고정한 최종 모델(2025-12-30까지 학습)을 재학습 없이 그대로 사용함.",
            f"- 예측 평균이 2025년 7월 일평균 {context['july_2025_mean']:,.0f}건보다 {gap:+.1f}% 차이 남. 6월 말의 높은 수준이 지연값에 반영된 결과로 보이며, 장마 등 7월 날씨 변화는 입력에 없음.",
            "- 참고 범위는 테스트 MAE를 더하고 뺀 값으로 확률적 신뢰구간이 아님.",
            f"- 자료가 {context['data_end']}에 끝나 2026-07-03부터는 D-2 실제값이 없어 예측을 거부함.",
            "- 반기 공개자료로는 매일 예측을 갱신할 수 없어 실제 운영에는 일 집계 공급이 필요함."])
    if key == "usage":
        models = table("06_eda_models.csv")
        context = json.loads((OUT / "07_context.json").read_text(encoding="utf-8"))
        ours, prior = models.loc[f"5회차 {context['model']}"], models.loc["4회차 weather_calendar Ridge"]
        return ("- 매일 D-1 00:00에 다음 날 서울시 전체 수요 1개 값을 출력해 운영 규모 계획의 참고 지표로 사용함.\n"
                "- 예측값은 오차 범위와 함께 제시하며 단독 의사결정 근거로 사용하지 않음.\n"
                "- 개별 대여소 재배치·시간대 운영에는 대여소별 수요·재고 자료와 별도 모델이 필요함.\n"
                f"- 관측 날씨를 쓴 4회차 사후 실험의 MAE {prior.MAE:,.0f}건과 현재 {ours.MAE:,.0f}건의 차이로 볼 때 기상예보 결합이 우선 개선 과제임.\n"
                "- 공휴일은 사전에 알 수 있어 누수 없이 추가 가능하므로 8번 추가 실험 후보로 둠.")
    if key == "holidays":
        counts = pd.read_csv(OUT / "08_holiday_counts.csv").set_index("split")
        return (f"- 대체·임시·선거 공휴일을 포함한 공식 공휴일을 holiday 피처로 추가함(holidays 패키지 KR).\n"
                f"- 평일 공휴일은 학습 {counts.loc['train', 'weekday_holidays']}일, 테스트 {counts.loc['test', 'weekday_holidays']}일로 표본이 매우 적음.\n"
                "- 공휴일은 사전에 공표되어 D-1 00:00에 사용할 수 있으며 대여건수와 무관하게 생성됨.\n"
                "- 노동절은 2026년부터 공휴일이며 2025-05-01은 공휴일이 아니어서 두 해의 정의가 다름.")
    if key == "cv8":
        result = pd.read_csv(OUT / "08_cv.csv")
        base = result.query("features == 'base' and selected").iloc[0]
        holiday = result.query("features == 'holiday' and selected").iloc[0]
        change = (holiday.cv_MAE / base.cv_MAE - 1) * 100
        verdict = "채택함" if holiday.cv_MAE < base.cv_MAE else "채택하지 않음"
        ridge = result.query("model == 'ridge'").set_index("features").cv_MAE
        return (f"- 기존 피처의 선정 모델 {base.model} CV MAE {base.cv_MAE:,.0f}건은 5번 결과와 같은 조건임.\n"
                f"- Ridge는 공휴일 추가로 CV MAE가 {(ridge['holiday'] / ridge['base'] - 1) * 100:+.2f}% 변해 선형 모델은 공휴일 신호를 일부 활용한 것으로 보임.\n"
                "- 얕은 Random Forest(max_depth=3, min_samples_leaf=7)는 적은 공휴일 표본으로 분기를 만들기 어려웠을 가능성이 있음.\n"
                f"- 공휴일 추가 후 선정 모델 {holiday.model}의 CV MAE는 {holiday.cv_MAE:,.0f}건으로 {change:+.2f}% 변함.\n"
                f"- 사전에 정한 CV 기준에 따라 공휴일 피처를 {verdict}.\n"
                f"- fold 표준편차 {holiday.cv_std:,.0f}건 대비 차이 {abs(holiday.cv_MAE - base.cv_MAE):,.0f}건으로 변동 범위를 함께 고려해야 함.")
    if key == "test8":
        decision = json.loads((OUT / "08_decision.json").read_text(encoding="utf-8"))
        by_day = pd.read_csv(OUT / "08_day_type.csv").set_index(["features", "group"])
        base_mae, holiday_mae = decision["test_MAE"]["base"], decision["test_MAE"]["holiday"]
        hb, hh = by_day.loc[("base", "holiday")], by_day.loc[("holiday", "holiday")]
        return (f"- 테스트 MAE는 기존 {base_mae:,.0f}건, 공휴일 추가 {holiday_mae:,.0f}건으로 {(holiday_mae / base_mae - 1) * 100:+.2f}% 변함.\n"
                f"- 평일 공휴일 {int(hh.n)}일의 MAE는 {hb.MAE:,.0f}건에서 {hh.MAE:,.0f}건으로 변함.\n"
                f"- 테스트 순열 중요도에서 holiday의 MAE 증가는 {decision['holiday_importance_MAE']:,.0f}건임.\n"
                "- 테스트 결과는 보고용이며 채택 여부와 설정 선택에 사용하지 않음.\n"
                "- 평일 공휴일 표본이 적어 개선·악화를 일반화하기 어렵고 연휴 전후·공휴일 다음 주 lag_7 왜곡은 별도 피처로 다루지 않음.")
    raise KeyError(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", default=list(range(1, 9)), choices=range(1, 9))
    args = parser.parse_args()
    folder = ROOT / "notebooks/ml"
    folder.mkdir(parents=True, exist_ok=True)
    for step in args.steps:
        title, sections = SECTIONS[step]
        nb = nbformat.v4.new_notebook(metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
        nb.cells = [nbformat.v4.new_markdown_cell(f"# {step}. {title}\n\n서울시 따릉이 일별 수요 예측. [요구사항](../../docs/ml/REQUIREMENTS.md)."),
                    nbformat.v4.new_code_cell(setup_for(step))]
        for code, key in sections:
            nb.cells.append(nbformat.v4.new_code_cell(code))
            nb.cells.append(nbformat.v4.new_markdown_cell("실행 후 결과 해석 생성", metadata={"result_note": key}))
        km = KernelManager(kernel_name="python3")
        km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        NotebookClient(nb, km=km, timeout=600, resources={"metadata": {"path": str(ROOT)}}).execute()
        for cell in nb.cells:
            if "result_note" in cell.metadata:
                cell.source = notes(cell.metadata["result_note"])
        nbformat.validate(nb)
        nbformat.write(nb, folder / f"{step:02d}.ipynb")
        print(f"{step}: executed and saved", flush=True)


if __name__ == "__main__":
    main()
