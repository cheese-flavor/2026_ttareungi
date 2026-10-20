"""Build and execute requirements 1..6, adding short result-based Markdown notes."""
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
display(worst.round(1))
# Public notebooks show only a few error examples; full daily records remain untracked.
(LOCAL / "ml_error_examples.json").write_text(worst.to_json(orient="records", date_format="iso"), encoding="utf-8")
''', "importance")]),
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
        imp = pd.read_csv(OUT / "06_importance.csv")
        return (f"- 순열 중요도에서 {imp.iloc[0].feature}의 평균 MAE 증가가 {imp.iloc[0].MAE_increase:,.0f}건으로 가장 큼.\n"
                "- 중요도는 모델의 입력 의존도이며 수요에 대한 인과효과를 의미하지 않음.\n"
                "- 지연값·이동평균은 서로 연관되어 개별 중요도가 분산될 수 있음.\n"
                "- 큰 오류의 원인을 날씨·공휴일로 단정할 수 없으며 후속 데이터로 확인해야 함.")
    raise KeyError(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", default=list(range(1, 7)), choices=range(1, 7))
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
