# 따릉이 일별 수요 예측

서울시 전체의 다음 날 총 대여건수를 달력 정보와 과거 대여건수로 예측한다.
기존 데이터 분석에서 확인한 계절·요일 패턴을 머신러닝으로 확장한다.
선행 분석: [GitHub 기존 분석 기준본](https://github.com/cheese-flavor/2026_ttareungi/tree/3beaa189148826459b8a6534c5dbee283f2ab8ea).

## 예측 조건

- D일 총 대여건수를 D-1일 00:00에 예측한다.
- 일별 관측 대여건수는 D-2일까지 확보됐다고 가정한다.
- 반기 공개자료로 과거 운영을 재현한 실험이며, 실제 운영에는 내부 일별 집계 공급이 필요하다.
- 당일 관측 날씨는 입력으로 사용하지 않는다.
- 매 예측 시점에 가용한 과거 실제값을 갱신하는 평가이며, 여러 달을 한 번에 예측하는 방식이 아니다.

## 실행

저장소 루트에서 Python 3.11 기준으로 실행한다.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/download_ml_data.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/run_ml.py
```

일부 단계만 재실행하려면 `python scripts/run_ml.py --steps 4 5 6`처럼 지정한다.
5번은 선정 모델을 로컬에 저장하므로 6번 최초 실행 전에 5번을 수행해야 한다. 7번은 6번이 저장한 최종 모델과 테스트 예측을 사용한다.
실행기는 각 노트북을 새 커널에서 실행하고 실제 결과에 근거한 짧은 개조식 해석을 추가한다.

## 산출물

- `notebooks/ml/01.ipynb`~`07.ipynb`: 문제 정의, 전처리, 기준선, 모델 비교, CV, 최종 평가, 활용 방안.
- `src/ml/`: 재사용 가능한 피처·분할·모델·튜닝·평가·예측(`predict.forecast`) 코드.
- `outputs/ml/`: 집계 성능, CV 실험 기록, 그래프, 모델 선정 설정.
- `data/processed/`: 로컬 전처리 자료, 모델과 일별 예측 상세. Git 추적에서 제외한다.
- 8번 선택 확장은 후속 범위이며 아직 구현하지 않았다.

## 모델 사용 예

```python
import joblib
from pathlib import Path
from src.ml.features import load_daily
from src.ml.predict import forecast

root = Path(".")
model = joblib.load(root / "data/processed/ml_final_pipeline.joblib")
forecast(load_daily(root), "2026-07-01", model)  # 입력 피처 10개와 prediction 1행
```

D일을 예측하려면 D-2까지의 일별 대여건수가 연속으로 최소 28일 필요하다. 조건을 충족하지 않으면 `ValueError`가 발생한다.

## 평가와 한계

학습은 2025-01-30~09-29, 검증은 2025-10-01~12-31이다.
학습 내부 시간순 CV는 3분할·검증 45일·gap 1일로 수행한다.
학습 평균·전주 동일 요일 기준선과 Ridge·Random Forest를 비교하며 CV 평균 MAE로 모델을 선정한다.
최종 재학습은 2025-12-30까지, 테스트는 2026-01-01~06-30이다.
테스트 진단을 모델 선택·설정 변경에 사용하지 않는다.

2026년 상반기는 선행 EDA에서 이미 관찰한 기간이다. 완전히 새로운 외부 평가로 해석하지 않는다.
약 1년의 학습자료와 상반기 테스트로 연중 일반화를 입증할 수 없으며, 날씨·공휴일·운영 변화가 입력에 없다.
순열 중요도는 모델 의존도를 나타내며 인과관계의 증거가 아니다.
대여소별 자료가 없어 개별 대여소 배치에 직접 적용할 수 없다.

데이터 출처·복구 기록: [DATA.md](DATA.md). 요구사항: [REQUIREMENTS.md](REQUIREMENTS.md).
실제 실행 결과: [RESULTS.md](RESULTS.md).
