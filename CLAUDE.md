# CLAUDE.md

서울시 따릉이 **일별 총 대여건수 예측**(회귀) 프로젝트다. 기준 문서는 머신러닝 5회차 명세서(`머신러닝pjt_명세서.pdf`, 요구사항 F501~F508)다.
4회차 EDA는 별도 저장소에서 진행했으며, `docs/ml/README.md`에 GitHub 기준본 링크가 있다. 이 저장소는 그 결과를 바탕으로 한다.

- 명세서 PDF는 대외비다. `.gitignore`의 `*.pdf`로 제외되어 있으므로 커밋하거나 외부에 올리지 않는다.
- `docs/ml/REQUIREMENTS.md`의 1~8번은 명세서의 F501~F508에 대응한다.

## 진행 현황 (2026-10-20 기준, F501~F508 완료·README 포함 origin/main 60ba2f3 푸시)

| 명세 | 요구사항 | 상태 | 근거 / 남은 점 |
|---|---|---|---|
| F501 | 머신러닝 문제 정의 | ✅ 완료 | `notebooks/ml/01.ipynb`. D일 총 대여건수를 D-1 00:00에 예측 |
| F502 | 전처리·피처·누수 점검 | ✅ 완료 | `02.ipynb`, `src/ml/features.py`, `tests/test_ml_contract.py` |
| F503 | 분할·기준선 | ✅ 완료 | `03.ipynb`, `src/ml/splits.py`. 기준선은 학습 평균과 전주 동일 요일(lag_7) |
| F504 | 모델 2종 이상 비교 | ✅ 완료 | `04.ipynb`. Ridge와 Random Forest를 MAE·RMSE·WAPE·R2로 비교 |
| F505 | 교차검증·튜닝 | ✅ 완료 | `05.ipynb`, `outputs/ml/05_*`. TimeSeriesSplit은 3분할에 gap 1일 |
| F506 | 최종 평가·해석 | ✅ 완료 (47cce60) | `06.ipynb`. 테스트 MAE는 18,370으로 lag_7보다 12.72% 낮다. 월·요일·수요 구간별 오차, 순열 중요도, 잔차 그래프가 있다. EDA 연결은 `06_eda_*.csv`와 `notes("eda")`에 있다. |
| F507 | 모델 활용 방안·한계 | ✅ 완료 (9700f45) | `07.ipynb`, `src/ml/predict.py`, `outputs/ml/07_*`, 테스트 6개 |
| F508 | 추가 실험·모델 개선 | ✅ 완료 (2fadb1f) | `08.ipynb`, `src/ml/calendar_features.py`. 공휴일 피처는 CV +0.12%여서 미채택 |

로컬 상태: 0단계를 완료했다(2026-10-20). `.venv`, `data/raw`, `data/processed/*.joblib`이 준비되어 있고 노트북 1~6의 재현도 확인했다.
재실행하면 `fit_seconds`, `mean_fit_seconds`, 셀 id, 실행 시각만 달라진다. 이 차이만 있으면 해당 단계의 파일을 `git restore`로 되돌리고 커밋하지 않는다.

## 실행 계획

### 0단계. 환경 복구와 재현 확인
1. `python -m venv .venv`를 실행하고 `.venv\Scripts\python.exe -m pip install -r requirements.txt`로 패키지를 설치한다. Python 3.11.9는 확인됐다.
2. `.venv\Scripts\python.exe scripts/download_ml_data.py`로 원본 CSV 3개를 `data/raw/`에 받는다.
3. `.venv\Scripts\python.exe -m unittest discover -s tests -v`를 실행한다. 테스트 3개가 통과해야 한다.
4. `.venv\Scripts\python.exe scripts/run_ml.py --steps 5 6`을 실행해 `data/processed/*.joblib`을 다시 만든다.
5. `git diff outputs/ml`로 결과가 재현되는지 확인한다. 수치가 바뀌면 원인을 찾기 전까지 커밋하지 않는다.

### 1단계. F506 보완: EDA와의 연결
- ✅ 4회차 저장소를 `C:\Users\SSAFY\Desktop\2026_ttareungi`(커밋 `3beaa18`에 고정)로 받았다. 이 폴더는 참고용이므로 수정하거나 코드를 실행하지 않는다.
- ✅ 받은 내용을 `docs/ml/EDA_SUMMARY.md`로 요약했다. 명세서 준비사항의 "선행 분석 자료"에 해당한다.
  - 4회차 F406의 날씨+달력 Ridge는 테스트 MAE 12,383이지만 당일 관측 날씨를 쓴 사후 실험이다. 5회차 RF(18,370)와 직접 비교할 수 없고, 이 점을 해석에 명시한다.
- `run_ml.py`의 `notes("test")`와 `notes("importance")`에 EDA와 연결한 해석을 추가한다. 연결할 내용은 다음과 같다.
  - `year_cos`와 `rolling_7`의 중요도가 높은 것은 EDA의 계절성·추세와 맞는지 본다.
  - 요일 피처의 중요도가 낮다. EDA의 요일 차이를 lag_7이 흡수했을 가능성이 있다.
  - 4~5월에 MAE가 크다. EDA의 봄철 수요 급증이나 날씨 변동과 관련이 있는지 본다.
  - low 수요 구간의 WAPE가 34.5%다. 우천·한파일이 원인일 수 있으며, 날씨 입력이 없다는 한계와 연결한다.
- 커밋 메시지: `[6] link test errors and importance to prior EDA`

### 2단계. F507 모델 활용 방안 및 한계 (필수)
- `src/ml/predict.py`를 추가한다. `forecast(daily, target_date, model)`은 D-2까지의 값만 사용하는지 검증하고, 피처 행과 예측값을 반환한다. 과거값이 부족하면 `ValueError`를 낸다.
- `tests/test_ml_contract.py`에 테스트를 추가한다. D-1 이후 값을 바꿔도 예측이 변하지 않는지, 데이터가 부족하면 실패하는지 확인한다.
- `scripts/run_ml.py`의 `SECTIONS`에 7번을 추가하고 `notes()` 키를 만든다. `--steps` 범위를 1~7로 넓히고 `notebooks/ml/07.ipynb`를 생성한다.
  - 입출력 예시 ①: 테스트 기간 중 하루를 골라 피처 10개, 예측값, 실제값, 오차를 보여 준다.
  - 입출력 예시 ②: 데이터 마지막 날(2026-06-30) 기준으로 2026-07-01과 07-02를 실제로 예측한다. 07-03부터는 D-2 조건 때문에 예측할 수 없다는 점도 보여 준다.
  - 활용 방법: 도시 전체의 일 단위 운영 규모(정비 인력, 재배치 차량)를 계획하는 참고 지표로 쓴다. MAE를 예측 오차 범위로 함께 제시한다.
  - 적용 범위와 한계: 대여소 단위로는 쓸 수 없다. 데이터가 반기마다 공개되고, 날씨·공휴일·운영 변화가 반영되지 않는다. 학습 기간은 약 1년이고, 테스트 기간은 EDA에서 이미 본 구간이다. 월별 오차 편차도 있다.
  - 개선 사항: 일 집계 데이터 공급, 공휴일 달력, 기상 *예보* 반영(관측값을 쓰면 누수다), 대여소 단위 데이터, 재학습 주기를 제안한다.
- 결과는 `outputs/ml/07_*.csv`에 저장한다. `README.md`와 `docs/ml/RESULTS.md`, `docs/ml/REQUIREMENTS.md`를 갱신한다.
- 커밋 메시지: `[7] document model usage, input/output examples and limits`

### 3단계. F508 추가 실험 (선택, F507 완료 후)
- 우선 후보는 **공휴일 피처**다. 미리 알 수 있는 정보라 누수가 없다. 다음 후보는 D-2까지의 ASOS 기상 관측값이다. ASOS 데이터는 아직 받지 않았다(`docs/ml/DATA.md` 참고).
- 비교 조건은 기존과 같게 둔다. 같은 분할, 같은 `time_cv()`, 같은 CV 평균 MAE 선정 규칙을 쓴다. 테스트 성능은 보고만 하고 모델 선정에 쓰지 않는다.
- 확장 전후의 전체·월별·요일별 성능과 중요도 변화를 비교하고, 개선 여부를 판단한다. 결과는 `08.ipynb`와 `outputs/ml/08_*`에 둔다.
- 커밋 메시지는 `[8] ...` 형식이다.

### 4단계. 마무리
- `README.md`와 `docs/ml/*.md`의 "7~8번은 후속 작업" 문구를 정리한다. 요구사항 번호에 F5xx 표기를 함께 적는다.
- `run_ml.py` 전체를 새 커널에서 다시 실행하고, 노트북에 오류가 없는지 확인한다.
- Git 저장소 형태로 제출한다.

## 작업 규칙

- **노트북은 직접 편집하지 않는다.** `scripts/run_ml.py`의 `SECTIONS`(코드)와 `notes()`(해석)를 고친 뒤 실행해서 생성한다.
- 해석은 실제 출력 파일을 읽어 만든 짧은 개조식 한국어로 쓴다(`~함`, `~임`). 수치를 하드코딩하지 않는다. 관찰한 사실과 해석을 구분한다.
- 시간 규칙: D일 예측은 D-1 00:00에 하며, 대여건수는 D-2까지만 사용한다. 당일 관측 날씨는 쓰지 않는다. 새 피처를 추가할 때마다 누수 테스트도 추가한다.
- 테스트 결과를 모델 선택이나 설정 변경에 쓰지 않는다. 모델 선정 기준은 학습 구간의 시간순 CV 평균 MAE다.
- 재사용하는 로직은 `src/ml/`에 둔다. `random_state=42`와 `n_jobs=1`을 유지한다.
- `data/`(원본, 일별 예측 상세, 모델)는 Git에 올리지 않는다. 집계 결과만 `outputs/ml/`에 둔다.
- 인증키가 필요하면 `.env`나 환경변수로 관리한다.

## Git 커밋·푸시 방식

원격 저장소는 SSAFY GitLab이다: `origin = https://lab.ssafy.com/totori0710/05-pjt.git`. 기본 브랜치는 `main`이다.
지금까지의 작업 방식은 `feature/ml-1-6` 브랜치에서 작업한 뒤 `main`에 fast-forward로 병합한 것이다. 그래서 히스토리가 일직선이다.

**GitHub 공개본** (`github = https://github.com/cheese-flavor/2026_ttareungi.git`)
- 4회차 저장소에 5회차 작업을 함께 올린다. 커밋 히스토리는 GitLab과 별개라서 해시가 다르다.
- GitHub에 반영할 때는 `github/main`에서 브랜치를 만들고, 새 커밋을 cherry-pick한 뒤 PR로 병합한다.
- 루트 README는 GitLab과 다르다. 4회차 섹션을 유지해야 하므로 덮어쓰지 않는다.
- **기존 파일을 삭제하지 않는다.** 푸시 전에 `git diff --diff-filter=D github/main HEAD`로 삭제된 파일이 없는지 확인한다.
- 2026-10-20: `feature/ml-7-8`(ef4037a)을 푸시했다. PR은 아직 만들지 않았다.

**브랜치**
- 단계마다 `main`에서 기능 브랜치를 만든다. 이름은 `feature/ml-<번호>` 형식이다. 예: `feature/ml-6-eda`, `feature/ml-7`, `feature/ml-8`.
- `main`에 직접 커밋하지 않는다. 작업이 끝나면 `main`에 fast-forward로 병합하거나 GitLab MR로 병합한다. 히스토리는 일직선으로 유지한다.

**커밋**
- 요구사항 하나당 커밋 하나를 원칙으로 한다. 메시지 형식은 `[N] 영어 소문자 명령형 설명`이다.
  - 예: `[7] document model usage, input/output examples and limits`
  - 요구사항과 관계없는 작업에는 `chore: ...`, `docs: ...`를 쓴다. 예: `chore: initialize ML project ...`
- 하나의 커밋에 코드(`src/`, `scripts/`, `tests/`), 생성된 노트북(`notebooks/ml/NN.ipynb`), 집계 결과(`outputs/ml/NN_*`), 관련 문서(`docs/ml/*.md`, `README.md`)를 함께 넣는다.
- 커밋 전 확인 사항:
  1. `unittest`가 통과한다.
  2. 해당 단계를 `run_ml.py --steps N`으로 새 커널에서 실행했을 때 오류가 없다.
  3. `git status`에 `data/`, `.venv/`, `*.pdf`, `.env`가 없다. `.gitignore`로 제외되어 있지만 그래도 확인한다.
  4. 이전 단계의 `outputs/ml` 수치가 의도치 않게 바뀌지 않았다.
- 파일은 `git add <경로>`로 하나씩 지정한다. `git add -A`나 `git add .`는 쓰지 않는다.

**푸시**
```powershell
git switch main; git pull --ff-only origin main
git switch -c feature/ml-7
# ... 작업, 테스트, 실행 ...
git add src/ml/predict.py tests/test_ml_contract.py scripts/run_ml.py notebooks/ml/07.ipynb outputs/ml/07_* docs/ml README.md
git commit -m "[7] document model usage, input/output examples and limits"
git push -u origin feature/ml-7
# 병합: GitLab MR 또는 로컬 fast-forward
git switch main; git merge --ff-only feature/ml-7; git push origin main
```
- force push(`--force`)와 이미 푸시한 커밋의 amend나 rebase는 하지 않는다. 수정할 일이 생기면 새 커밋을 추가한다.
- Claude는 사용자가 요청할 때만 커밋하고 푸시한다. 푸시하기 전에 커밋 목록(`git log origin/main..HEAD --oneline`)을 보여 주고 확인을 받는다.

## 주요 명령

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v   # 계약 테스트
.venv\Scripts\python.exe scripts/run_ml.py                  # 전체 노트북 생성·실행
.venv\Scripts\python.exe scripts/run_ml.py --steps 6        # 일부만 실행 (6번은 5번에서 만든 모델 필요)
```
