# 따릉이 이용량 × 날씨 데이터 분석 프로젝트

## 5회차 머신러닝 후속 작업

- [머신러닝 실행 방법·평가 조건](docs/ml/README.md)

- [요구사항 1~8 및 완료 기준](docs/ml/REQUIREMENTS.md)
- [데이터 준비 기록](docs/ml/DATA.md)
- 공개 따릉이 CSV 복구: `python scripts/download_ml_data.py` (Python 3.9+, 데이터는 GitHub에서 제외)

요구사항 1~6번은 코드·실행 결과·개조식 해석을 작성했습니다. 7~8번은 후속 작업입니다.
새 과제 저장소: [GitLab 05-pjt](https://lab.ssafy.com/totori0710/05-pjt).
아래 내용과 루트 PLAN/HANDOFF는 기존 4회차 분석 기록입니다.

서울시 공공자전거(따릉이) 일별 대여건수와 기상청 ASOS 일별 기상 데이터를 결합해
날씨가 자전거 이용량에 미치는 영향을 분석한다.

- 요구사항/실행계획: [PLAN.md](PLAN.md)
- **F405~F406 인수인계 문서: [HANDOFF.md](HANDOFF.md)** (F401~F404 완료 상태에서 이어받는 사람용)
- 데이터 수집 기록(F402): [docs/data_collection_log.md](docs/data_collection_log.md)
- 협업 방식(Git): [docs/collaboration_guide.md](docs/collaboration_guide.md)
- 원본 데이터와 명세서 PDF는 이 공개 저장소에 포함하지 않습니다.

## 폴더 구조
```
data/raw/        원본 데이터 (수정 금지)
data/processed/  정제·병합된 데이터
notebooks/       분석 ipynb
docs/            기록·가이드 문서
```

## 실행 방법
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/execute_notebook.py
```
원본 CSV(`data/raw/`)만 있으면 `notebooks/analysis.ipynb`를 처음부터 끝까지 실행해
`data/processed/`의 중간·최종 산출물(`merged_daily.csv` 등)이 재생성된다.


## F405·F406 검토 안내

- F405: 인사이트, 관찰/해석 구분, 한계와 후속 수요 예측 문제 작성 완료.
- F406: 날씨 회귀와 달력 변수를 추가한 회귀를 시간순으로 비교 완료.
- [검토 안내](docs/review_f405_f406.md)에서 결과와 검토 지점을 확인합니다.
- 결과 표·그림은 `outputs/f406/`에 있으며 실행 결과와 해석은 노트북에도 포함됩니다.
- 검토 대기 상태입니다. master 병합·최종 제출 전 담당자 검토가 필요합니다.
- GitLab에는 데이터를 포함하고 PDF는 제외합니다. 추후 GitHub 반영 시에는 데이터와 PDF를 모두 제외합니다.


## GitHub 공개본 실행 전 준비

GitLab master의 F401~F406 코드·결과를 반영한 공개본입니다. 원본/전처리 데이터와 명세서 PDF, 일별 실제값을 포함한 예측 CSV는 제외했습니다. 노트북의 요약 통계·그래프·실행 결과는 유지합니다.

재실행하려면 `docs/data_collection_log.md`의 출처와 기간에 맞는 원본 CSV 4개를 별도로 확보해 `data/raw/`에 배치해야 합니다. 파일명은 노트북 F401에서 확인할 수 있습니다. F403에서 `data/processed/merged_daily.csv`가 생성됩니다. 데이터 없이 저장된 노트북 결과와 해석을 열람하는 것은 가능합니다.

F405·F406은 GitLab master에 병합되었습니다. 검토 문서의 '검토 대기' 문구는 병합 전 작업 기록입니다.
