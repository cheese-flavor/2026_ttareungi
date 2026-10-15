# 따릉이 이용량 × 날씨 데이터 분석 프로젝트

서울시 공공자전거(따릉이) 일별 대여건수와 기상청 ASOS 일별 기상 데이터를 결합해
날씨가 자전거 이용량에 미치는 영향을 분석한다.

- 요구사항/실행계획: [PLAN.md](PLAN.md)
- **F405~F406 인수인계 문서: [HANDOFF.md](HANDOFF.md)** (F401~F404 완료 상태에서 이어받는 사람용)
- 데이터 수집 기록(F402): [docs/data_collection_log.md](docs/data_collection_log.md)
- 협업 방식(Git): [docs/collaboration_guide.md](docs/collaboration_guide.md)
- 원본 명세서: `데이터분석_및_시각화PJT_명세서.pdf`

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
