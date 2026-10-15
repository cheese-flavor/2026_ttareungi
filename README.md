# 따릉이 이용량 × 날씨 데이터 분석 프로젝트

서울시 공공자전거(따릉이) 일별 대여건수와 기상청 ASOS 일별 기상 데이터를 결합해
날씨가 자전거 이용량에 미치는 영향을 분석한다.

- 요구사항/실행계획: [PLAN.md](PLAN.md)
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
pip install pandas matplotlib seaborn jupyter
jupyter notebook notebooks/analysis.ipynb
```
원본 CSV(`data/raw/`)만 있으면 `notebooks/analysis.ipynb`를 처음부터 끝까지 실행해
`data/processed/`의 중간·최종 산출물(`merged_daily.csv` 등)이 재생성된다.
