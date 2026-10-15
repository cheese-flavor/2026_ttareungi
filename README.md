# 2026_ttareungi

서울시 따릉이 데이터 분석 및 시각화 프로젝트입니다.

## 역할

| 요구사항 | 담당 | 산출물 |
| --- | --- | --- |
| 분석 문제 및 데이터 선정 | EunahCho-kr | 질문, 분석 단위, 주요 변수, 선정 근거 |
| 데이터 수집 및 출처 기록 | EunahCho-kr | 수집 코드, 출처·수집 시점·기간·이용 조건 |
| 정제 및 데이터 품질 점검 | EunahCho-kr | 정제 코드, 데이터 사전, 품질 점검 및 인계 |
| 탐색적 분석 및 시각화 | cheese-flavor | 통계, 시각화, 그래프별 해석 |
| 인사이트·한계 및 후속 문제 | cheese-flavor | 관찰과 해석, 한계, 후속 머신러닝 문제 |
| 분석 확장 (선택) | cheese-flavor | 추가 데이터 결합 또는 방법 확장, 전후 비교 |

## 시작하기

Python 3.11 이상을 권장합니다. PowerShell 기준:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
jupyter lab
```

클라우드에서 합의한 데이터 버전을 내려받고 `.env`의 `DATA_DIR`을 해당 로컬 폴더로 설정합니다. 클라우드 공급자와 데이터셋·분석 기간은 아직 미정입니다. 인증정보는 공유하거나 커밋하지 않습니다.

## 실행 및 협업

1. `notebooks/01_f401_problem.ipynb`부터 `03_f403_cleaning.ipynb`까지 순서대로 수행합니다.
2. F403 완료 시 `docs/handoff_f403.md`를 채우고 정제 데이터 버전을 고정합니다.
3. PR 검토 후 `main`에 병합합니다.
4. 후속 작업자는 최신 `main`에서 브랜치를 만들고 `04_f404_eda.ipynb`부터 진행합니다.
5. 각 노트북에는 코드, 실행 결과, 해석을 함께 저장합니다.

권장 브랜치: `feat/f401-f403-data`, `feat/f404-f405-analysis`, `feat/f406-extension`.
같은 노트북을 동시에 수정하지 않고 PR 단위로 검토합니다.

## 파일 관리

- `docs/`: 분석 질문, 출처, 데이터 사전, 인수인계
- `notebooks/`: 요구사항별 실행 결과 및 해석
- `src/`: 반복 사용하는 Python 함수
- `data/`, `private/`, PDF, `.env`, 인증 파일: Git 제외
- `outputs/`: 로컬 생성 결과. 제출할 그림·표는 검토 후 별도 공개 경로에 추가
- 원본 명세서는 로컬 `private/`에만 보관하며 공개하지 않습니다.

실제 데이터의 재배포 가능 여부는 데이터 출처의 이용 조건을 확인해 기록합니다.
