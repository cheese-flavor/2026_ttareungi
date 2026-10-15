# 협업 방식 가이드 (Git / Drive)

## 1. 원칙
- **코드·데이터·분석 결과의 단일 진실 원천(source of truth)은 Git 저장소**다.
- Drive는 Git으로 관리하기 애매한 산출물(발표자료, 회의록, 결과 이미지 백업)을 위한 보조 공간이다.
- 최종 제출은 PDF 명세서(1.6)에 따라 **Git Repository** 형태로 한다.

---

## 2. Git 사용 방식

### 저장소
- `https://lab.ssafy.com/dmsdk702/ttareungi_pjt.git` (GitLab), 기본 브랜치: `master`

### 브랜치 전략
- `master`: 항상 동작/제출 가능한 상태 유지
- 작업자별 기능 브랜치를 요구사항 단위로 생성

| 브랜치명 예시 | 담당 |
|---|---|
| `feature/F401-topic-definition` | 본인 |
| `feature/F402-data-collection` | 본인 |
| `feature/F403-preprocessing` | 본인 |
| `feature/F404-eda-viz` | 신현준 |
| `feature/F405-insight` | 신현준 |
| `feature/F406-extension` (선택) | 신현준 |

### 커밋 컨벤션
```
[F402] 따릉이/ASOS 원본 데이터 추가 및 수집기록 작성
[F403] 인코딩 통일 및 날짜 범위 트림, 결측치 처리
```
- 접두어 `[F4xx]`로 어떤 요구사항 작업인지 명시
- 한 커밋은 가능하면 하나의 논리적 변경 단위로

### 병합(Merge Request) 규칙
1. 기능 브랜치에서 작업 → push
2. GitLab에서 `master`로 Merge Request(MR) 생성
3. (2인 팀이라도) 서로 변경사항 간단히 확인 후 머지 — 특히 F403→F404 경계에서 데이터 스키마가 바뀌면 꼭 리뷰
4. 충돌(conflict) 발생 시 데이터 파일(`data/processed/*.csv`)은 임의 수정 금지, 작성자에게 먼저 확인

### 핸드오프 시점 (F403 → F404)
- 본인이 `feature/F403-preprocessing`을 `master`에 머지 완료 후, 2차 담당자에게 아래를 안내:
  - 최신 `master` pull 받을 것
  - `docs/data_collection_log.md`, `PLAN.md` 섹션 2(알려진 이슈) 확인
  - `data/processed/` 산출물과 컬럼 설명 확인
  - 이후 `feature/F404-eda-viz` 브랜치에서 작업 시작

---

## 3. Google Drive 사용 방식

> 저장소 외부에 두는 이유: ipynb 렌더링 공유, 발표자료(ppt), 회의록 등은 Git보다 Drive가 다루기 편함.

### 제안 폴더 구조
```
[Drive] 따릉이_데이터분석프로젝트/
├── 01_원본데이터_백업/        # data/raw 백업본 (Git과 동일 내용 유지)
├── 02_결과물_백업/            # ipynb를 html/pdf로 export한 백업, 주요 그래프 이미지
├── 03_발표자료/                # 보고서/슬라이드(ppt, pdf)
└── 04_회의록_진행기록/         # 역할분담, 핸드오프 메모, 이슈 트래킹
```

### 동기화 규칙
- Drive는 **백업·공유 용도**이며 직접 편집 금지(충돌 방지) — 편집은 로컬에서 하고 결과물만 업로드
- 각 요구사항(F4xx) 완료 시 `04_회의록_진행기록/`에 한 줄 업데이트(무엇을 했는지, 다음 담당자가 할 일)
- Drive 폴더 링크: [따릉이_데이터분석프로젝트](https://drive.google.com/drive/folders/1tciLmBLNx3q8DAy-FAladtRsZJeQy0iN?usp=drive_link)

---

## 4. 요약 플로우

```
[본인] F401~F403 작업
   → feature 브랜치 커밋 → master MR 머지
   → docs/data_collection_log.md, PLAN.md 최신화
   → Drive에 진행 기록 + 산출물 백업 업로드
   → 2차 담당자에게 "master 최신화 완료, F404부터 시작 가능" 안내

[2차 담당자] F404~F406 작업
   → master pull → feature 브랜치 생성 → 작업
   → 완료 시 동일하게 MR 머지 + Drive 기록
```
