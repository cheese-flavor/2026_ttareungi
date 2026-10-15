# 협업 방식 가이드 (Git)

## 1. 원칙
- **코드·데이터·분석 결과의 단일 진실 원천(source of truth)은 Git 저장소**다.
- Google Drive는 사용하지 않기로 결정했다(2026-10-15). 발표자료 등 Git 외 공유가 필요한 자료가
  생기면 별도로 다시 논의한다.
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
| `master` | 항상 최신 완료 상태 (F401~F404 완료본이 들어있음, 2026-10-15 생성) |
| `feature/F401-F403-setup` (작업 기록용, 더 이상 사용 안 함) | 본인 |
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
3. (2인 팀이라도) 서로 변경사항 간단히 확인 후 머지 — 특히 F404→F405 경계에서 결과물(그래프 해석 등)이 늘어나면 꼭 리뷰
4. 충돌(conflict) 발생 시 데이터 파일(`data/processed/*.csv`)은 임의 수정 금지, 작성자에게 먼저 확인

### 핸드오프 시점 (F404 → F405) — 완료됨, 2026-10-15
- `master` 브랜치에 F401~F404 전체가 반영되어 있다. 2차 담당자(신현준)는:
  - [`HANDOFF.md`](../HANDOFF.md) 를 먼저 읽을 것 (진행 현황·데이터 스키마·전제/한계·F405·F406 요구사항 정리됨)
  - `master` pull 받은 뒤 `feature/F405-insight` 브랜치를 새로 만들어 작업 시작

---

## 3. 요약 플로우

```
[본인] F401~F404 작업
   → feature 브랜치 커밋 → master MR 머지
   → docs/data_collection_log.md, PLAN.md 최신화
   → 2차 담당자에게 "master 최신화 완료, F405부터 시작 가능" 안내

[2차 담당자] F405~F406 작업
   → master pull → feature 브랜치 생성 → 작업
   → 완료 시 동일하게 MR 머지
```
