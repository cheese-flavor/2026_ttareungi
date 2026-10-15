# 저장소별 업로드 기준

- 작업 기준: GitLab의 master. 후속 작업은 feature/F405-insight에서 진행한다.
- GitLab: data/raw와 data/processed를 포함한다. 명세서 PDF와 인증정보는 제외한다.
- GitHub: 향후 동일한 코드·분석 문서를 반영하되 data/ 전체와 명세서 PDF, 인증정보를 제외한다.
- .gitignore는 원격 저장소별로 다르게 적용되지 않으므로 GitLab 작업 브랜치에서 data/를 무시하지 않는다.
- GitHub에는 GitLab 브랜치 이력을 그대로 push하지 않고, 데이터와 PDF를 제외한 별도 배포본을 구성해 반영한다. 노트북의 분석 결과와 해석은 유지하되 원본 데이터 전체를 출력·내장했는지 점검한다.
- 기존 GitLab 이력에 이미 포함된 PDF는 과거 커밋에 남는다. 이번 변경은 현재 브랜치에서 추적을 해제하며 기존 이력을 재작성하지 않는다.
- 로컬 원격 이름: gitlab = GitLab, origin = 기존 GitHub. 기본 push 대상은 gitlab로 설정한다.
