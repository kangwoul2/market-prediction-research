# 재현성 기록

## 기준 자료

- 원래 연구 설명: 학사학위논문 PDF
- 기존 모델 비교 수치: `model_performance_results.csv`
- 연구 실행 기록: 번호가 붙은 Jupyter Notebook
- 저장 모델: `LSTM_model.h5`, `GRU_model.h5`, `Transformer_model.keras`
- Research V2 검증 결과: `research_v2/verified_results.csv`

## 기존 노트북을 지우지 않는 이유

기존 노트북에는 연구 당시의 탐색 과정과 시행착오가 남아 있습니다. 결과를 더 깔끔하게 보이게 만들기 위해 출력이나 수치를 임의로 수정하지 않습니다.

원래 연구와 이후 V2를 구분해서 보존하는 이유는 **평가 방식이 달라졌기 때문**입니다.

## 수치 작성 원칙

README나 이력서에 성능 수치를 사용할 때는 저장소에 실제 결과 파일이 존재하는 값만 사용합니다.

- 기존 연구 수치 → `model_performance_results.csv`
- V2 수치 → `research_v2/verified_results.csv`

평가 조건이 다른 두 결과를 직접적인 개선율로 계산하지 않습니다.

## Research V2 재현 방법

```bash
pip install -r requirements-research-v2.txt
python research_v2/run_research.py
python research_v2/run_extended_research.py
```

GitHub Actions도 같은 실행 경로를 사용합니다.

## 재현성을 높이기 위해 지키는 기준

1. 데이터와 사용 종목 목록을 명확하게 기록
2. 학습·검증·테스트를 시간 순으로 분리
3. 전처리 기준은 학습 데이터에서만 계산
4. 난수 시드 기록
5. 모델과 변수 설정 기록
6. 한 명령으로 학습·평가·결과 저장이 가능하게 구성
7. 결과 파일을 기존 결과와 구분해 보존

면접에서는 **“좋은 점수보다 같은 코드와 데이터로 같은 평가 과정을 다시 실행할 수 있는지를 중요하게 봤다”**고 설명합니다.