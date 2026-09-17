# Prospective Evaluation Protocol

## 목적

개발 과정에서 2026-09-15까지의 target 결과를 확인했습니다.

따라서 이후 새 데이터는 잠긴 설정을 다시 고르는 데 사용하지 않고, 기존 모델의 실제 미래 일반화 성능을 확인하는 용도로만 사용합니다.

## 잠긴 설정

```text
Task                 next-day meaningful move binary
Training window      730 days
Volatility window    7 days
k                    1.1
Feature family       price_crypto
Classifier           ExtraTrees
Top K features       50
Decision threshold   0.50
```

잠금 파일:

```text
reports/outputs/final_development_lock.json
```

## 금지 사항

새 결과를 확인한 뒤 같은 잠금 연구 안에서 아래 값을 변경하지 않습니다.

```text
training window
volatility window
k
feature family
model type
feature count
decision threshold
```

성능이 낮더라도 기존 잠금 결과는 유지합니다.

새로운 아이디어는 별도의 개발 연구로 분리합니다.

## 날짜별 기록

```text
feature_date
target_end_date
predicted_move_probability
decision_threshold
predicted_label
actual_label
actual_next_return
trailing_7d_volatility
correct
```

## 누적 평가 지표

```text
Accuracy
Balanced Accuracy
Macro F1
ROC AUC
PR AUC
Brier Score
Confusion Matrix
MOVE prevalence
```

Accuracy는 클래스 불균형 때문에 단독으로 해석하지 않습니다.

## 평가 시점

```text
30 labeled days
60 labeled days
90 labeled days
180 labeled days
365 labeled days
```

30일 결과는 초기 관찰값으로만 보고, 90일 이상부터 일반화 판단에 더 큰 의미를 둡니다.

## Baseline

같은 날짜에 다음 baseline도 함께 계산합니다.

```text
Always No Move
Previous Day Move
```

## 원칙

> 개발 이후의 새로운 데이터는 모델을 더 좋아 보이게 수정하기 위한 재료가 아니라, 이미 고정한 가설과 설정이 실제 미래에서도 유지되는지 확인하기 위한 평가 데이터로 사용합니다.
