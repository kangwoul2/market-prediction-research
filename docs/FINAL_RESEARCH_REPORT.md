# Market Prediction Research — Final Research Report

## 1. 연구 목적

이 연구의 목적은 Bitcoin 다음 날 가격을 높은 정확도로 맞히는 모델을 만드는 것에만 있지 않습니다.

핵심 질문은 다음과 같습니다.

> 오늘 시점까지 실제로 관측 가능한 시장, 다른 암호자산, 온체인, 거시시장, 심리 데이터를 사용했을 때 다음 날 Bitcoin 움직임에 대해 미래정보 누수 없이 반복 가능한 예측 신호가 존재하는가?

초기 방향 분류 연구를 재검토하면서 데이터 누수, 무작위 분할, 테스트 구간 재사용, 클래스 불균형의 영향을 확인했습니다. 이후 point-in-time 정렬, purged walk-forward validation, train-only feature selection, 문제 정의 비교, 재현성 기록을 중심으로 검증 구조를 다시 설계했습니다.

## 2. 데이터

개발 데이터:

```text
기간      2015-01-01 ~ 2026-09-17
행        4,278
생성 열   311
```

주요 데이터:

```text
BTC 가격과 거래량
ETH, BNB, XRP, SOL, ADA, DOGE, LTC
SPY, QQQ, Gold, DXY, VIX, TLT
Blockchain.com 온체인 지표
Fear & Greed
```

X 게시글 감성 기능은 구현했지만 최종 잠금 성능에는 충분한 역사 데이터가 없어 포함하지 않았습니다.

## 3. 시간 누수 방지

예측 규칙은 `t일 종료 시점까지 알 수 있는 정보로 t+1일 결과를 예측`하는 것입니다.

각 validation fold에서 다음 조건을 강제했습니다.

```text
train feature date < validation start date
train target end date < validation start date
```

정답 계산 구간이 validation과 겹치는 train row는 purge했습니다.

외부 일 단위 데이터는 미래 값을 과거 결측에 채우지 않았고, 미국 시장 휴장일에는 마지막 관측값만 forward-fill한 뒤 계산된 피처를 다시 1일 lag했습니다.

피처 선택도 fold의 train 구간에서만 수행했습니다.

## 4. 문제 정의 비교

동일한 시간 순 검증 환경에서 다음 문제를 비교했습니다.

| Task | 결과 |
|---|---:|
| 3-class | Macro F1 0.4445 |
| Direction binary | Macro F1 0.5021, ROC AUC 0.5130 |
| Move binary | Macro F1 0.6517, ROC AUC 0.7143 |
| Return regression | Pearson 약 0, sign accuracy 약 0.51 |

다음 날 방향 부호와 수익률 값 자체에서는 안정적인 신호가 거의 없었습니다.

반면 다음 날 큰 움직임 발생 여부에서는 상대적으로 강한 신호가 관찰되어 후속 연구를 이 문제에 집중했습니다.

## 5. Move target 정의

```text
MOVE = 1 if abs(next_day_return) > k × trailing_volatility
MOVE = 0 otherwise
```

개발 CV에서 선택된 값:

```text
volatility window = 7 days
k = 1.1
```

이 설정에서 전체 MOVE 비율은 약 26.7%였습니다.

## 6. 학습 기간 비교

```text
365일
730일
1095일
1825일
expanding
```

본검증 결과:

| Training window | Macro F1 | Balanced Accuracy | ROC AUC | Robust Score |
|---|---:|---:|---:|---:|
| 730 | 0.6450 | 0.6511 | 0.6906 | 0.6370 |
| 1095 | 0.6274 | 0.6297 | 0.6844 | 0.6192 |

최근 730일이 선택되었습니다.

이는 오래된 데이터가 일반적으로 필요 없다는 의미가 아니라 이번 문제와 검증 구간에서는 최근 약 2년의 데이터가 더 안정적이었다는 의미입니다.

## 7. 라벨 기준 비교

volatility window 7, 14, 21, 30, 60일과 k 0.5, 0.7, 0.9, 1.1을 탐색했습니다.

본검증 상위 결과:

| Volatility Window | k | Macro F1 | ROC AUC | Robust Score |
|---|---:|---:|---:|---:|
| 7 | 1.1 | 0.6507 | 0.7238 | 0.6408 |
| 7 | 0.9 | 0.6450 | 0.6906 | 0.6370 |
| 21 | 0.7 | 0.5960 | 0.6393 | 0.5925 |

7일 변동성과 k 1.1이 선택되었습니다.

## 8. 피처군과 외부시장 비교

| Feature Set | Model | Macro F1 | Balanced Accuracy | ROC AUC | Robust Score |
|---|---|---:|---:|---:|---:|
| BTC + Crypto | ExtraTrees | 0.6576 | 0.6644 | **0.7395** | **0.6502** |
| BTC + Macro | ExtraTrees | **0.6606** | **0.6684** | 0.7263 | 0.6494 |
| BTC + All External | ExtraTrees | 0.6507 | 0.6612 | 0.7238 | 0.6408 |

거시시장 데이터도 경쟁력 있는 신호를 보였습니다. 단일 Macro F1은 BTC + Macro가 조금 높았지만 fold 변동성을 함께 고려한 Robust Score는 BTC + Crypto가 근소하게 높았습니다.

차이가 매우 작기 때문에 외부시장 피처 간 우열을 강하게 주장하지 않습니다.

## 9. 피처 수 비교

| Top K | Macro F1 | Balanced Accuracy | ROC AUC | Robust Score |
|---|---:|---:|---:|---:|
| 50 | **0.6743** | **0.6990** | **0.7458** | **0.6674** |
| 75 | 0.6710 | 0.6888 | 0.7457 | 0.6616 |
| 30 | 0.6590 | 0.6859 | 0.7414 | 0.6505 |
| 100 | 0.6576 | 0.6644 | 0.7395 | 0.6502 |
| 150 | 0.6516 | 0.6595 | 0.7329 | 0.6431 |

50개가 가장 안정적이었습니다. 피처를 더 많이 넣는 것이 자동으로 일반화 성능 향상으로 이어지지 않았습니다.

## 10. 최종 개발 설정

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

최종 6-fold OOF:

```text
Accuracy              0.7157
Balanced Accuracy     0.6637
Macro F1              0.6589
ROC AUC               0.7212
Brier Score            0.2041
Fold Macro F1 mean    0.6497
Fold Macro F1 std     0.0387
Worst fold Macro F1   0.5918
```

## 11. Threshold 시간순 검증

OOF 앞쪽 60%에서 threshold를 선택하고 뒤쪽 40%에서는 값을 바꾸지 않고 검증했습니다.

선택 threshold:

```text
0.50
```

뒤쪽 432개 구간:

```text
Accuracy              0.6921
Balanced Accuracy     0.6861
Macro F1              0.6588
ROC AUC               0.7239
PR AUC                0.4833
Brier Score            0.2122
```

이는 개발 데이터 안에서 threshold 선택과 검증을 시간상 분리한 결과이며 실제 미래 성능을 보장하지 않습니다.

## 12. Baseline 비교

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Always No Move | 0.7176 | 0.5000 | 0.4178 |
| Previous Day Move | 0.5972 | 0.5056 | 0.5056 |
| Final Model | 0.6921 | 0.6861 | 0.6588 |

Always No Move는 클래스 불균형 때문에 Accuracy가 더 높습니다. 그러나 MOVE class를 구분하지 못해 Balanced Accuracy와 Macro F1은 낮습니다.

## 13. 실험 관리

실험 공간이 커지면서 다음 구조를 추가했습니다.

```text
persistent experiment cache
stage checkpoint
coarse-to-fine search
train-only feature selection
```

데이터, 코드, 설정이 동일한 조합은 재학습하지 않고 결과를 재사용합니다.

## 14. 재현성

최종 잠금 실행에 대해 Python 버전, 패키지 버전, 데이터 hash, 설정 hash와 reproducibility ID를 기록했습니다.

```text
reproducibility_id
a2d0aa38c88ed237c753f43d5c578da2d9cf50f9315cd3f45fa53890d4a9ec6d
```

## 15. Development Lock

개발 과정에서 확인한 마지막 target_end_date는 2026-09-15입니다.

이후 새 데이터는 다음 항목을 다시 고르는 데 사용하지 않습니다.

```text
training window
volatility window
k
feature family
model
feature count
decision threshold
```

## 16. 최종 해석

확인된 결과:

```text
다음 날 상승/하락 방향 예측은 거의 랜덤 수준이었습니다.
다음 날 수익률 직접 회귀도 유의미한 상관 신호가 없었습니다.
다음 날 큰 움직임 발생 여부에서는 상대적으로 안정적인 신호가 있었습니다.
최근 730일 학습창이 선택되었습니다.
7일 변동성의 1.1배 기준이 가장 안정적이었습니다.
50개 피처가 더 많은 피처보다 일반화 성능이 좋았습니다.
cross-crypto와 macro 정보 모두 move detection에서 경쟁력 있는 신호를 보였습니다.
```

주장하지 않는 내용:

```text
Bitcoin 다음 날 방향을 안정적으로 맞힙니다.
실제 투자 수익을 보장합니다.
현재 OOF 결과가 실제 미래 성능과 같습니다.
거시시장이나 온체인 정보가 인과적으로 Bitcoin을 움직입니다.
X sentiment가 최종 성능을 개선했습니다.
```

## 17. 결론

이 연구의 핵심 산출물은 높은 점수 하나가 아니라 다음과 같습니다.

```text
point-in-time 데이터 정렬
purged walk-forward validation
문제 정의 비교
cross-asset feature engineering
feature family ablation
sliding window 비교
train-only feature selection
chronological threshold verification
experiment cache
reproducibility manifest
development config lock
```

실제 미래 일반화 성능은 잠긴 설정을 새로운 데이터에 그대로 적용하면서 별도로 확인해야 합니다.
