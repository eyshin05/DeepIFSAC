# DeepIFSAC sklearn-style Imputer 설계 문서

**날짜**: 2026-04-09  
**대상 Repository**: DeepIFSAC (Attention + Contrastive Learning 기반 결측값 보완 프레임워크)

---

## 1. 배경 및 목표

현재 DeepIFSAC는 `my_train.py` 스크립트 방식으로만 사용 가능하며, 데이터 로딩이 OpenML에 종속되어 있다. 사용자가 임의의 numpy/pandas 데이터에 바로 적용할 수 없는 구조이다.

**목표**: sklearn-style API (`fit` / `transform` / `get_features`)를 갖춘 imputer 클래스를 추가하여 임의 데이터셋에 바로 적용할 수 있도록 한다. 기존 스크립트 방식은 그대로 유지한다.

---

## 2. 요구사항 요약

| 항목 | 결정 |
|------|------|
| 주요 목적 | 결측값 보완 + 피처 추출(임베딩) |
| 입력 형식 | `np.ndarray` 또는 `pd.DataFrame` (NaN으로 결측 표시) |
| 카테고리 감지 | pandas dtype 자동 감지 + `cat_features` 파라미터 병용 |
| 사전학습 | `pretrain=True/False`로 옵션 제어 |
| 출력 | `transform()` → 보완된 X (원본 공간), `get_features()` → Transformer 임베딩 |
| 클래스 구조 | `TabularPreprocessor` (노출) + `DeepIFSACImputer` |

---

## 3. 파일 구조

```
DeepIFSAC/
├── imputer/                      ← 신규 추가
│   ├── __init__.py               ← TabularPreprocessor, DeepIFSACImputer 노출
│   ├── preprocessor.py           ← TabularPreprocessor
│   └── imputer.py                ← DeepIFSACImputer
├── models/                       ← 기존 유지 (수정 없음)
│   ├── model.py
│   └── pretrainmodel.py
├── pretraining.py                ← 인터페이스 일부 수정 (범용화)
├── corruptor.py                  ← 기존 유지
├── augmentations.py              ← 기존 유지
├── data_openml.py                ← 기존 유지
└── my_train.py                   ← 기존 유지
```

---

## 4. TabularPreprocessor

**파일**: `imputer/preprocessor.py`

sklearn의 `BaseEstimator`, `TransformerMixin` 상속.

### 4.1 초기화

```python
TabularPreprocessor(cat_features=None)
```

- `cat_features`: 카테고리 컬럼 인덱스 리스트 (ndarray용). `None`이면 pandas dtype에서 자동 감지 (`object`, `category`).

### 4.2 fit(X)

1. X가 DataFrame이면 dtype (`object`, `category`)으로 카테고리 컬럼 감지. ndarray면 `cat_features` 사용.
2. 카테고리 컬럼: `sklearn.preprocessing.LabelEncoder` 각 컬럼별 피팅. `cat_dims_` (고유값 수 + 1, unknown 처리용) 저장.
3. 연속형 컬럼: 컬럼별 mean, std 계산 (결측값 제외). `mean_`, `std_` 저장.
4. 저장 속성: `cat_idxs_`, `con_idxs_`, `cat_dims_`, `encoders_`, `mean_`, `std_`, `n_features_in_`

### 4.3 transform(X)

입력 X를 모델 입력 형식으로 변환하여 dict 반환:

```python
{
    'X_cat': np.ndarray (n, n_cat),       # 인코딩된 카테고리 (정수), 결측=0
    'X_con': np.ndarray (n, n_con),       # 정규화된 연속형, 결측=0.0
    'cat_mask': np.ndarray (n, n_cat),    # 1=값 존재, 0=결측
    'con_mask': np.ndarray (n, n_con),    # 1=값 존재, 0=결측
}
```

### 4.4 inverse_transform(X_cat, X_con)

- `X_cat` 각 컬럼에 LabelEncoder.inverse_transform 적용 (카테고리 복원)
- `X_con` 각 컬럼에 `mean_ + std_ * X_con` 역정규화
- 반환: 원본 피처 공간의 np.ndarray (n, n_features)

---

## 5. DeepIFSACImputer

**파일**: `imputer/imputer.py`

sklearn의 `BaseEstimator`, `TransformerMixin` 상속.

### 5.1 초기화

```python
DeepIFSACImputer(
    cat_features=None,
    pretrain=True,
    pretrain_epochs=100,
    pt_tasks=['denoising', 'contrastive'],
    pt_aug=['cutmix'],
    epochs=50,
    embedding_size=32,
    transformer_depth=6,
    attention_heads=8,
    attention_type='colrow',
    missing_rate=0.3,
    missing_type='mcar',
    corruption_type='cutmix',
    batch_size=256,
    lr=1e-4,
    device='auto',
    random_state=42,
)
```

- `device='auto'`: CUDA 사용 가능 시 자동으로 `cuda:0` 선택, 없으면 `cpu`
- `missing_rate`: 사전학습 시 인위적으로 생성하는 결측률 (실제 X의 결측과 별개)

### 5.2 fit(X, y=None)

```
X (원본, NaN 포함)
    ↓ self._preprocessor.fit_transform(X)
    ↓ X_cat, X_con, cat_mask, con_mask
    ↓ Corruptor → X_cat_imp, X_con_imp (사전학습용 보완 데이터)
    ↓ DataSetCatCon_imputedX 생성
    ↓ [if pretrain=True] DeepIFSAC_pretrain()
    ↓ denoising fine-tuning (pretrain=False면 여기서만 학습)
    ↓ self.model_ 저장
```

- 내부적으로 `self._preprocessor = TabularPreprocessor(cat_features)` 생성 후 fit
- `y`는 현재 무시 (레이블 없는 보완에 집중). 향후 지도 학습 확장 여지 남김

### 5.3 transform(X)

결측값이 채워진 X를 반환 (원본과 동일한 shape, 존재하는 값은 유지).

```
X
    ↓ self._preprocessor.transform(X)  → X_cat, X_con, cat_mask, con_mask
    ↓ embed_data_mask()                → 마스크 임베딩 적용
    ↓ self.model_.forward()            → cat_outs, con_outs (전체 피처 예측)
    ↓ 결측 위치(mask=0)만 예측값으로 교체
    ↓ self._preprocessor.inverse_transform()
→ np.ndarray (n, n_features)
```

### 5.4 get_features(X)

Transformer 마지막 레이어 임베딩 반환.

```
X
    ↓ self._preprocessor.transform(X)
    ↓ self.model_.transformer()        → hidden states
    ↓ 전체 피처에 대한 임베딩 concat
→ np.ndarray (n, embedding_size * n_features)
```

### 5.5 fit_transform(X, y=None)

`fit(X, y).transform(X)` 와 동일. `TransformerMixin`이 자동 제공하므로 별도 구현 불필요.

---

## 6. pretraining.py 수정

현재 `DeepIFSAC_pretrain()`은 OpenML 특화 데이터 구조에 의존. 범용화를 위해:

- 입력 파라미터를 `X_train` (raw numpy) 대신 이미 전처리된 `X_cat`, `X_con`, `masks` 형태로도 받을 수 있도록 오버로드 또는 내부 분기 추가
- 기존 스크립트 방식(`my_train.py`)과의 호환성 유지

---

## 7. 사용 예시

```python
import numpy as np
import pandas as pd
from imputer import DeepIFSACImputer, TabularPreprocessor

# 데이터 준비 (NaN으로 결측 표시)
df = pd.read_csv("data.csv")  # object 컬럼은 자동으로 카테고리 인식

# 결측값 보완
imputer = DeepIFSACImputer(pretrain=True, pretrain_epochs=100)
imputer.fit(df)
df_imputed = imputer.transform(df)  # np.ndarray, shape 동일

# 임베딩 추출
X_embed = imputer.get_features(df)  # (n_samples, embedding_size * n_features)

# 다운스트림 ML
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(X_embed_train, y_train)

# TabularPreprocessor 단독 사용
preprocessor = TabularPreprocessor(cat_features=[0, 2])
preprocessor.fit(X_train)
processed = preprocessor.transform(X_test)
```

---

## 8. 검증 계획

1. **단위 테스트 (TabularPreprocessor)**
   - DataFrame + ndarray 입력 각각 동작 확인
   - `inverse_transform(transform(X))` ≈ X (결측 제외)
   - 신규 카테고리 값 처리 (unknown)

2. **통합 테스트 (DeepIFSACImputer)**
   - `fit().transform()` shape 보존 확인
   - 결측 위치만 채워지고, 기존 값은 유지되는지 확인
   - `get_features()` shape: `(n_samples, embedding_size * n_features)`

3. **회귀 테스트**
   - `my_train.py` 기존 스크립트가 그대로 동작하는지 확인 (imputer/ 추가 후)

4. **NRMSE 검증**
   - 인위적으로 결측 생성 후 보완 → 원본과 NRMSE 비교
   - 기존 논문 수준의 성능 재현 가능 여부 확인
