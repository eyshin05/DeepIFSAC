import numpy as np
import pandas as pd
from deepifsac import TabularPreprocessor


def test_import():
    assert TabularPreprocessor is not None


# --- fit() ---

def test_fit_detects_categorical_from_dtype():
    df = pd.DataFrame({
        'age': [25.0, 30.0, 45.0],
        'gender': pd.Categorical(['M', 'F', 'M']),
    })
    p = TabularPreprocessor()
    p.fit(df)
    assert p.cat_idxs_ == [1]
    assert p.con_idxs_ == [0]
    assert 1 in p.encoders_
    assert p.n_features_in_ == 2
    assert len(p.cat_dims_) == 1


def test_fit_uses_cat_features_for_ndarray():
    X = np.array([[25.0, 0], [30.0, 1], [45.0, 0]], dtype=object)
    p = TabularPreprocessor(cat_features=[1])
    p.fit(X)
    assert p.cat_idxs_ == [1]
    assert p.con_idxs_ == [0]


def test_fit_computes_mean_std():
    df = pd.DataFrame({
        'age': [10.0, 20.0, 30.0],
        'g': pd.Categorical(['a', 'b', 'a']),
    })
    p = TabularPreprocessor()
    p.fit(df)
    np.testing.assert_allclose(p.mean_[0], 20.0)
    assert p.std_[0] > 0


# --- transform() ---

def test_transform_shapes_and_masks():
    df = pd.DataFrame({
        'age': [25.0, np.nan, 45.0],
        'gender': pd.Categorical(['M', None, 'F']),
        'income': [50000.0, 60000.0, np.nan],
    })
    p = TabularPreprocessor()
    p.fit(df)
    result = p.transform(df)
    assert result['X_cat'].shape == (3, 1)
    assert result['X_con'].shape == (3, 2)
    assert result['X_combined'].shape == (3, 3)
    assert result['nan_mask'].shape == (3, 3)
    assert result['con_mask'][1, 0] == 0.0   # age[1]=NaN
    assert result['cat_mask'][1, 0] == 0.0   # gender[1]=None
    assert result['con_mask'][2, 1] == 0.0   # income[2]=NaN


def test_transform_fills_nan_with_train_mean():
    df_train = pd.DataFrame({'x': [10.0, 20.0, 30.0]})
    df_test = pd.DataFrame({'x': [np.nan]})
    p = TabularPreprocessor()
    p.fit(df_train)
    result = p.transform(df_test)
    np.testing.assert_allclose(result['X_con'][0, 0], 20.0)


# --- inverse_transform() ---

def test_inverse_transform_roundtrip():
    df = pd.DataFrame({
        'age': [25.0, 30.0, 45.0],
        'gender': pd.Categorical(['M', 'F', 'M']),
    })
    p = TabularPreprocessor()
    p.fit(df)
    result = p.transform(df)
    restored = p.inverse_transform(result['X_combined'])
    np.testing.assert_allclose(restored[:, 0].astype(float), [25.0, 30.0, 45.0], rtol=1e-4)
    assert list(restored[:, 1]) == ['M', 'F', 'M']
