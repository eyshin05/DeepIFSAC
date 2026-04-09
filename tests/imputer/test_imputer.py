import numpy as np
import pandas as pd


def test_import():
    from deepifsac import DeepIFSACImputer
    assert DeepIFSACImputer is not None


def _make_simple_df(n=80):
    return pd.DataFrame({
        'age': [float(i % 40 + 20) for i in range(n)],
        'income': [float((i % 30 + 10) * 1000) for i in range(n)],
    })


def _small_imputer(**kwargs):
    from deepifsac import DeepIFSACImputer
    defaults = dict(
        pretrain=False,
        pretrain_epochs=1,
        embedding_size=4,
        transformer_depth=1,
        attention_heads=2,
        batch_size=8,
        device='cpu',
    )
    defaults.update(kwargs)
    return DeepIFSACImputer(**defaults)


# --- fit() ---

def test_fit_sets_model_and_preprocessor():
    imputer = _small_imputer()
    imputer.fit(_make_simple_df())
    assert hasattr(imputer, 'model_')
    assert hasattr(imputer, 'preprocessor_')


def test_fit_with_categorical():
    df = pd.DataFrame({
        'age': [float(i % 40 + 20) for i in range(80)],
        'gender': pd.Categorical(['M', 'F'] * 40),
        'income': [float((i % 30 + 10) * 500) for i in range(80)],
    })
    imputer = _small_imputer()
    imputer.fit(df)
    assert hasattr(imputer, 'model_')


# --- transform() ---

def test_transform_no_nan_in_output():
    imputer = _small_imputer()
    imputer.fit(_make_simple_df())
    df_test = pd.DataFrame({'age': [25.0, np.nan], 'income': [np.nan, 30000.0]})
    result = imputer.transform(df_test)
    assert result.shape == (2, 2)
    assert not np.isnan(result).any()


def test_transform_preserves_present_values():
    imputer = _small_imputer()
    imputer.fit(_make_simple_df())
    df_test = pd.DataFrame({'age': [25.0, np.nan], 'income': [np.nan, 30000.0]})
    result = imputer.transform(df_test)
    np.testing.assert_allclose(result[0, 0], 25.0, rtol=1e-4)
    np.testing.assert_allclose(result[1, 1], 30000.0, rtol=1e-4)


# --- get_features() ---

def test_get_features_shape():
    emb_size = 8
    imputer = _small_imputer(embedding_size=emb_size)
    imputer.fit(_make_simple_df())
    features = imputer.get_features(_make_simple_df().iloc[:5])
    n_features = 2  # age + income
    assert features.shape == (5, n_features * emb_size)
    assert not np.isnan(features).any()


# --- public API ---

def test_public_api_import():
    from deepifsac import TabularPreprocessor, DeepIFSACImputer
    assert TabularPreprocessor is not None
    assert DeepIFSACImputer is not None
