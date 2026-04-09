import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor that splits cat/con features, applies LabelEncoding, and builds missing masks.

    Parameters
    ----------
    cat_features : list of int, optional
        Indices of categorical columns. If None, inferred from pandas dtype (object/category).
    """

    def __init__(self, cat_features=None):
        self.cat_features = cat_features

    def fit(self, X, y=None):
        X = self._to_dataframe(X)
        n_features = X.shape[1]

        if self.cat_features is not None:
            self.cat_idxs_ = sorted(list(self.cat_features))
        else:
            self.cat_idxs_ = [
                i for i, col in enumerate(X.columns)
                if X[col].dtype.name in ('object', 'category')
            ]
        self.con_idxs_ = [i for i in range(n_features) if i not in self.cat_idxs_]
        self.n_features_in_ = n_features
        self.feature_names_in_ = list(X.columns)

        self.encoders_ = {}
        self.cat_dims_ = []
        for idx in self.cat_idxs_:
            col = X.iloc[:, idx].astype(str).fillna("MissingValue")
            enc = LabelEncoder()
            enc.fit(col.values)
            self.encoders_[idx] = enc
            self.cat_dims_.append(len(enc.classes_))

        self.mean_ = np.zeros(len(self.con_idxs_), dtype=np.float32)
        self.std_ = np.ones(len(self.con_idxs_), dtype=np.float32)
        for k, idx in enumerate(self.con_idxs_):
            col = pd.to_numeric(X.iloc[:, idx], errors='coerce')
            self.mean_[k] = float(col.mean())
            s = float(col.std())
            self.std_[k] = s if s > 1e-6 else 1e-6

        return self

    def transform(self, X):
        X = self._to_dataframe(X)
        n = len(X)

        X_combined = np.zeros((n, self.n_features_in_), dtype=np.float64)
        nan_mask = np.ones((n, self.n_features_in_), dtype=np.int64)

        X_cat = np.zeros((n, len(self.cat_idxs_)), dtype=np.int64)
        cat_mask = np.ones((n, len(self.cat_idxs_)), dtype=np.float32)

        for k, idx in enumerate(self.cat_idxs_):
            col = X.iloc[:, idx]
            is_null = col.isna()
            nan_mask[:, idx] = (~is_null).astype(int)
            cat_mask[:, k] = (~is_null).astype(float)
            col_str = col.astype(str).where(~is_null, "MissingValue")
            enc = self.encoders_[idx]
            codes = np.array([
                enc.transform([v])[0] if v in enc.classes_
                else enc.transform(["MissingValue"])[0]
                for v in col_str
            ], dtype=np.int64)
            X_cat[:, k] = codes
            X_combined[:, idx] = codes

        X_con = np.zeros((n, len(self.con_idxs_)), dtype=np.float32)
        con_mask = np.ones((n, len(self.con_idxs_)), dtype=np.float32)

        for k, idx in enumerate(self.con_idxs_):
            col = pd.to_numeric(X.iloc[:, idx], errors='coerce')
            is_null = col.isna()
            nan_mask[:, idx] = (~is_null).astype(int)
            con_mask[:, k] = (~is_null).astype(float)
            filled = col.fillna(float(self.mean_[k])).values.astype(np.float32)
            X_con[:, k] = filled
            X_combined[:, idx] = filled

        return {
            'X_cat': X_cat,
            'X_con': X_con,
            'cat_mask': cat_mask,
            'con_mask': con_mask,
            'X_combined': X_combined,
            'nan_mask': nan_mask,
        }

    def inverse_transform(self, X_combined):
        """Restore the combined matrix back to the original feature space."""
        n = X_combined.shape[0]
        result = np.empty((n, self.n_features_in_), dtype=object)
        for k, idx in enumerate(self.cat_idxs_):
            enc = self.encoders_[idx]
            codes = np.clip(X_combined[:, idx].astype(int), 0, len(enc.classes_) - 1)
            result[:, idx] = enc.inverse_transform(codes)
        for k, idx in enumerate(self.con_idxs_):
            result[:, idx] = X_combined[:, idx].astype(float)
        return result

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        elif isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        else:
            raise ValueError(f"Expected DataFrame or ndarray, got {type(X)}")
