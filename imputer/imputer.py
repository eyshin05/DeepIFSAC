import sys
import os
import numpy as np
from argparse import Namespace
from sklearn.base import BaseEstimator, TransformerMixin

# 루트 경로를 sys.path에 추가 (기존 모듈 임포트용)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from imputer.preprocessor import TabularPreprocessor


class DeepIFSACImputer(BaseEstimator, TransformerMixin):
    """sklearn-style DeepIFSAC imputer.

    fit(X)으로 모델 학습, transform(X)으로 결측값 보완,
    get_features(X)으로 Transformer 임베딩 추출.

    Parameters
    ----------
    cat_features : list of int, optional
        카테고리 컬럼 인덱스. None이면 pandas dtype 자동 감지.
    pretrain : bool
        True면 Contrastive 사전학습 포함, False면 Denoising만 수행.
    pretrain_epochs : int
        사전학습 에포크 수.
    embedding_size : int
        Transformer 임베딩 차원.
    transformer_depth : int
        Transformer 레이어 수.
    attention_heads : int
        Multi-head attention 헤드 수.
    attention_type : str
        어텐션 타입. 'col', 'colrow', 'row', 'rowcol', 'parallel', 'colrowatt'.
    missing_rate : float
        학습 시 추가 인위적 결측률 (0~1).
    missing_type : str
        결측 패턴. 'mcar', 'mnar', 'mar'.
    corruption_type : str
        증강 방식. 'cutmix', 'zeroes', 'median', 'no_corruption'.
    batch_size : int
        배치 크기.
    device : str
        'auto', 'cpu', 'cuda:0' 등.
    random_state : int
        재현성을 위한 시드.
    """

    def __init__(
        self,
        cat_features=None,
        pretrain=True,
        pretrain_epochs=100,
        pt_tasks=None,
        pt_aug=None,
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
    ):
        self.cat_features = cat_features
        self.pretrain = pretrain
        self.pretrain_epochs = pretrain_epochs
        self.pt_tasks = pt_tasks
        self.pt_aug = pt_aug
        self.embedding_size = embedding_size
        self.transformer_depth = transformer_depth
        self.attention_heads = attention_heads
        self.attention_type = attention_type
        self.missing_rate = missing_rate
        self.missing_type = missing_type
        self.corruption_type = corruption_type
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """모델을 학습한다.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            결측값(NaN)이 포함된 학습 데이터.
        y : ignored

        Returns
        -------
        self
        """
        import torch
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        device = self._resolve_device()

        # 1. 전처리
        self.preprocessor_ = TabularPreprocessor(cat_features=self.cat_features)
        self.preprocessor_.fit(X)
        processed = self.preprocessor_.transform(X)

        X_combined = processed['X_combined'].astype(np.float32)
        nan_mask = processed['nan_mask'].astype(np.int64)
        n = X_combined.shape[0]

        cat_idxs = self.preprocessor_.cat_idxs_
        con_idxs = self.preprocessor_.con_idxs_

        # 2. 학습용 imputed 버전 (NaN을 mean으로 채운 버전)
        import torch
        X_train_imp = torch.tensor(X_combined)
        train_mask_full = torch.tensor(1 - nan_mask, dtype=torch.float32)  # 1=missing, (n, n_features)
        # pretraining.py의 denoising loss는 con_outs (연속형 출력)와 train_mask를 곱하므로
        # t_mask를 연속형 컬럼만으로 슬라이싱해야 shape mismatch를 방지함
        train_mask = train_mask_full[:, con_idxs] if len(con_idxs) > 0 else train_mask_full

        # 3. 정규화 파라미터
        train_mean = self.preprocessor_.mean_
        train_std = self.preprocessor_.std_
        continuous_mean_std = np.array([train_mean, train_std], dtype=np.float32)
        imp_continuous_mean_std = continuous_mean_std

        # 4. cat_dims (CLS 토큰 prepend)
        cat_dims = np.array([1] + list(self.preprocessor_.cat_dims_), dtype=int)

        # 5. DataSet 구성
        from data_openml import DataSetCatCon_imputedX
        X_dict = {'data': X_combined, 'mask': nan_mask}
        Y_dict = {'data': np.zeros((n, 1), dtype=np.int64)}
        train_ds = DataSetCatCon_imputedX(
            X_dict, X_train_imp.cpu().numpy(), Y_dict,
            train_mask, cat_idxs, 'clf',
            continuous_mean_std, imp_continuous_mean_std,
        )

        # 6. 모델 생성
        from models.pretrainmodel import DeepIFSAC
        self.model_ = DeepIFSAC(
            categories=tuple(cat_dims),
            num_continuous=len(con_idxs),
            dim=self.embedding_size,
            dim_out=1,
            depth=self.transformer_depth,
            heads=self.attention_heads,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2),
            cont_embeddings='MLP',
            attentiontype=self.attention_type,
            final_mlp_style='sep',
            y_dim=2,
        ).to(device)

        # 7. 사전학습
        from pretraining import DeepIFSAC_pretrain
        opt = self._make_opt()
        self.model_, _, _ = DeepIFSAC_pretrain(
            self.model_, cat_idxs, X_dict, Y_dict,
            X_train_imp, train_mask,
            continuous_mean_std, imp_continuous_mean_std,
            opt, device,
        )

        self.cat_idxs_ = cat_idxs
        self.con_idxs_ = con_idxs
        self.continuous_mean_std_ = continuous_mean_std
        self.imp_continuous_mean_std_ = imp_continuous_mean_std
        self.device_ = device
        return self

    def transform(self, X):
        """결측값을 보완한 배열을 반환한다.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            결측값(NaN)이 포함된 데이터.

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
            결측 위치가 모델 예측값으로 채워진 배열.
            카테고리 컬럼은 LabelEncoder 정수 코드로 반환됨.
            원본 레이블로 복원하려면 preprocessor_.inverse_transform() 사용.
        """
        import torch
        from augmentations import embed_data_mask

        device = self.device_
        self.model_.eval()

        processed = self.preprocessor_.transform(X)
        X_cat = processed['X_cat']
        X_con = processed['X_con']
        cat_mask = processed['cat_mask']
        con_mask = processed['con_mask']
        X_combined = processed['X_combined']
        nan_mask = processed['nan_mask']

        n = X_combined.shape[0]
        imp_mean, imp_std = self.imp_continuous_mean_std_

        # CLS 토큰 prepend
        cls_col = np.zeros((n, 1), dtype=np.int64)
        cls_mask_arr = np.ones((n, 1), dtype=np.float32)
        X_cat_cls = np.concatenate([cls_col, X_cat], axis=1)
        cat_mask_cls = np.concatenate([cls_mask_arr, cat_mask], axis=1)

        all_con_preds, all_cat_preds = [], []

        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                x_categ = torch.tensor(X_cat_cls[start:end], dtype=torch.long).to(device)
                x_cont = torch.tensor(X_con[start:end], dtype=torch.float32).to(device)
                x_cont_norm = (
                    x_cont - torch.tensor(imp_mean).to(device)
                ) / torch.tensor(imp_std).to(device)
                x_cat_m = torch.tensor(cat_mask_cls[start:end], dtype=torch.long).to(device)
                x_con_m = torch.tensor(con_mask[start:end], dtype=torch.long).to(device)

                _, x_categ_enc, x_cont_enc = embed_data_mask(
                    x_categ, x_cont_norm, x_cat_m, x_con_m, self.model_, False
                )
                cat_outs, con_outs = self.model_(x_categ_enc, x_cont_enc)

                if con_outs:
                    con_pred = torch.cat([c for c in con_outs], dim=1)
                    con_pred_denorm = (
                        con_pred * torch.tensor(imp_std).to(device)
                        + torch.tensor(imp_mean).to(device)
                    )
                    all_con_preds.append(con_pred_denorm.cpu().numpy())

                if cat_outs:
                    batch_cat = []
                    for j in range(1, len(cat_outs)):  # CLS(0) 제외
                        batch_cat.append(torch.argmax(cat_outs[j], dim=1).cpu().numpy())
                    if batch_cat:
                        all_cat_preds.append(np.stack(batch_cat, axis=1))

        X_result = X_combined.copy()

        if all_con_preds:
            con_pred_all = np.concatenate(all_con_preds, axis=0)
            for k, idx in enumerate(self.con_idxs_):
                missing = (nan_mask[:, idx] == 0)
                X_result[missing, idx] = con_pred_all[missing, k]

        if all_cat_preds:
            cat_pred_all = np.concatenate(all_cat_preds, axis=0)
            for k, idx in enumerate(self.cat_idxs_):
                missing = (nan_mask[:, idx] == 0)
                X_result[missing, idx] = cat_pred_all[missing, k]

        return X_result.astype(np.float64)

    def get_features(self, X):
        """Transformer 마지막 레이어 임베딩을 반환한다.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray

        Returns
        -------
        np.ndarray, shape (n_samples, n_features * embedding_size)
            각 샘플의 Transformer 히든 스테이트 (CLS 토큰 제외 후 flatten).
        """
        import torch
        from augmentations import embed_data_mask

        device = self.device_
        self.model_.eval()

        processed = self.preprocessor_.transform(X)
        X_cat = processed['X_cat']
        X_con = processed['X_con']
        cat_mask = processed['cat_mask']
        con_mask = processed['con_mask']
        n = X_cat.shape[0] if len(self.cat_idxs_) > 0 else X_con.shape[0]

        imp_mean, imp_std = self.imp_continuous_mean_std_
        cls_col = np.zeros((n, 1), dtype=np.int64)
        cls_mask_arr = np.ones((n, 1), dtype=np.float32)
        X_cat_cls = np.concatenate([cls_col, X_cat], axis=1)
        cat_mask_cls = np.concatenate([cls_mask_arr, cat_mask], axis=1)

        all_embeddings = []
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                x_categ = torch.tensor(X_cat_cls[start:end], dtype=torch.long).to(device)
                x_cont = torch.tensor(X_con[start:end], dtype=torch.float32).to(device)
                x_cont_norm = (
                    x_cont - torch.tensor(imp_mean).to(device)
                ) / torch.tensor(imp_std).to(device)
                x_cat_m = torch.tensor(cat_mask_cls[start:end], dtype=torch.long).to(device)
                x_con_m = torch.tensor(con_mask[start:end], dtype=torch.long).to(device)

                _, x_categ_enc, x_cont_enc = embed_data_mask(
                    x_categ, x_cont_norm, x_cat_m, x_con_m, self.model_, False
                )
                hidden = self.model_.transformer(x_categ_enc, x_cont_enc)
                # CLS 토큰(index 0) 제외 후 flatten: (batch, n_features * dim)
                features = hidden[:, 1:, :].flatten(1)
                all_embeddings.append(features.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self):
        import torch
        if self.device == 'auto':
            return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)

    def _make_opt(self):
        pt_tasks = self.pt_tasks
        if pt_tasks is None:
            pt_tasks = ['denoising', 'contrastive'] if self.pretrain else ['denoising']
        pt_aug = self.pt_aug if self.pt_aug is not None else (
            ['cutmix'] if self.pretrain else []
        )
        return Namespace(
            dset_id='imputer',
            attentiontype=self.attention_type,
            missing_type=self.missing_type,
            corruption_type=self.corruption_type,
            missing_rate=self.missing_rate,
            dset_seed=str(self.random_state),
            batchsize=self.batch_size,
            pt_tasks=pt_tasks,
            pt_aug=pt_aug,
            pt_aug_lam=0.3,
            pt_projhead_style='diff',
            nce_temp=0.7,
            lam0=0.5,
            lam1=10.0,
            lam2=1.0,
            lam3=10.0,
            vision_dset=False,
            dtask='clf',
            pretrain_epochs=self.pretrain_epochs,
            have_xOrg=False,
        )
