# from sklearn.base import BaseEstimator, TransformerMixin
# from datetime import datetime
# import re
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import FunctionTransformer
# from sklearn import set_config
# from sklearn.utils._tags import Tags, TransformerTags, TargetTags

# set_config(transform_output="pandas")

# # =========================================================
# # Helper
# # =========================================================
# def ensure_dataframe(X, columns=None):
#     if isinstance(X, np.ndarray):
#         return pd.DataFrame(X, columns=columns)
#     return X

# def sklearn_tags():
#     return Tags(
#         estimator_type="transformer",
#         target_tags=TargetTags(required=False),
#         transformer_tags=TransformerTags(),
#     )

# # =========================================================
# # Mixed Variable Transformers
# # =========================================================
# class TermTransformer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         X = ensure_dataframe(X)
#         X = X.copy()
#         X["term"] = X["term"].str.replace(" months", "", regex=False).astype(float)
#         return X

#     def __sklearn_tags__(self):
#         return sklearn_tags()



# class IssueDTransformer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         X = ensure_dataframe(X)
#         X = X.copy()
#         X["issue_d"] = pd.to_datetime(X["issue_d"], format="%b-%Y", errors="coerce")
#         X["issue_d"] = X["issue_d"].dt.strftime("%b")
#         return X

#     def __sklearn_tags__(self):
#         return sklearn_tags()


# class EmpLengthTransformer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         X = ensure_dataframe(X)
#         X = X.copy()

#         X["emp_length"] = X["emp_length"].replace("10+ years", "10")
#         X["emp_length"] = X["emp_length"].replace("< 1 year", "0")

#         X["emp_length"] = X["emp_length"].astype(str).apply(
#             lambda x: float(re.findall(r"\d+", x)[0])
#             if re.findall(r"\d+", x)
#             else np.nan
#         )
#         return X

#     def __sklearn_tags__(self):
#         return sklearn_tags()


# class EarliestCrLineTransformer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         X = ensure_dataframe(X)
#         X = X.copy()
#         X["earliest_cr_line"] = pd.to_datetime(
#             X["earliest_cr_line"], format="%b-%Y", errors="coerce"
#         ).dt.year
#         return X

#     def __sklearn_tags__(self):
#         return sklearn_tags()


# # =========================================================
# # Feature Engineering
# # =========================================================
# class NewFeatureGenerator(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         X = ensure_dataframe(X)

#         current_year = 2015

#         cr_history = (
#             current_year - X["earliest_cr_line"]
#             if "earliest_cr_line" in X.columns
#             else 0
#         )

#         installment_to_income_ratio = (
#             X.get("installment", 0) / X.get("annual_inc", 1)
#         ).replace([np.inf, -np.inf], np.nan).fillna(0)

#         loan_to_inc_ratio = (
#             X.get("loan_amnt", 0) / X.get("annual_inc", 1)
#         ).replace([np.inf, -np.inf], np.nan).fillna(0)

#         return pd.DataFrame(
#             {
#                 "cr_history": cr_history,
#                 "installment_to_income_ratio": installment_to_income_ratio,
#                 "loan_to_inc_ratio": loan_to_inc_ratio,
#             },
#             index=X.index,
#         )

#     def __sklearn_tags__(self):
#         return sklearn_tags()


# class NewFeatureAddingTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.generator = NewFeatureGenerator()
#         self.col_map = {
#             "earliest_cr_line": [
#                 "earliest_cr_line_pipeline__earliest_cr_line",
#                 "earliest_cr_line",
#             ],
#             "installment": [
#                 "installment_passthrough__installment",
#                 "installment",
#             ],
#             "annual_inc": [
#                 "annual_inc_pipeline__annual_inc",
#                 "annual_inc",
#             ],
#             "loan_amnt": [
#                 "loan_amnt_pipeline__loan_amnt",
#                 "loan_amnt",
#             ],
#         }

#     def _resolve(self, X, names):
#         for n in names:
#             if n in X.columns:
#                 return n
#         return None

#     def fit(self, X, y=None):
#         X = ensure_dataframe(X)
#         X_gen = pd.DataFrame(index=X.index)

#         for k, v in self.col_map.items():
#             col = self._resolve(X, v)
#             if col is None:
#                 raise ValueError(f"Missing columns {v}. Available: {list(X.columns)}")
#             X_gen[k] = X[col]

#         self.generator.fit(X_gen, y)
#         return self

#     def transform(self, X, y=None):
#         X = ensure_dataframe(X)
#         X_gen = pd.DataFrame(index=X.index)

#         for k, v in self.col_map.items():
#             col = self._resolve(X, v)
#             if col is None:
#                 raise ValueError(f"Missing columns {v}. Available: {list(X.columns)}")
#             X_gen[k] = X[col]

#         new_feats = self.generator.transform(X_gen)
#         return pd.concat([X, new_feats], axis=1)

#     def __sklearn_tags__(self):
#         return sklearn_tags()


# # =========================================================
# # Outlier Handling
# # =========================================================
# class OutlierCapper(BaseEstimator, TransformerMixin):
#     def __init__(self, caps=None):
#         # ALWAYS define the attribute
#         self.caps = caps

#     def fit(self, X, y=None):
#         # Set defaults during fit if caps not provided
#         if self.caps is None:
#             self.caps = {
#                 "annual_inc": 260000,
#                 "dti": 100,
#                 "bc_util": 100,
#                 "revol_util": 100,
#             }
#         return self

#     def transform(self, X, y=None):
#         X = ensure_dataframe(X)
#         X = X.copy()

#         for col, cap in self.caps.items():
#             if col in X.columns:
#                 X[col] = np.minimum(X[col], cap)
#         return X

#     def __sklearn_tags__(self):
#         return sklearn_tags()


# # =========================================================
# # Binning Functions
# # =========================================================
# def bin_pub_rec(X):
#     X = ensure_dataframe(X)
#     return pd.cut(X.iloc[:, 0], [-0.1, 0.9, 2.9, 5.9, np.inf], labels=False).to_frame(
#         X.columns[0]
#     )


# def bin_emp_length(X):
#     X = ensure_dataframe(X)
#     return pd.cut(X.iloc[:, 0], [-0.1, 3, 9, np.inf], labels=False).to_frame(
#         X.columns[0]
#     )


# def bin_delinq_2yrs(X):
#     X = ensure_dataframe(X)
#     return pd.cut(X.iloc[:, 0], [-0.1, 3, 11, 19, np.inf], labels=False).to_frame(
#         X.columns[0]
#     )


# def bin_fico_range_low(X):
#     X = ensure_dataframe(X)
#     return pd.cut(X.iloc[:, 0], [-1, 649, 699, 749, np.inf], labels=False).to_frame(
#         X.columns[0]
#     )


# # =========================================================
# # Binarizers
# # =========================================================
# def binarize_revol_util(X):
#     X = ensure_dataframe(X)
#     return (X.iloc[:, 0] > 75).astype(int).to_frame(X.columns[0])


# def binarize_bc_util(X):
#     X = ensure_dataframe(X)
#     return (X.iloc[:, 0] > 80).astype(int).to_frame(X.columns[0])


# # =========================================================
# # Math Transform
# # =========================================================
# def apply_log1p_df(X):
#     X = ensure_dataframe(X)
#     return np.log1p(X)


# # =========================================================
# # Categorical Cleanup
# # =========================================================
# def transform_home_ownership(X):
#     X = ensure_dataframe(X)
#     valid = ["MORTGAGE", "RENT", "OWN"]
#     return X.iloc[:, 0].apply(lambda x: x if x in valid else "Other").to_frame(X.columns[0])


# def transform_purpose(X):
#     X = ensure_dataframe(X)
#     valid = ["debt_consolidation", "credit_card"]
#     return X.iloc[:, 0].apply(lambda x: x if x in valid else "Other").to_frame(X.columns[0])


# # =========================================================
# # Column Dropper
# # =========================================================
# class ColumnDroppingTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, columns):
#         self.columns = columns

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         X = ensure_dataframe(X)
#         return X.drop(columns=[c for c in self.columns if c in X.columns])

#     def __sklearn_tags__(self):
#         return sklearn_tags()


# # =========================================================
# # Model Wrapper Transformers
# # =========================================================
# class PredictProbaTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, model=None):
#         self.model = model

#     def fit(self, X, y=None):
#         return self

#     def _get_model(self):
#         for attr in ["model", "estimator", "clf", "classifier"]:
#             if hasattr(self, attr) and getattr(self, attr) is not None:
#                 return getattr(self, attr)
#         raise AttributeError(
#             "PredictProbaTransformer: underlying model not found. "
#             "Expected attribute 'model', 'estimator', 'clf', or 'classifier'."
#         )

#     def transform(self, X, y=None):
#         X = ensure_dataframe(X)
#         model = self._get_model()
#         proba = model.predict_proba(X)
#         return pd.DataFrame(
#             proba,
#             index=X.index,
#             columns=[f"class_{i}_proba" for i in range(proba.shape[1])]
#         )

#     def __sklearn_tags__(self):
#         return sklearn_tags()

# class RegPredictProbaTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, model=None):
#         self.model = model

#     def fit(self, X, y=None):
#         return self

#     def _get_model(self):
#         for attr in ["model", "estimator", "regressor"]:
#             if hasattr(self, attr) and getattr(self, attr) is not None:
#                 return getattr(self, attr)
#         raise AttributeError(
#             "RegPredictProbaTransformer: underlying model not found. "
#             "Expected attribute 'model', 'estimator', or 'regressor'."
#         )

#     def transform(self, X, y=None):
#         X = ensure_dataframe(X)
#         model = self._get_model()
#         preds = model.predict(X)
#         return pd.DataFrame(
#             preds, index=X.index, columns=["regression_prediction"]
#         )

#     def __sklearn_tags__(self):
#         return sklearn_tags()

# class IntRatePredictProbaTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, model=None):
#         self.model = model

#     def fit(self, X, y=None):
#         return self

#     def _get_model(self):
#         for attr in ["model", "estimator", "regressor"]:
#             if hasattr(self, attr) and getattr(self, attr) is not None:
#                 return getattr(self, attr)
#         raise AttributeError(
#             "IntRatePredictProbaTransformer: underlying model not found. "
#             "Expected attribute 'model', 'estimator', or 'regressor'."
#         )

#     def transform(self, X, y=None):
#         X = ensure_dataframe(X)
#         model = self._get_model()
#         preds = model.predict(X)
#         return pd.DataFrame(
#             preds, index=X.index, columns=["int_rate_prediction"]
#         )

#     def __sklearn_tags__(self):
#         return sklearn_tags()
# # =========================================================
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import re
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import _tags as sklearn_tags
import pandas as pd
import numpy as np

def ensure_dataframe(X, columns=None):
    if isinstance(X, pd.DataFrame):
        X = X.copy()
        X.columns = X.columns.astype(str)
        return X

    if isinstance(X, pd.Series):
        return X.to_frame()

    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if columns is None:
            columns = [str(i) for i in range(X.shape[1])]

        return pd.DataFrame(X, columns=[str(c) for c in columns])

    df = pd.DataFrame(X)
    df.columns = df.columns.astype(str)
    return df

# ------------------------------------------------------------------
# sklearn compatibility patch (older pickles + sklearn >=1.4)
# ------------------------------------------------------------------
if not getattr(BaseEstimator, '_patched_sklearn_tags', False):
    BaseEstimator.__sklearn_tags__ = lambda self: sklearn_tags.Tags(
        estimator_type=None,
        target_tags=sklearn_tags.TargetTags(required=False),
        transformer_tags=sklearn_tags.TransformerTags(),
    )
    BaseEstimator._patched_sklearn_tags = True

#Mixed Variables
class TermTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = ensure_dataframe(X).copy()
        X_transformed['term'] = X_transformed['term'].str.replace(' months', '', regex=False).astype(float)
        return X_transformed

class IssueDTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = ensure_dataframe(X).copy()
        X_transformed['issue_d'] = pd.to_datetime(X_transformed['issue_d'], format='%b-%Y')
        X_transformed['issue_d'] = X_transformed['issue_d'].dt.strftime('%b')
        return X_transformed

class EmpLengthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = ensure_dataframe(X).copy()
        X_transformed['emp_length'] = X_transformed['emp_length'].replace('10+ years', '10')
        X_transformed['emp_length'] = X_transformed['emp_length'].replace('< 1 year', '0')
        X_transformed['emp_length'] = X_transformed['emp_length'].astype(str).apply(lambda x: float(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else (x if x == 'nan' else None))
        X_transformed['emp_length'] = X_transformed['emp_length'].replace('nan', np.nan).astype(float)
        return X_transformed

class EarliestCrLineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = ensure_dataframe(X).copy()
        X_transformed['earliest_cr_line'] = pd.to_datetime(
            X_transformed['earliest_cr_line'], format='%b-%Y', errors='coerce'
        ).dt.year
        return X_transformed

#New Feature Extraction
class NewFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = ensure_dataframe(X)
        # X is expected to have 'earliest_cr_line' (numeric year), 'installment', 'annual_inc', 'loan_amnt'
        current_year = 2015

        cr_history = pd.Series(np.nan, index=X.index)
        if 'earliest_cr_line' in X.columns and pd.api.types.is_numeric_dtype(X['earliest_cr_line']):
            cr_history = current_year - X['earliest_cr_line']

        installment_to_income_ratio = (X['installment'] / X['annual_inc'])
        installment_to_income_ratio = installment_to_income_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

        loan_to_inc_ratio = (X['loan_amnt'] / X['annual_inc'])
        loan_to_inc_ratio = loan_to_inc_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Return  DF with only the new features
        return pd.DataFrame({
            'cr_history': cr_history,
            'installment_to_income_ratio': installment_to_income_ratio,
            'loan_to_inc_ratio': loan_to_inc_ratio
        }, index=X.index)

class NewFeatureAddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.new_feature_generator = NewFeatureGenerator()
        self.column_name_map = {
            'earliest_cr_line': 'earliest_cr_line_pipeline__earliest_cr_line',
            'installment': 'installment_passthrough__installment',
            'annual_inc': 'annual_inc_pipeline__annual_inc',
            'loan_amnt': 'loan_amnt_pipeline__loan_amnt'
        }

    def fit(self, X, y=None):
        X = ensure_dataframe(X)
        X_for_generation_fit = pd.DataFrame(index=X.index)
        for gen_col, pipeline_col in self.column_name_map.items():
            if pipeline_col in X.columns:
                X_for_generation_fit[gen_col] = X[pipeline_col]
            elif gen_col in X.columns:
                X_for_generation_fit[gen_col] = X[gen_col]
            else:
                X_for_generation_fit[gen_col] = np.nan

        self.new_feature_generator.fit(X_for_generation_fit, y)
        return self

    def transform(self, X, y=None):
        X = ensure_dataframe(X)
        X_for_generation_transform = pd.DataFrame(index=X.index)
        for gen_col, pipeline_col in self.column_name_map.items():
            if pipeline_col in X.columns:
                X_for_generation_transform[gen_col] = X[pipeline_col]
            elif gen_col in X.columns:
                X_for_generation_transform[gen_col] = X[gen_col]
            else:
                X_for_generation_transform[gen_col] = np.nan

        new_features_df = self.new_feature_generator.transform(X_for_generation_transform)

        # Concatenate new features with the existing DataFrame X (which contains all processed original columns)
        Xt_with_new_features = pd.concat([X, new_features_df], axis=1)
        Xt_with_new_features.columns = Xt_with_new_features.columns.astype(str)
        return Xt_with_new_features

#Outlier
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, capping_thresholds=None):
        if capping_thresholds is None:
            self.capping_thresholds = {
                'annual_inc': 260000.0,
                'dti': 100.0,
                'bc_util': 100.0,
                'revol_util': 100.0
            }
        else:
            self.capping_thresholds = capping_thresholds
        # Backward-compat alias for older pickles expecting `caps`
        self.caps = self.capping_thresholds

    def fit(self, X, y=None):
        if not hasattr(self, 'caps') or self.caps is None:
            self.caps = self.capping_thresholds
        return self

    def transform(self, X, y=None):
        X_transformed = ensure_dataframe(X).copy()
        thresholds = getattr(self, 'caps', None) or self.capping_thresholds
        for col, threshold in thresholds.items():
            if col in X_transformed.columns:
                #winsorization (capping) for upper outliers
                X_transformed[col] = np.where(X_transformed[col] > threshold, threshold, X_transformed[col])
        return X_transformed

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Ensure both attributes exist after unpickling
        if not hasattr(self, 'capping_thresholds') and hasattr(self, 'caps'):
            self.capping_thresholds = self.caps
        if not hasattr(self, 'caps') and hasattr(self, 'capping_thresholds'):
            self.caps = self.capping_thresholds

#Binning
def _to_array(X):
    """Extract a 1-D numpy array from whatever input type."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, 0].values
    if isinstance(X, pd.Series):
        return X.values
    arr = np.asarray(X)
    return arr.ravel()

def bin_pub_rec(X_df):
    arr = _to_array(X_df)
    bins = [-0.1, 0.9, 2.9, 5.9, np.inf]
    return pd.cut(pd.Series(arr), bins=bins, labels=False, right=True, include_lowest=True).values.reshape(-1, 1)

def bin_emp_length(X_df):
    arr = _to_array(X_df)
    bins = [-0.1, 3, 9, np.inf]
    return pd.cut(pd.Series(arr), bins=bins, labels=False, right=True, include_lowest=True).values.reshape(-1, 1)

def bin_delinq_2yrs(X_df):
    arr = _to_array(X_df)
    bins = [-0.1, 3, 11, 19, np.inf]
    return pd.cut(pd.Series(arr), bins=bins, labels=False, right=True, include_lowest=True).values.reshape(-1, 1)

def bin_fico_range_low(X_df):
    arr = _to_array(X_df)
    bins = [-1, 649, 699, 749, np.inf]
    return pd.cut(pd.Series(arr), bins=bins, labels=False, right=True, include_lowest=True).values.reshape(-1, 1)

#Binarizer
def binarize_revol_util(X_df):
    arr = _to_array(X_df)
    return (arr > 75).astype(int).reshape(-1, 1)

def binarize_bc_util(X_df):
    arr = _to_array(X_df)
    return (arr > 80).astype(int).reshape(-1, 1)

#Mathematical transformations
def apply_log1p_df(X_df):
    if isinstance(X_df, np.ndarray):
        return np.log1p(X_df)
    return np.log1p(X_df.values).reshape(X_df.shape)

#Categorical Column transformations
def transform_home_ownership(X_df):
    arr = _to_array(X_df)
    valid_ownership = ['MORTGAGE', 'RENT', 'OWN']
    return np.array([x if x in valid_ownership else 'Other' for x in arr], dtype=object).reshape(-1, 1)

def transform_purpose(X_df):
    arr = _to_array(X_df)
    valid_purpose = ['debt_consolidation', 'credit_card']
    return np.array([x if x in valid_purpose else 'Other' for x in arr], dtype=object).reshape(-1, 1)

#cols to drop
class ColumnDroppingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = ensure_dataframe(X)
        # columns_to_drop may use original names; also try string versions
        drop_set = set(self.columns_to_drop) | set(str(c) for c in self.columns_to_drop)
        cols_to_actually_drop = [col for col in X.columns if col in drop_set]
        return X.drop(columns=cols_to_actually_drop)

class PredictProbaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        proba = self.estimator.predict_proba(X)
        # Return only the positive-class probability (column 1) so that
        # FeatureUnion produces one column per base model, matching the
        # meta-model's expected feature count.
        arr = np.asarray(proba)
        return arr[:, 1].reshape(-1, 1)


class RegPredictProbaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pred = self.estimator.predict(X)
        if isinstance(pred, (pd.Series, pd.DataFrame)):
            pred = pred.values
        return np.asarray(pred).reshape(-1, 1)


class IntRatePredictProbaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pred = self.estimator.predict(X)
        if isinstance(pred, (pd.Series, pd.DataFrame)):
            pred = pred.values
        return np.asarray(pred).reshape(-1, 1)