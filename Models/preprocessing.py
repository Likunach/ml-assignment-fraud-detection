import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def load_data():
    train_transaction = pd.read_csv('Data/train_transaction.csv')
    train_identity    = pd.read_csv('Data/train_identity.csv')
    return train_transaction.merge(train_identity, on='TransactionID', how='left')


def load_test_data():
    test_transaction = pd.read_csv('Data/test_transaction.csv')
    test_identity    = pd.read_csv('Data/test_identity.csv')
    return test_transaction.merge(test_identity, on='TransactionID', how='left')


class FraudPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold: float = 0.5):
        self.missing_threshold = missing_threshold

    def fit(self, X, y=None):
        X = X.copy()

        if 'TransactionID' in X.columns:
            X = X.drop(columns=['TransactionID'])

        missing_rate       = X.isnull().mean()
        self.cols_to_drop_ = missing_rate[
            missing_rate > self.missing_threshold
        ].index.tolist()
        X = X.drop(columns=self.cols_to_drop_)

        self.num_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols_ = X.select_dtypes(include='object').columns.tolist()

        self.engineered_from_ = ['TransactionDT', 'TransactionAmt']

        self.num_medians_ = X[self.num_cols_].median()
        self.cat_modes_   = (
            X[self.cat_cols_].mode().iloc[0]
            if self.cat_cols_ else pd.Series(dtype=object)
        )
    
        if self.cat_cols_:
            X_cat_filled = X[self.cat_cols_].fillna(self.cat_modes_)
            self.dummies_cols_ = pd.get_dummies(
                X_cat_filled, drop_first=True
            ).columns.tolist()
        else:
            self.dummies_cols_ = []

        return self

    def transform(self, X, y=None):
        X = X.copy()

        if 'TransactionID' in X.columns:
            X = X.drop(columns=['TransactionID'])

        cols_present = [c for c in self.cols_to_drop_ if c in X.columns]
        X = X.drop(columns=cols_present)


        for col in self.num_cols_:
            if col in X.columns:
                X[col] = X[col].fillna(self.num_medians_[col])


        for col in self.cat_cols_:
            if col in X.columns:
                X[col] = X[col].fillna(self.cat_modes_[col])

        # Feature engineering
        X['hour']                   = X['TransactionDT'] % 86400 // 3600
        X['day_of_week']            = X['TransactionDT'] // 86400 % 7
        X['TransactionAmt_log']     = np.log1p(X['TransactionAmt'])
        X['TransactionAmt_decimal'] = (
            X['TransactionAmt'] - X['TransactionAmt'].astype(int)
        )


        cat_cols_present = [c for c in self.cat_cols_ if c in X.columns]
        if cat_cols_present:
            X = pd.get_dummies(X, columns=cat_cols_present, drop_first=True)


        for col in self.dummies_cols_:
            if col not in X.columns:
                X[col] = 0

        
        engineered_raw = [c for c in self.engineered_from_ if c in X.columns]
        X = X.drop(columns=engineered_raw)

        kept_num_cols = [
            c for c in self.num_cols_
            if c in X.columns and c not in self.engineered_from_
        ]
        new_features = [
            'hour', 'day_of_week', 'TransactionAmt_log', 'TransactionAmt_decimal'
        ]
        final_cols = (
            kept_num_cols
            + new_features
            + [c for c in self.dummies_cols_ if c in X.columns]
        )

       
        seen = set()
        final_cols = [c for c in final_cols if not (c in seen or seen.add(c))]

        return X[final_cols]
        

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features):
        self.selected_features = selected_features

    def fit(self, X, y=None):
        self.features_to_keep_ = [f for f in self.selected_features
                                   if f in X.columns]
        return self

    def transform(self, X, y=None):
        return X[self.features_to_keep_]