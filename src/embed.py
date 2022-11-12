import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
import warnings
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from feature_selection import filter_method as ft
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings("ignore")


def mutual_info(X, y, select_k=10):
    #    mi = mutual_info_classif(X,y)
    #    mi = pd.Series(mi)
    #    mi.index = X.columns
    #    mi.sort_values(ascending=False)

    if select_k >= 1:
        sel_ = SelectKBest(mutual_info_classif, k=select_k).fit(X, y)
        col = X.columns[sel_.get_support()]

    elif 0 < select_k < 1:
        sel_ = SelectPercentile(mutual_info_classif, percentile=select_k * 100).fit(X, y)
        col = X.columns[sel_.get_support()]

    else:
        raise ValueError("select_k must be a positive number")

    return col


# 2018.11.27 edit Chi-square test
def chi_square_test(X, y, select_k=10):
    """
    Compute chi-squared stats between each non-negative feature and class.
    This score should be used to evaluate categorical variables in a classification task
    """
    if select_k >= 1:
        sel_ = SelectKBest(chi2, k=select_k).fit(X, y)
        col = X.columns[sel_.get_support()]
    elif 0 < select_k < 1:
        sel_ = SelectPercentile(chi2, percentile=select_k * 100).fit(X, y)
        col = X.columns[sel_.get_support()]
    else:
        raise ValueError("select_k must be a positive number")

    return col


data = load_breast_cancer()
data = pd.DataFrame(np.c_[data['data'], data['target']],
                  columns= np.append(data['feature_names'], ['target']))
print(data)

X_train, X_test, y_train, y_test = train_test_split(data.drop(labels=['target'], axis=1),
                                                    data.target, test_size=0.2,
                                                    random_state=0)

sfs1 = SFS(RandomForestClassifier(n_jobs=-1,n_estimators=5),
           k_features=10,
           forward=True,
           floating=False,
           verbose=1,
           scoring='roc_auc',
           cv=3)

sfs1 = sfs1.fit(np.array(X_train), y_train)

selected_feat1 = X_train.columns[list(sfs1.k_feature_idx_)]
print(selected_feat1)

mi = ft.mutual_info(X=X_train,y=y_train,select_k=3)
print(mi)

chi = ft.chi_square_test(X=X_train,y=y_train,select_k=3)
print(chi)

