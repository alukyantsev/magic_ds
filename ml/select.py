import pandas as pd
import numpy as np
import eli5
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from . visualize import *

# выводим список топ колонок, влияющих на целевую переменную через SelectKBest
# принимает только значения >= 0
def select_univariate(X, y, limit=10):

    select = SelectKBest(score_func=chi2, k=limit).fit(X,y)
    df_scores = pd.DataFrame(select.scores_)
    df_columns = pd.DataFrame(X.columns)
    return visualize_features_select(df_scores, df_columns, limit=limit)

# выводим список топ колонок, влияющих на целевую переменную через feature_importances_
def select_importances(X, y, model, limit=10):

    model.fit(X, y)
    df_scores = pd.DataFrame(model.feature_importances_ * 1000)
    df_columns = pd.DataFrame(X.columns)    
    return visualize_features_select(df_scores, df_columns, limit=limit)

# выводим список топ колонок, влияющих на целевую переменную через SelectFromModel
def select_model(X, y, model, limit=10):

    select = SelectFromModel(model)
    select.fit_transform(X, y)
    df_bool = pd.DataFrame(select.get_support())
    df_scores = pd.DataFrame(select.estimator_.coef_).abs()
    df_columns = pd.DataFrame(X.columns)
    return visualize_features_select(df_scores, df_columns, df_bool, limit=limit)

# выводим список топ колонок, влияющих на целевую переменную через eli5
def select_eli5(X, y, model, limit=10, mute=0):

    model.fit(X, y)
    display(eli5.explain_weights(model, top=limit))

    if not mute:
        df0 = eli5.explain_weights_df(model, top=limit)
        df_scores = df0['weight'] * 100
        df_columns = df0['feature']
        return visualize_features_select(df_scores, df_columns, limit=limit)
    else:
        return model

# выводим список топ колонок, влияющих на целевую переменную через corr
def select_corr(df, target, limit=10):

    corr = df.corr()[target].drop(labels=[target]).abs().map(lambda x: x * 1000)
    df_scores = pd.DataFrame(corr.values)
    df_columns = pd.DataFrame(corr.index)
    return visualize_features_select(df_scores, df_columns, limit=limit)