import pandas as pd
import numpy as np
import xgboost as xgb
from . common import *
from . drop import *
from . encode import *
from . select import select_eli5
from . split import *

# делаем магию и по начальному датасету делаем первый прогноз по модели xgboost
def magic_pred(df, target, replace={}):

    df0 = df.copy()
    # убираем пропуски
    if len(replace):
        df0 = df0.replace(replace)
    df0 = drop_gap(df0)
    # числовые значения привели к float
    df0 = transform_float64(df0, [target])
    df0 = transform_float64(df0)
    # нечисловые значения кодируем в label encoder
    df0 = encode_le(df0, df0.select_dtypes(exclude=['number']))
    # разбиваем выборку
    X_train, X_valid, X_test, y_train, y_valid = \
        split_df(df0, target, prc_train=99.999, prc_valid=0.001, prc_test=0)
    # запускаем модель
    model = select_eli5(
        X_train, y_train,
        #xgb.XGBRegressor(learning_rate=0.01, max_depth=5, min_child_weight=3, n_estimators=300, random_state=42),
        xgb.XGBClassifier(learning_rate=0.01, max_depth=5, min_child_weight=3, n_estimators=300, random_state=42, objective='binary:logistic', eval_metric='mlogloss'),
        mute=1, limit=100)
    # делаем предикт
    if(len(X_test)):
        y_pred = model.predict(X_test)
        return model, y_pred
    else:
        return model
