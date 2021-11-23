import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler, PolynomialFeatures
from . fillna import *

# считаем логарифм списка колонок
def transform_log(df, columns):

    df0 = df.copy()
    for c in columns:
        df0[c] = df[c].map(lambda x: np.log(x) if x > 0 else np.nan) \
                      .replace({np.inf: np.nan, -np.inf: np.nan})
    
    df0 = fillna_median(df0, columns)
    return df0

# проводим нормализацию списка колонок
def transform_normalize(df, columns, method='yeo-johnson'):

    transformer = PowerTransformer(method=method, standardize=False)
    df0 = df.copy()
    for c in columns:
        df0[c] = transformer.fit_transform(df[c].values.reshape(df.shape[0], -1))
    return df0

# проводим стандартизацию списка колонок
def transform_scaler(df, columns):

    scaler = StandardScaler()
    df0 = df.copy()
    df0[columns] = scaler.fit_transform(df[columns])
    return df0

# проводим логарифмизацию, нормализацию, а потом стандартизацию списка колонок
#
# method='yeo-johnson' - works with positive and negative values
# method='box-cox' - only works with strictly positive values
#
def transform_features(df, columns_log=[], columns_normalize=[], columns_scaler=[], method='yeo-johnson'):

    df0 = transform_log(df, columns_log) if len(columns_log) > 0 else df
    df1 = transform_normalize(df0, columns_normalize) if len(columns_normalize) > 0 else df0
    df2 = transform_scaler(df1, columns_scaler) if len(columns_scaler) > 0 else df1
    return df2

# делаем полиномиальное преобразование
def transform_poly(X, degree=2):

    X0 = X.copy()
    poly = PolynomialFeatures(degree=degree)
    X0 = poly.fit_transform(X0)
    return X0