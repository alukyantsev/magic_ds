import pandas as pd
import numpy as np

# преобразуем список колонок типа uint8 в int64 (возникают в результате one-hot-encoding)
def transform_int64(df, columns=[]):

    df0 = df.copy()
    for c in columns if len(columns) > 0 else list(df.select_dtypes(include=['uint8']).columns):
        df0[c] = df[c].astype('int64')
    return df0

# преобразуем список колонок типа number в float64
def transform_float64(df, columns=[]):

    df0 = df.copy()
    for c in columns if len(columns) > 0 else list(df.select_dtypes(include=['number']).columns):
        df0[c] = df[c].astype('float64')
    return df0