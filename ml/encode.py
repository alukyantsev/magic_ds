import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# кодируем список колонок через one-hot-encoding
def encode_ohe(df, columns):

    for c in columns:
        df0 = pd.get_dummies(df[c], prefix=c, dummy_na=False)
        df = pd.concat([df, df0], axis=1)
    return df

# кодируем список колонок через label-encoding
def encode_le(df, columns):

    df0 = df.copy()
    for c in columns:
        label = LabelEncoder()
        label.fit(df[c].drop_duplicates())
        df0[c] = label.transform(df[c])
    return df0
