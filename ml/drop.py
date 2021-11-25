import pandas as pd
import numpy as np
from . gap import *

# удаляет колонки
def drop_features(df, columns):

    df0 = df.copy()
    df0 = df0.drop(columns = columns)
    return df0

# удаляет колонки с пропусками более limit %
def drop_gap(df, columns=[], limit=50):

    gap_df = gap_info(df)
    gap_columns = columns if len(columns) > 0 else list(gap_df[ gap_df['% of Total Values'] > limit ].index)
    print('We will remove %d columns with limit %d%%.' % (len(gap_columns), limit))
    df0 = df.copy()
    df0 = df0.drop(columns = gap_columns)
    return df0
