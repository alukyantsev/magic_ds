import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, shapiro, normaltest, zscore
from . visualize import *

# анализируем распределение данных в списке колонок
def analyze_unique(df, columns=[]):

    for c in columns if len(columns) > 0 else df.columns:
        if df[c].dtypes != 'O':
            print('='*20 + ' ' + c + ' (' + str(df[c].nunique()) + ' unique) ' + '='*20)
            value_counts_c = df[c].value_counts()
            print(value_counts_c, '\n')
            if len(columns) == 1:
                return value_counts_c

# анализируем как влияют колонки из списка на целевую переменную
def analyze_target(df, target, columns=[]):

    # функция группировки по признаку с расчетом среднего значения целевой переменной
    mean_target = lambda f: df[[f, target]][~df[target].isnull()].groupby(f, as_index=False).mean().sort_values(by=f, ascending=True)

    for c in columns if len(columns) > 0 else df.columns:
        if c != target and df[c].dtypes != 'O':
            print('='*20 + ' ' + c + ' ' + '='*20)
            mean_target_c = mean_target(c)
            print(mean_target_c, '\n')
            if len(columns) == 1:
                return mean_target_c

# сводная информация по корреляциям
def analyze_corr(df):

    return df.corr()

# анализируем список колонок на нормализацию
def analyze_normal(df, columns=[], skew_score=0.5):

    abnormal_list = []
    for c in columns if len(columns) > 0 else df.columns:
        if df[c].dtypes != 'O':
            df_skew = skew(df[c])
            print('='*20 + ' ' + c + ' ' + '='*20)
            visualize_features_hist(df, c)
            print('mean : ', np.mean(df[c]))
            print('var  : ', np.var(df[c]))
            print('skew : ', df_skew)
            print('kurt : ', kurtosis(df[c]))
            print('shapiro : ', shapiro(df[c]))
            print('normaltest : ', normaltest(df[c]))
            print('\n')
            if abs(df_skew) > skew_score:
                abnormal_list.append(c)

    return abnormal_list

# анализируем список колонок на выбросы
def analyze_outlier_zscore(df, columns=[], limit=2):

    for c in columns if len(columns) > 0 else df.columns:
        if df[c].dtypes in ['float', 'int']:
            z = np.abs(zscore( df[ ~df[c].isnull() ][c] ))
            outlier_list = df.iloc[ np.where(z > limit)[0] ].index.tolist()
            print('='*20 + ' ' + c + ' ' + '='*20)
            visualize_features_boxplot(df, c)
            print(outlier_list, '\n')
            if len(columns) == 1:
                return outlier_list
