import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# визуализация пропусков
def visualize_gap(df, size=(20,12)):

    fig, ax = plt.subplots(figsize=size)
    sns_heatmap = sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.show()

# визуализация матрицы корреляции
def visualize_corr(df, size=(20,12)):

    fig, ax = plt.subplots(figsize=size)
    sns_heatmap = sns.heatmap(df.corr(), annot = True, vmin=-1, vmax=1, center=0, cmap='coolwarm')
    plt.show()

# визуализация списка колонок гистограммой
def visualize_features_hist(df, columns):

    plt.style.use('ggplot')
    plt.hist(df[columns], bins=60)
    plt.show()

# визуализация лучших по скорингу колонок
def visualize_features_select(df_scores, df_columns, df_bool=pd.DataFrame(), \
    scores_columns='Score', spec_columns='Specs', bool_columns='Using', limit=10):

    bool_active = len(df_bool) > 0
    concat_list = [df_columns, df_scores, df_bool] if bool_active else [df_columns, df_scores]
    df0_list = [spec_columns, scores_columns, bool_columns] if bool_active else [spec_columns, scores_columns]
    df0 = pd.concat(concat_list, axis=1)
    df0.columns = df0_list
    top_limit = df0.nlargest(limit, scores_columns)
    top_limit.set_index(spec_columns).sort_values(by=scores_columns, ascending=True).plot(kind='barh')
    plt.show()
    return top_limit

# визуализация кривой обучения
def visualize_learning_curves(train_errors, valid_errors, scoring_name='Scoring'):

    plt.plot(train_errors, 'r-+', linewidth=2, label='train')
    plt.plot(valid_errors, 'b-', linewidth=3, label='valid')
    plt.xlabel('Train Size')
    plt.ylabel(scoring_name)
    plt.legend()
    plt.show()
