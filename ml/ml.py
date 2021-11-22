# визуализация пропусков
def gap_visualize(df, size=(20,12)):

    fig, ax = plt.subplots(figsize=size)
    sns_heatmap = sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.show()

# сводная информация по пропускам
def gap_info(df):

    mis_val = df.isnull().sum()
    mis_val_percent = 100 * mis_val / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table. \
        rename( columns = {0 : 'Missing Values', 1 : '% of Total Values'} )
    mis_val_table_ren_columns = mis_val_table_ren_columns[ mis_val_table_ren_columns.iloc[:,1] != 0 ]. \
        sort_values('% of Total Values', ascending=False).round(1)
    print('Your selected dataframe has ' + str(df.shape[1]) + ' columns.\n'      
          'There are ' + str(mis_val_table_ren_columns.shape[0]) + ' columns that have missing values.')
    return mis_val_table_ren_columns

# возвращает перечень колонок со значением < 0
def gap_negative(df):

    list_negative = []
    for c in df.select_dtypes(include=['number']).columns:
        count = df[df[c]<0][c].count()
        print('%d\t- %s' % (count, c))
        if count > 0:
            list_negative.append(c)
    return list_negative

# удаляем колонки с пропусками более limit %
def gap_drop(df, columns=[], limit=50):

    gap_df = gap_info(df)
    gap_columns = columns if len(columns) > 0 else list(gap_df[ gap_df['% of Total Values'] > limit ].index)
    print('We will remove %d columns with limit %d%%.' % (len(gap_columns), limit))
    df0 = df.copy()
    df0 = df0.drop(columns = gap_columns)
    return df0

# визуализация матрицы корреляции
def corr_visualize(df, size=(40,24)):

    fig, ax = plt.subplots(figsize=size)
    sns_heatmap = sns.heatmap(df.corr(), annot = True, vmin=-1, vmax=1, center=0, cmap='coolwarm')
    plt.show()

# сводная информация по корреляциям
def corr_info(df):

    return df.corr()

# выводим распределение данных в списке колонок
def features_unique(df, columns=[]):

    for c in columns if len(columns) > 0 else df.columns:
        print('='*20 + ' ' + c + ' (' + str(df[c].nunique()) + ' unique) ' + '='*20)
        value_counts_c = df[c].value_counts()
        print(value_counts_c, '\n')
        if len(columns) == 1:
            return value_counts_c

# смотрим как влияют колонки из списка на целевую переменную
def features_target(df, target, columns=[]):

    # функция группировки по признаку с расчетом среднего значения целевой переменной
    mean_target = lambda f: df[[f, target]][~df[target].isnull()].groupby(f, as_index=False).mean().sort_values(by=f, ascending=True)

    for c in columns if len(columns) > 0 else df.columns:
        if c != target:
            print('='*20 + ' ' + c + ' ' + '='*20)
            mean_target_c = mean_target(c)
            print(mean_target_c, '\n')
            if len(columns) == 1:
                return mean_target_c

# удаляем колонки
def features_drop(df, columns):

    df0 = df.copy()
    df0 = df0.drop(columns = columns)
    return df0

# кодируем список колонок через one-hot-encoding
def features_ohe(df, columns):

    for c in columns:
        df0 = pd.get_dummies(df[c], prefix=c, dummy_na=False)
        df = pd.concat([df, df0], axis=1)
    return df

# кодируем список колонок через label-encoding
def features_le(df, columns):

    df0 = df.copy()
    for c in columns:
        label = LabelEncoder()
        label.fit(df[c].drop_duplicates())
        df0[c] = label.transform(df[c])
    return df0

# преобразуем список колонок типа uint8 в int64 (возникают в результате one-hot-encoding)
def features_transform_int64(df, columns=[]):

    df0 = df.copy()
    for c in columns if len(columns) > 0 else list(df.select_dtypes(include=['uint8']).columns):
        df0[c] = df[c].astype('int64')
    return df0

# преобразуем список колонок типа number в float64
def features_transform_float64(df, columns=[]):

    df0 = df.copy()
    for c in columns if len(columns) > 0 else list(df.select_dtypes(include=['number']).columns):
        df0[c] = df[c].astype('float64')
    return df0

# заполняем пропуски в списке колонок средним значением
def features_fillna_mean(df, columns):

    df0 = df.copy()
    for c in columns:
        df0[c] = df[c].fillna( df[c].mean() )
    return df0

# заполняем пропуски в списке колонок медианным значением
def features_fillna_median(df, columns):

    df0 = df.copy()
    for c in columns:
        df0[c] = df[c].fillna( df[c].median() )
    return df0

# заполняем отрицательные значения в списке колонок медианным
def features_fillna_negative(df, columns):

    df0 = df.copy()
    for c in columns:
        median = df[c].median()
        df0[c] = df[c].map(lambda x: median if x<0 else x)
    return df0

# заполняем пропуски в списке колонок модовым значением
def features_fillna_mode(df, columns):

    df0 = df.copy()
    for c in columns:
        df0[c] = df[c].fillna( df[c].mode().values[0] )
    return df0

# заполняем пропуски в списке колонок переданным значением
def features_fillna_value(df, columns, value=-1):

    df0 = df.copy()
    for c in columns:
        df0[c] = df[c].fillna( value )
    return df0

# заполняем пропуски в списке колонок NaN
def features_fillna_nan(df, columns):

    df0 = df.copy()
    for c in columns:
        df0[c] = df[c].fillna( np.nan )
    return df0

# анализируем список колонок на нормализацию
def features_analyze_normal(df, columns=[]):

    plt.style.use('ggplot')
    for c in columns if len(columns) > 0 else df.columns:
        print('='*20 + ' ' + c + ' ' + '='*20)
        plt.hist(df[c], bins=60)
        plt.show()
        print('mean : ', np.mean(df[c]))
        print('var  : ', np.var(df[c]))
        print('skew : ', skew(df[c]))
        print('kurt : ', kurtosis(df[c]))
        print('shapiro : ', shapiro(df[c]))
        print('normaltest : ', normaltest(df[c]))
        print('\n')

# проводим нормализацию списка колонок
def features_normalize(df, columns, method='yeo-johnson'):

    transformer = PowerTransformer(method=method, standardize=False)
    df0 = df.copy()
    for c in columns:
        df0[c] = transformer.fit_transform(df[c].values.reshape(df.shape[0], -1))
    return df0

# проводим стандартизацию списка колонок
def features_scaler(df, columns):

    scaler = StandardScaler()
    df0 = df.copy()
    df0[columns] = scaler.fit_transform(df[columns])
    return df0

# считаем логарифм списка колонок
def features_log(df, columns):

    df0 = df.copy()
    for c in columns:
        df0[c] = df[c].map(lambda x: np.log(x) if x > 0 else np.nan) \
                      .replace({np.inf: np.nan, -np.inf: np.nan})
    
    df0 = features_fillna_median(df0, columns)
    return df0

# проводим логарифмизацию, нормализацию, а потом стандартизацию списка колонок
#
# method='yeo-johnson' - works with positive and negative values
# method='box-cox' - only works with strictly positive values
#
def features_log_normalize_scaler(df, columns_log=[], columns_normalize=[], columns_scaler=[], method='yeo-johnson'):

    df0 = features_log(df, columns_log) if len(columns_log) > 0 else df
    df1 = features_normalize(df0, columns_normalize) if len(columns_normalize) > 0 else df0
    df2 = features_scaler(df1, columns_scaler) if len(columns_scaler) > 0 else df1
    return df2

# визуализируем информацию о лучших колонках
def features_select_visualize(df_scores, df_columns, df_bool=pd.DataFrame(), \
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

# выводим список топ колонок, влияющих на целевую переменную через SelectKBest
# принимает только значения >= 0
def features_select_univariate(X, y, limit=10):

    select = SelectKBest(score_func=chi2, k=limit).fit(X,y)
    df_scores = pd.DataFrame(select.scores_)
    df_columns = pd.DataFrame(X.columns)
    return features_select_visualize(df_scores, df_columns, limit=limit)

# выводим список топ колонок, влияющих на целевую переменную через feature_importances_
def features_select_importances(X, y, model, limit=10):

    model.fit(X, y)
    df_scores = pd.DataFrame(model.feature_importances_ * 1000)
    df_columns = pd.DataFrame(X.columns)    
    return features_select_visualize(df_scores, df_columns, limit=limit)

# выводим список топ колонок, влияющих на целевую переменную через SelectFromModel
def features_select_model(X, y, model, limit=10):

    select = SelectFromModel(model)
    select.fit_transform(X, y)
    df_bool = pd.DataFrame(select.get_support())
    df_scores = pd.DataFrame(select.estimator_.coef_).abs()
    df_columns = pd.DataFrame(X.columns)
    return features_select_visualize(df_scores, df_columns, df_bool, limit=limit)

# выводим список топ колонок, влияющих на целевую переменную через eli5
def features_select_eli5(X, y, model, limit=10, mute=0):

    model.fit(X, y)
    display(eli5.explain_weights(model, top=limit))

    if not mute:
        df0 = eli5.explain_weights_df(model, top=limit)
        df_scores = df0['weight'] * 100
        df_columns = df0['feature']
        return features_select_visualize(df_scores, df_columns, limit=limit)
    else:
        return model

# выводим список топ колонок, влияющих на целевую переменную через corr
def features_select_corr(df, target, limit=10):

    corr = df.corr()[target].drop(labels=[target]).abs().map(lambda x: x * 1000)
    df_scores = pd.DataFrame(corr.values)
    df_columns = pd.DataFrame(corr.index)
    return features_select_visualize(df_scores, df_columns, limit=limit)

# делаем магию и по начальному датасету делаем первый прогноз по модели xgboost
def features_select_top(df, target, replace={}):

    df0 = df.copy()
    # убираем пропуски
    df0 = df0.replace(replace)
    df0 = gap_drop(df0)
    # числовые значения привели к float
    df0 = features_transform_float64(df0, [target])
    df0 = features_transform_float64(df0)
    # нечисловые значения кодируем в label encoder
    df0 = features_le(df0, df0.select_dtypes(exclude=['number']))
    # разбиваем выборку
    X_train, X_valid, X_test, y_train, y_valid = \
        df_split(df0, target, prc_train=99.999, prc_valid=0.001, prc_test=0)
    # запускаем модель
    model = features_select_eli5(
        X_train, y_train,
        xgb.XGBRegressor(learning_rate=0.01, max_depth=5, min_child_weight=3, n_estimators=300, random_state=42),
        mute=1, limit=100)
    # делаем предикт
    if(len(X_test)):
        y_pred = model.predict(X_test)
        return model, y_pred
    else:
        return model

# выборка со всеми известными результатами - делим данные на три части
# данные для обучения 70%, данные для валидации 10% и данные для тестирования 20%
# X_train, X_valid, X_test, y_train, y_valid, y_test = df_split(df_normalize, 'Churn')
#
# выборка с тестовыми данным для предсказания - выделяем выборку test по неизвестному таргету
# делим оставшиеся данные на 2 части: данные для обучения 85%, данные для валидации 15%
# X_train, X_valid, X_test, y_train, y_valid = df_split(df_normalize, 'Churn', prc_train=85, prc_valid=15, prc_test=0)
#
# обучаем на обучающей выборке, подбираем гиперпараметры на валидационной выборке,
# финальный тест делаем на тестовой выборке
#
# https://medium.com/artificialis/what-is-validation-data-and-what-is-it-used-for-158d685fb921
#
def df_split(df, target, prc_train=70, prc_valid=10, prc_test=20, target_test_value=np.nan, random_state=42):

    if prc_test > 0:
        X = df.drop(target, axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size = (prc_test + prc_valid) / 100,
            random_state = random_state)
        X_test, X_valid, y_test, y_valid = train_test_split(
            X_test, y_test,
            test_size = prc_valid / (prc_test + prc_valid),
            random_state = random_state)
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    elif prc_test == 0:
        mask1 = df[target].map(lambda x: math.isnan(x))
        mask2 = df[target] == target_test_value
        mask = mask1 if math.isnan(target_test_value) else mask2
        X_test = df[mask].drop(target, axis=1)
        X = df[~mask].drop(target, axis=1)
        y = df[~mask][target]
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y,
            test_size = prc_valid / 100,
            random_state = random_state)
        return X_train, X_valid, X_test, y_train, y_valid

# делаем визуализацию кривой обучения на основе модели и данных для обучения
def learning_curves_visualize(
    model,
    X_train, y_train, X_valid, y_valid,
    learning_curves_dots=100,
    scoring_greater_is_better=True,
    cv=5):

    train_errors, valid_errors = [], []

    # определяем learning_curves_step исходя из желаемого числа точек на графике
    learning_curves_step = math.ceil(len(X_train) / learning_curves_dots)
    
    # собираем информацию об обучении модели и ошибках
    for m in range(cv, len(X_train)+1, learning_curves_step):
        model.fit(X_train[:m], y_train[:m])
        y_train_pred = model.predict(X_train[:m])
        y_valid_pred = model.predict(X_valid)
        if scoring_greater_is_better:
            train_errors.append(accuracy_score(y_train[:m], y_train_pred))
            valid_errors.append(accuracy_score(y_valid, y_valid_pred))
        else:
            train_errors.append(mean_squared_error(y_train[:m], y_train_pred))
            valid_errors.append(mean_squared_error(y_valid, y_valid_pred))
    train_errors = train_errors if scoring_greater_is_better else np.sqrt(train_errors)
    valid_errors = valid_errors if scoring_greater_is_better else np.sqrt(valid_errors)

    # выводим график кривых обучения
    plt.plot(train_errors, 'r-+', linewidth=2, label='train')
    plt.plot(valid_errors, 'b-', linewidth=3, label='valid')
    plt.xlabel('Train Size')
    plt.ylabel('Scoring')
    plt.legend()
    plt.show()

# подбор параметров модели через GridSearchCV
#
# model, y_pred = gsearch_fit(estimator, param_grid,
#                             X_train, y_train, X_valid, y_valid, X_test, y_test,
#                             scoring='accuracy', result=10, mute=False)
# если переданы параметры X_test, y_test, то считается все 3 стадии проверки
# возвращается модель и предикт по X_test
#
# model, y_pred = gsearch_fit(estimator, param_grid,
#                             X_train, y_train, X_valid, y_valid, X_test,
#                             scoring='accuracy', result=10, mute=False)
# если не передан y_test, то считаются 2 стадии, а на 3 стадии делается предикт по X_test
# возвращается модель и предикт по X_test
#
# model = gsearch_fit(estimator, param_grid,
#                     X_train, y_train, X_valid, y_valid,
#                     scoring='accuracy', result=10, mute=False)
# если не передан X_test, y_test, то считаются 2 стадии
# возвращается модель
#
def gsearch_fit(estimator, param_grid,
                X_train, y_train,
                X_valid, y_valid,
                X_test=pd.DataFrame(), y_test=pd.Series(dtype='int64'),
                scoring='accuracy',
                result=20,
                mute=False,
                n_jobs=4,
                cv=5,
                learning_curves_dots=100
               ):

    # задаем словарь скоринга
    if scoring == 'accuracy':
        scoring_f = accuracy_score
        scoring_greater_is_better = True
    if scoring == 'f1':
        scoring_f = f1_score
        scoring_greater_is_better = True
    if scoring == 'precision':
        scoring_f = precision_score
        scoring_greater_is_better = True
    if scoring == 'recall':
        scoring_f = recall_score
        scoring_greater_is_better = True
    if scoring == 'roc_auc':
        scoring_f = roc_auc_score
        scoring_greater_is_better = True
    if scoring == 'max_error':
        scoring_f = max_error
        scoring_greater_is_better = False
    if scoring == 'neg_mean_absolute_error':
        scoring_f = mean_absolute_error
        scoring_greater_is_better = False
    if scoring == 'neg_mean_squared_error':
        scoring_f = mean_squared_error
        scoring_greater_is_better = False
    if scoring == 'neg_median_absolute_error':
        scoring_f = median_absolute_error
        scoring_greater_is_better = False
    if scoring == 'r2':
        scoring_f = r2_score
        scoring_greater_is_better = True

    ###
    ### STAGE 1: обучаем модель на тренировочных данных
    ###
    
    # обучаем модель
    gsearch = GridSearchCV(
        estimator = estimator(),
        param_grid = param_grid,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cv
    )
    gsearch.fit(X_train, y_train)
    
    # выводим результаты
    if not mute:
        print('='*20 + ' Stage 1: train ' + '='*20)
        print('Best estimator: %s' % (gsearch.best_estimator_))
        print('Best train score: %f %s using %s\n' % (abs(gsearch.best_score_), gsearch.scorer_, gsearch.best_params_))

    # выбираем лучшие результаты
    stage1_best_params = []
    means = gsearch.cv_results_['mean_test_score']
    stds = gsearch.cv_results_['std_test_score']
    params = gsearch.cv_results_['params']
    ranks = gsearch.cv_results_['rank_test_score']
    for mean, stdev, param, rank in zip(means, stds, params, ranks):
        # print('%s. %f (%f) with: %r' % (rank, abs(mean), stdev, param))
        if rank < (result + 1):
            stage1_best_params.append({'rank': rank, 'param': param})

    # сортируем параметры по ранку
    stage1_sorted_params = []
    for param in sorted(stage1_best_params, key=(lambda x: x['rank'])):
        stage1_sorted_params.append(param['param'])

    ###
    ### STAGE 2: валидируем модель на валидационных данных
    ###
    
    # проверяем лучшие параметры по валидационной выборке
    stage2_valid_score = 0 if scoring_greater_is_better else np.inf
    stage2_valid_param = {}
    for param in stage1_sorted_params:
        stage2_model = estimator(**param).fit(X_train, y_train)
        y_valid_pred = stage2_model.predict(X_valid)
        stage2_score = scoring_f(y_valid, y_valid_pred)
        if (stage2_score > stage2_valid_score) and scoring_greater_is_better:
            stage2_valid_score = stage2_score
            stage2_valid_param = param
        if (stage2_score < stage2_valid_score) and not scoring_greater_is_better:
            stage2_valid_score = stage2_score
            stage2_valid_param = param
    
    # выводим результаты
    if not mute:
        print('='*20 + ' Stage 2: valid ' + '='*20)
        print('Best valid score: %f using %s\n' % (stage2_valid_score, stage2_valid_param))

    ###
    ### STAGE 2vis: визуализируем кривую обучение на основе модели с шага 2
    ###

    # вызываем функцию визуализации кривых обучения
    if learning_curves_dots != 0:
        learning_curves_visualize(
            estimator(**stage2_valid_param),
            X_train, y_train, X_valid, y_valid,
            learning_curves_dots=learning_curves_dots,
            scoring_greater_is_better=scoring_greater_is_better,
            cv=cv
        )

    ###
    ### STAGE 3: делаем предикт по тестовым данным
    ###

    # обучаем модель на лучших параметрах с предыдущего шага
    stage3_model = estimator(**stage2_valid_param).fit(X_train, y_train)
    
    # если тестовой выборки нет, то значим просто завершаем работу и возвращаем модель
    if len(X_test) == 0:
        return stage3_model
    
    # предсказываем целевую переменную по X_test
    y_test_pred = stage3_model.predict(X_test)
    
    # если нет результатов для сверки, то завершаем работу и возвращаем модель и предикт
    if len(y_test) == 0:
        return stage3_model, y_test_pred

    # если есть данные для сверки, то считаем скоринг
    stage3_score = scoring_f(y_test, y_test_pred)

    # выводим результаты
    if not mute:
        print('='*20 + ' Stage 3: test ' + '='*21)
        print('Best test score: %f using %s\n' % (stage3_score, stage2_valid_param))
    
    # возвращаем модель и предикт
    return stage3_model, y_test_pred