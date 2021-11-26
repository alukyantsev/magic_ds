import pandas as pd
import numpy as np
import math
from sklearn.metrics import make_scorer, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, max_error, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from . visualize import *
from . lazypredict import LazyClassifier, LazyRegressor

# делаем грубую оценку моделей по датасету
def model_score(estimator, X_train, y_train, X_valid, y_valid, verbose=0, custom_metric=None):

    clf = estimator(verbose=verbose, ignore_warnings=True, custom_metric=custom_metric, predictions=True,
                         random_state=42, classifiers='all')
    models, predictions = clf.fit(X_train, X_valid, y_train, y_valid)
    return models, predictions

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
def model_fit(estimator, param_grid,
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
    gsearch.fit(X_train.values, y_train.values)
    
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
    stage2_valid_score = -np.inf if scoring_greater_is_better else np.inf
    stage2_valid_param = {}
    for param in stage1_sorted_params:
        stage2_model = estimator(**param).fit(X_train.values, y_train.values)
        y_valid_pred = stage2_model.predict(X_valid.values)
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
        learning_curves(
            estimator(**stage2_valid_param),
            X_train, y_train, X_valid, y_valid,
            learning_curves_dots=learning_curves_dots,
            scoring=scoring,
            scoring_f=scoring_f,
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

# делаем визуализацию кривой обучения на основе модели и данных для обучения
def learning_curves(
    model,
    X_train, y_train, X_valid, y_valid,
    learning_curves_dots=100,
    scoring='neg_mean_squared_error',
    scoring_f=accuracy_score,
    cv=5):

    train_errors, valid_errors = [], []

    # определяем learning_curves_step исходя из желаемого числа точек на графике
    learning_curves_step = math.ceil(len(X_train) / learning_curves_dots)
    
    # собираем информацию об обучении модели и ошибках
    for m in range(cv, len(X_train)+1, learning_curves_step):
        model.fit(X_train[:m].values, y_train[:m].values)
        y_train_pred = model.predict(X_train[:m].values)
        y_valid_pred = model.predict(X_valid.values)
        train_errors.append(scoring_f(y_train[:m], y_train_pred))
        valid_errors.append(scoring_f(y_valid, y_valid_pred))
    train_errors = train_errors if scoring != 'neg_mean_squared_error' else np.sqrt(train_errors)
    valid_errors = valid_errors if scoring != 'neg_mean_squared_error' else np.sqrt(valid_errors)

    # выводим график кривых обучения
    visualize_learning_curves(train_errors, valid_errors, scoring_name=scoring)
