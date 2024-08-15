import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from os import path
from sklearn.model_selection import GridSearchCV
import pickle


def predict(csvPath, X_keys):
    '''
    Args:
    csvPath : 导入 csv 文件的绝对路径
    X_keys  : X 字段名, 应当是一维的列表，其中的元素为 string, 如:
              ['Burial depth', 'Poisson ratio']

    '''

    data = pd.read_csv(csvPath)
    y = data.Wear
    X = data[X_keys]
    # 切分训练集、测试集,切分比例8 : 2
    train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)

    # 参数调优并预测

    # cv函数调参
    print('数据转换')
    lgb_train = lgb.Dataset(train_X, train_y, free_raw_data=False)
    lgb_eval = lgb.Dataset(test_X, test_y, reference=lgb_train, free_raw_data=False)

    # 设置初始参数--不含交叉验证参数
    print('设置参数')
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'auc',
        'nthread': 4,
        'learning_rate': 0.1
    }
    # 交叉验证(调参)
    print('交叉验证')
    max_auc = float('0')
    Best_para_Wear = {}

    print("调参1：提高准确率")
    for num_leaves in range(5, 31, 2):
        for max_depth in range(3, 9, 1):
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth
            from lightgbm import early_stopping, log_evaluation
            callbacks = [early_stopping(stopping_rounds=10), log_evaluation(period=50)]
            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                metrics=['auc'],
                callbacks=callbacks,
                stratified=False
            )
            mean_auc = pd.Series(cv_results['valid auc-mean']).max()
            boost_rounds = pd.Series(cv_results['valid auc-mean']).idxmax()

            if mean_auc >= max_auc:
                max_auc = mean_auc
                Best_para_Wear['num_leaves'] = num_leaves
                Best_para_Wear['max_depth'] = max_depth
    if 'num_leaves' and 'max_depth' in Best_para_Wear.keys():
        params['num_leaves'] = Best_para_Wear['num_leaves']
        params['max_depth'] = Best_para_Wear['max_depth']

    # 过拟合
    print("调参2：降低过拟合")
    for max_bin in range(5, 256, 10):
        for min_data_in_leaf in range(1, 102, 10):
            params['max_bin'] = max_bin
            params['min_data_in_leaf'] = min_data_in_leaf
            callbacks = [early_stopping(stopping_rounds=10), log_evaluation(period=50)]
            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                metrics=['auc'],
                callbacks=callbacks,
                stratified=False
            )

            mean_auc = pd.Series(cv_results['valid auc-mean']).max()
            boost_rounds = pd.Series(cv_results['valid auc-mean']).idxmax()

            if mean_auc >= max_auc:
                max_auc = mean_auc
                Best_para_Wear['max_bin'] = max_bin
                Best_para_Wear['min_data_in_leaf'] = min_data_in_leaf
    if 'max_bin' and 'min_data_in_leaf' in Best_para_Wear.keys():
        params['min_data_in_leaf'] = Best_para_Wear['min_data_in_leaf']
        params['max_bin'] = Best_para_Wear['max_bin']

    print("调参3：降低过拟合")
    for feature_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
            for bagging_freq in range(0, 50, 5):
                params['feature_fraction'] = feature_fraction
                params['bagging_fraction'] = bagging_fraction
                params['bagging_freq'] = bagging_freq
                callbacks = [early_stopping(stopping_rounds=10), log_evaluation(period=50)]
                cv_results = lgb.cv(
                    params,
                    lgb_train,
                    seed=1,
                    nfold=5,
                    metrics=['auc'],
                    callbacks=callbacks,
                    stratified=False
                )

                mean_auc = pd.Series(cv_results['valid auc-mean']).max()
                boost_rounds = pd.Series(cv_results['valid auc-mean']).idxmax()

                if mean_auc >= max_auc:
                    max_auc = mean_auc
                    Best_para_Wear['feature_fraction'] = feature_fraction
                    Best_para_Wear['bagging_fraction'] = bagging_fraction
                    Best_para_Wear['bagging_freq'] = bagging_freq

    if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in Best_para_Wear.keys():
        params['feature_fraction'] = Best_para_Wear['feature_fraction']
        params['bagging_fraction'] = Best_para_Wear['bagging_fraction']
        params['bagging_freq'] = Best_para_Wear['bagging_freq']

    print("调参4：降低过拟合")
    for lambda_l1 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        for lambda_l2 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.4, 0.6, 0.7, 0.9, 1.0]:
            params['lambda_l1'] = lambda_l1
            params['lambda_l2'] = lambda_l2
            callbacks = [early_stopping(stopping_rounds=10), log_evaluation(period=50)]
            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                metrics=['auc'],
                callbacks=callbacks,
                stratified=False
            )

            mean_auc = pd.Series(cv_results['valid auc-mean']).max()
            boost_rounds = pd.Series(cv_results['valid auc-mean']).idxmax()

            if mean_auc >= max_auc:
                max_auc = mean_auc
                Best_para_Wear['lambda_l1'] = lambda_l1
                Best_para_Wear['lambda_l2'] = lambda_l2
    if 'lambda_l1' and 'lambda_l2' in Best_para_Wear.keys():
        params['lambda_l1'] = Best_para_Wear['lambda_l1']
        params['lambda_l2'] = Best_para_Wear['lambda_l2']

    print("调参5：降低过拟合2")
    for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        params['min_split_gain'] = min_split_gain
        callbacks = [early_stopping(stopping_rounds=10), log_evaluation(period=50)]
        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=1,
            nfold=5,
            metrics=['auc'],
            callbacks=callbacks,
            stratified=False
        )

        mean_auc = pd.Series(cv_results['valid auc-mean']).max()
        boost_rounds = pd.Series(cv_results['valid auc-mean']).idxmax()

        if mean_auc >= max_auc:
            max_auc = mean_auc

            Best_para_Wear['min_split_gain'] = min_split_gain
    if 'min_split_gain' in Best_para_Wear.keys():
        params['min_split_gain'] = Best_para_Wear['min_split_gain']

    parameters = {'n_estimators': range(100, 1000, 200), 'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
    my_model = lgb.LGBMRegressor()
    grid_search = GridSearchCV(my_model, parameters, scoring='r2', cv=5)
    grid_search.fit(train_X, train_y)
    Best_para_Wear.update(grid_search.best_params_)

    train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)
    model_Wear = lgb.LGBMRegressor(objective='regression', num_leaves=Best_para_Wear['num_leaves'],
                                    learning_rate=Best_para_Wear['learning_rate'],
                                    n_estimators=Best_para_Wear['n_estimators'],
                                    verbosity=2, subsample=Best_para_Wear['subsample'],
                                    colsample_bytree=Best_para_Wear['colsample_bytree'],
                                    lambda_l1=Best_para_Wear['lambda_l1'],
                                    max_depth=Best_para_Wear['max_depth'])
    model_Wear.fit(train_X, train_y)
    predictions_Wear = model_Wear.predict(test_X)
    R2 = [r2_score(predictions_Wear, test_y), r2_score(model_Wear.predict(train_X), train_y)]
    print(R2)


    with open('model_Wear.pkl', 'wb') as f:
        pickle.dump(model_Wear, f)

    feature_importance_Wear = pd.DataFrame({'feature': model_Wear.feature_name_,
                                       'importance': model_Wear.feature_importances_})
    feature_importances_Wear = feature_importance_Wear.sort_values('importance', ascending=False)
    feature_importances_Wear.to_csv('..\importanceWear.csv')

    Pre_Wear = model_Wear.predict(X.values)
    data['PreWear'] = Pre_Wear
    outdata_data_Path = path.join(path.dirname(csvPath), 'output_data_Wear.csv')
    data.to_csv(outdata_data_Path, mode='w', index=None)


    # 将磨损预测值写入文件
    outdata_pre = pd.DataFrame(data={'Prepart_Wear': test_y,
                                    'Preed_Wear': predictions_Wear})

    outdata_train = pd.DataFrame(data={'Trainpart_Wear': train_y,
                                    'Trained_Wear': model_Wear.predict(train_X)})
    data['PreWear'] = Pre_Wear

    # outdata_train 保存路径，目录和 csvPath 相同, 文件名为 output_train_Wear.csv
    outdata_train_Path = path.join(path.dirname(csvPath), 'output_train_Wear.csv')
    outdata_train.to_csv(outdata_train_Path, mode='w', index=True)
    # outdata_pre 保存路径，目录和 csvPath 相同, 文件名为 output_pre_Wear.csv
    outdata_pre_Path = path.join(path.dirname(csvPath), 'output_pre_Wear.csv')
    outdata_pre.to_csv(outdata_pre_Path, mode='w', index=True)

    outdata_data_Path = path.join(path.dirname(csvPath), 'output_data_Wear.csv')
    data.to_csv(outdata_data_Path, mode='w', index=None)

    # 预测比能
    # data2 = data
    data2 = pd.read_csv('..\output_data_Wear.csv')
    y2 = data2.SE
    X2 = data2[X_keys+['PreWear']]

    train_X2, test_X2, train_y2, test_y2 = train_test_split(X2.values, y2.values, test_size=0.2)

    # cv函数调参
    print('数据转换')
    lgb_train = lgb.Dataset(train_X2, train_y2, free_raw_data=False)
    lgb_eval = lgb.Dataset(test_X2, test_y2, reference=lgb_train, free_raw_data=False)

    # 设置初始参数--不含交叉验证参数
    print('设置参数')
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'auc',
        'nthread': 4,
        'learning_rate': 0.1
    }
    # 交叉验证(调参)
    print('交叉验证')
    max_auc = float('0')
    Best_para_SE = {}

    print("调参1：提高准确率")
    for num_leaves in range(5, 31, 2):
        for max_depth in range(3, 9, 1):
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth
            from lightgbm import early_stopping, log_evaluation
            callbacks = [early_stopping(stopping_rounds=10), log_evaluation(period=50)]
            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                metrics=['auc'],
                callbacks=callbacks,
                stratified=False
            )
            mean_auc = pd.Series(cv_results['valid auc-mean']).max()
            boost_rounds = pd.Series(cv_results['valid auc-mean']).idxmax()

            if mean_auc >= max_auc:
                max_auc = mean_auc
                Best_para_SE['num_leaves'] = num_leaves
                Best_para_SE['max_depth'] = max_depth
    if 'num_leaves' and 'max_depth' in Best_para_SE.keys():
        params['num_leaves'] = Best_para_SE['num_leaves']
        params['max_depth'] = Best_para_SE['max_depth']

    # 过拟合
    print("调参2：降低过拟合")
    for max_bin in range(5, 256, 10):
        for min_data_in_leaf in range(1, 102, 10):
            params['max_bin'] = max_bin
            params['min_data_in_leaf'] = min_data_in_leaf
            callbacks = [early_stopping(stopping_rounds=10), log_evaluation(period=50)]
            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                metrics=['auc'],
                callbacks=callbacks,
                stratified=False
            )

            mean_auc = pd.Series(cv_results['valid auc-mean']).max()
            boost_rounds = pd.Series(cv_results['valid auc-mean']).idxmax()

            if mean_auc >= max_auc:
                max_auc = mean_auc
                Best_para_SE['max_bin'] = max_bin
                Best_para_SE['min_data_in_leaf'] = min_data_in_leaf
    if 'max_bin' and 'min_data_in_leaf' in Best_para_SE.keys():
        params['min_data_in_leaf'] = Best_para_SE['min_data_in_leaf']
        params['max_bin'] = Best_para_SE['max_bin']

    print("调参3：降低过拟合")
    for feature_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
            for bagging_freq in range(0, 50, 5):
                params['feature_fraction'] = feature_fraction
                params['bagging_fraction'] = bagging_fraction
                params['bagging_freq'] = bagging_freq
                callbacks = [early_stopping(stopping_rounds=10), log_evaluation(period=50)]
                cv_results = lgb.cv(
                    params,
                    lgb_train,
                    seed=1,
                    nfold=5,
                    metrics=['auc'],
                    callbacks=callbacks,
                    stratified=False
                )

                mean_auc = pd.Series(cv_results['valid auc-mean']).max()
                boost_rounds = pd.Series(cv_results['valid auc-mean']).idxmax()

                if mean_auc >= max_auc:
                    max_auc = mean_auc
                    Best_para_SE['feature_fraction'] = feature_fraction
                    Best_para_SE['bagging_fraction'] = bagging_fraction
                    Best_para_SE['bagging_freq'] = bagging_freq

    if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in Best_para_SE.keys():
        params['feature_fraction'] = Best_para_SE['feature_fraction']
        params['bagging_fraction'] = Best_para_SE['bagging_fraction']
        params['bagging_freq'] = Best_para_SE['bagging_freq']

    print("调参4：降低过拟合")
    for lambda_l1 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        for lambda_l2 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.4, 0.6, 0.7, 0.9, 1.0]:
            params['lambda_l1'] = lambda_l1
            params['lambda_l2'] = lambda_l2
            callbacks = [early_stopping(stopping_rounds=10), log_evaluation(period=50)]
            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                metrics=['auc'],
                callbacks=callbacks,
                stratified=False
            )

            mean_auc = pd.Series(cv_results['valid auc-mean']).max()
            boost_rounds = pd.Series(cv_results['valid auc-mean']).idxmax()

            if mean_auc >= max_auc:
                max_auc = mean_auc
                Best_para_SE['lambda_l1'] = lambda_l1
                Best_para_SE['lambda_l2'] = lambda_l2
    if 'lambda_l1' and 'lambda_l2' in Best_para_SE.keys():
        params['lambda_l1'] = Best_para_SE['lambda_l1']
        params['lambda_l2'] = Best_para_SE['lambda_l2']

    print("调参5：降低过拟合2")
    for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        params['min_split_gain'] = min_split_gain
        callbacks = [early_stopping(stopping_rounds=10), log_evaluation(period=50)]
        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=1,
            nfold=5,
            metrics=['auc'],
            callbacks=callbacks,
            stratified=False
        )

        mean_auc = pd.Series(cv_results['valid auc-mean']).max()
        boost_rounds = pd.Series(cv_results['valid auc-mean']).idxmax()

        if mean_auc >= max_auc:
            max_auc = mean_auc

            Best_para_SE['min_split_gain'] = min_split_gain
    if 'min_split_gain' in Best_para_SE.keys():
        params['min_split_gain'] = Best_para_SE['min_split_gain']

    parameters2 = {'n_estimators': range(100, 1000, 200), 'subsample': [0.8, 0.9, 1.0],
                   'colsample_bytree': [0.8, 0.9, 1.0], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
    my_model2 = lgb.LGBMRegressor()
    grid_search = GridSearchCV(my_model2, parameters2, scoring='r2', cv=5)
    grid_search.fit(train_X2, train_y2)
    Best_para_SE.update(grid_search.best_params_)

    train_X2, test_X2, train_y2, test_y2 = train_test_split(X2.values, y2.values, test_size=0.2)
    model_SE = lgb.LGBMRegressor(objective='regression', num_leaves=Best_para_SE['num_leaves'],
                                        learning_rate=Best_para_SE['learning_rate'],
                                        n_estimators=Best_para_SE['n_estimators'],
                                        verbosity=2, subsample=Best_para_SE['subsample'],
                                        colsample_bytree=Best_para_SE['colsample_bytree'],
                                        lambda_l1=Best_para_SE['lambda_l1'],
                                        max_depth=Best_para_SE['max_depth'])
    model_SE.fit(train_X2, train_y2)
    predictions_SE = model_SE.predict(test_X2)
    with open('model_SE.pkl', 'wb') as f:
        pickle.dump(model_SE, f)

    Pre_SE = model_SE.predict(X2.values)
    feature_importance_SE = pd.DataFrame({'feature': model_SE.feature_name_,
                                         'importance': model_SE.feature_importances_})
    feature_importances_SE = feature_importance_SE.sort_values('importance', ascending=False)
    feature_importances_SE.to_csv('..\importanceSE.csv')

    # 将SE预测值写入文件
    outdata_pre = pd.DataFrame(data={'Prepart_SE': test_y2,
                                    'Preed_SE': predictions_SE})

    outdata_train = pd.DataFrame(data={'Trainpart_SE': train_y2,
                                    'Trained_SE': model_SE.predict(train_X2)})
    data['PreSE'] = Pre_SE

    # outdata_train 保存路径，目录和 csvPath 相同, 文件名为 output_train_SE.csv
    outdata_train_Path = path.join(path.dirname(csvPath), 'output_train_SE.csv')
    outdata_train.to_csv(outdata_train_Path, mode='w', index=True)
    # outdata_pre 保存路径，目录和 csvPath 相同, 文件名为 output_pre_SE.csv
    outdata_pre_Path = path.join(path.dirname(csvPath), 'output_pre_SE.csv')
    outdata_pre.to_csv(outdata_pre_Path, mode='w', index=True)

    outdata_data_Path = path.join(path.dirname(csvPath), 'output_data_SE.csv')
    data.to_csv(outdata_data_Path, mode='w', index=None)

