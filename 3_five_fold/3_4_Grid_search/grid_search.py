import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from catboost import CatBoostClassifier
import process_function
import os
import pandas as pd
import numpy as np
import model_building
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from collections import Counter
from sklearn.utils import class_weight

def create_model(neurons=16, DropoutRate=0.05):
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(neurons, activation='relu'),
      tf.keras.layers.Dense(neurons*2, activation='relu'),
      tf.keras.layers.Dropout(DropoutRate),
      tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[model_building.f1_m])
    return model

def grid_search_est(base_model):
    
    if base_model == 'lr':
        return LogisticRegression(random_state=0, class_weight = 'balanced')  
    elif base_model == 'catboost':
        return CatBoostClassifier(random_state=0)
    elif base_model == 'rf':
        return RandomForestClassifier(random_state=0, class_weight = 'balanced')
    elif base_model == 'xgb':
        return xgb.XGBClassifier(random_state=0)
    elif base_model == 'mlp':
        return tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=50, verbose=1)
    elif base_model == 'svm':
        return SVC(random_state=0, probability=True, class_weight = 'balanced')

def grid_search_param(base_model):
    
    if base_model == 'lr':
        
        param_grid = {
                'penalty': ['l1', 'l2'],
                'C': [0.1, 1, 10, 100],
                'max_iter': [100, 200, 300]
            }
        
    elif base_model == 'catboost':
        
        param_grid = {
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.009, 0.03, 0.05, 0.1],
            'iterations': [100, 200, 500, 700, 1000],
            'l2_leaf_reg': [0.2, 0.5, 1, 3, 5, 7]
        }
        
    elif base_model == 'rf':
        
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 5, 10],
        }
        
    elif base_model == 'xgb':
        
        param_grid = {
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.009, 0.015, 0.05, 0.1, 0.15],
            'gamma': [0, 0.001, 0.01],
            'n_estimators': [100, 200, 300, 500]
        }

    elif base_model == 'mlp':
        
        neurons = [16, 32, 64]
        DropoutRate = [0, 0.05, 0.1, 0.2]
        param_grid = dict(neurons=neurons, DropoutRate=DropoutRate)

    elif base_model == 'svm':

        # param_grid = {
        #     'C': [0.1, 1, 10, 100],
        #     'kernel': ['rbf'],
        #     'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'],
        # }
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'poly'],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'],
        }
    return param_grid

def class_weights(y_cv):
    
    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_cv)
                
    class_weights = {}
    for class_label in np.unique(y_cv):
        class_weights[class_label] = classes_weights[y_cv == class_label].mean()
    
    return class_weights

def Grid_Search(data_5fold, HSC_redundant, base_model,
            is_ClassWeight, feature_engi, Num_f, imputation):
    
    if is_ClassWeight == False or feature_engi != 'rfe' or Num_f != 51 or imputation != 'mice':
        print('warning!!!!!')
        sys.exit()
    
    folder_name = 'GridSearch_result'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant = process_function.Split_XY(data_5fold, HSC_redundant)
    X_col = X_5Fold_list[0].columns
    X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant = process_function.To_nparray(X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant)
    
    X_cv = []; y_cv = []; custom_cv_iterator = []; start_indices = 0
    
    # start 5 fold
    for i in range(5): # 4train 1test

        X_train_list = []; y_train_list = []
        
        for j in range(5):
            if j != i: #如果不是當次的test就是train
                X_train_list.append(X_5Fold_list[j])
                y_train_list.append(y_5Fold_list[j])

        # 第i次的X_train, y_train, X_test, y_test
        X_train = np.concatenate((*X_train_list[0:4],)) # *可以用來解包
        y_train = np.concatenate((*y_train_list[0:4],)) # 把0~3的資料們合在一起

        # 加上不會被分到任何test的hsc的資料
        X_train = np.concatenate([X_train, HSC_X_redundant])
        y_train = np.concatenate([y_train, HSC_y_redundant])
        
        X_test, y_test = X_5Fold_list[i], y_5Fold_list[i]
        
        X_train, y_train = process_function.shuffle_XY(X_train, y_train) # shuffle
        X_test, y_test = process_function.shuffle_XY(X_test, y_test)

        # feature engineering
        selected_features = process_function.RFE_FeatureSelect(X_train, y_train, Num_f, X_col, base_model)
            
        # get impute train
        X_train, y_train = model_building.get_impute_data(imputation, i)

        X_train, X_test = pd.DataFrame(X_train, columns=X_col), pd.DataFrame(X_test, columns=X_col)
        X_train, X_test = X_train[selected_features], X_test[selected_features]
        
        # 當前的indices要加上前一個迴圈記錄的indices長度
        current_indices_train = np.arange(len(y_train)) + start_indices
        current_indices_test = np.arange(len(y_test)) + start_indices + len(y_train)
        custom_cv_iterator.append((current_indices_train, current_indices_test))
        start_indices = len(y_train) + len(y_test) + start_indices # update
        
        # train test合併
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        
        # append每一fold的Xy
        X_cv.append(X)
        y_cv.append(y)
    
    # 全部的Xy concatenate起來
    X_cv = np.concatenate(X_cv, axis=0)
    y_cv = np.concatenate(y_cv, axis=0)
    
    # set est
    model = grid_search_est(base_model)
    param_grid = grid_search_param(base_model)

    # f1_score
    scorer = make_scorer(f1_score, average='macro')
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                cv=custom_cv_iterator, scoring=scorer)
    
    if base_model == 'mlp':
        y_cv_forMLP = tf.keras.utils.to_categorical(y_cv)
        grid_search.fit(X_cv, y_cv_forMLP, class_weight=class_weights(y_cv))
    if (base_model == 'xgb' or base_model == 'catboost'):
        grid_search.fit(X_cv, y_cv, sample_weight=class_weight.compute_sample_weight(class_weight='balanced', y=y_cv))
    else:
        grid_search.fit(X_cv, y_cv)

    with open(os.path.join(folder_name, f'{base_model}_optimal_parameters.txt'), 'w') as f:

        print(f'\n\n{base_model}_parameters:', file=f)
        print('Best parameters:', grid_search.best_params_, file=f)
        print('Best score:', grid_search.best_score_, file=f)