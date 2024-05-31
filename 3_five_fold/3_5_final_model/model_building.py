import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
from catboost import CatBoostClassifier
import process_function
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import train_tool
from keras.models import Sequential
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from keras import backend as K
from collections import Counter
from sklearn.utils import class_weight
from keras.layers import Dropout

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def model_mlp(dim, is_TwoModel):

    np.random.seed(42)
    tf.random.set_seed(42)
    model = Sequential()
    model.add(Dense(16, input_dim=dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.05))
    
    if is_TwoModel:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(3, activation='softmax'))

    return model

def define_model(base_model, dim, is_TwoModel):
    
    if base_model == 'lr':
        return LogisticRegression(random_state=0, C=10,
                                  max_iter=300, penalty='l2')
    elif base_model == 'catboost':
        return CatBoostClassifier(random_state=0, depth=6, iterations=500,
                                  l2_leaf_reg=5, learning_rate=0.1)
    elif base_model == 'rf':
        return RandomForestClassifier(random_state=0, max_depth=None, min_samples_leaf=2,
                                      min_samples_split=10, n_estimators=100)
    elif base_model == 'xgb':
        return xgb.XGBClassifier(random_state=0, gamma=0.001, learning_rate=0.15, max_depth=3, n_estimators=200)
    elif base_model == 'mlp':
        return model_mlp(dim, is_TwoModel)
    elif base_model == 'svm':
        return SVC(random_state=0, probability=True, C=1, gamma=0.1, kernel='rbf')

def define_model_ClassWeight(base_model, dim, is_TwoModel):
    
    if base_model == 'lr':
        return LogisticRegression(random_state=0, class_weight = 'balanced',
                                  C=10, max_iter=300, penalty='l2')  
    elif base_model == 'catboost':
        return CatBoostClassifier(random_state=0, depth=6, iterations=500,
                                  l2_leaf_reg=5, learning_rate=0.1)
    elif base_model == 'rf':
        return RandomForestClassifier(random_state=0, class_weight = 'balanced', max_depth=None,
                                      min_samples_leaf=2, min_samples_split=10, n_estimators=100)
    elif base_model == 'xgb':
        return xgb.XGBClassifier(random_state=0, gamma=0.001, learning_rate=0.15, max_depth=3, n_estimators=200)
    elif base_model == 'mlp':
        return model_mlp(dim, is_TwoModel)
    elif base_model == 'svm':
        return SVC(random_state=0, probability=True, class_weight = 'balanced',
                   C=1, gamma=0.1, kernel='rbf')

def get_impute_data(imputation, i):
    
    data = pd.read_excel(f'../3_3_Impute_missing_values_experiment/save_impute_data_5fold/{imputation}/{i}fold.xlsx', engine='openpyxl')
    
    X_train = data.drop('Group', axis=1)
    y_train = data['Group']

    return X_train, y_train

def Run_5_fold(data_list_Total5Fold, HSC_redundant, folder_name,
            is_ClassWeight, isCopy, isENN, isENNaddSMOTE, is_TwoModel,
            feature_engi, Num_f, imputation):
    
    # 檢查有沒有衝突到，Impute存下來的資料無法跟接受以下處理的資料不通用
    if imputation is not None and (isCopy or isENN or isENNaddSMOTE or feature_engi=='pca'):
        print('warning : imputation is not None and (isCopy or isENN or isENNaddSMOTE or feature_engi==pca)!!!!!')
        sys.exit()
    elif sum([is_ClassWeight, isCopy, isENN, isENNaddSMOTE, is_TwoModel]) > 1: #同時只能一個補缺值方法
        print('warning : sum([is_ClassWeight, isCopy, isENN, isENNaddSMOTE, is_TwoModel]) > 1!!!!!')
        sys.exit()
    else:
        pass
    
    name = 'Combine_model'
    folder_name = f'{name}_result'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant = process_function.Split_XY(data_list_Total5Fold, HSC_redundant)
    X_col = X_5Fold_list[0].columns
    X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant = process_function.To_nparray(X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant)
    
    macro_f1List = []; PrecisionList = [[] for _ in range(5)]
    RecallList = [[] for _ in range(5)]; F1ScoreList = [[] for _ in range(5)]
    FPRList = [[] for _ in range(5)]; specificityList = [[] for _ in range(5)]
    conf_matrix_total = np.zeros((3, 3), dtype=int)
    all_precisions = []; all_recalls = []; all_class_auc_scores = []
    LabelName = ['HHC', 'HIC', 'HSC']
        
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

        # imbalanced processing: Copy、ENN、SMOTE
        if isCopy:
            X_train, y_train = train_tool.CopyAugmentation(X_train, y_train, folder_name, i)
        if isENN:
            X_train, y_train = train_tool.enn_augmentation(X_train, y_train, folder_name, i)
        if isENNaddSMOTE:
            X_train, y_train = train_tool.ENNaddSMOTE_augmentation(X_train, y_train, folder_name, i)

        # -----feature engineering---------
        cat_selected_features = process_function.RFE_FeatureSelect(X_train, y_train, Num_f, X_col, 'catboost')
        xgb_selected_features = process_function.RFE_FeatureSelect(X_train, y_train, Num_f, X_col, 'catboost')
        svm_selected_features = process_function.RFE_FeatureSelect(X_train, y_train, Num_f, X_col, 'catboost')
            
        # get impute train
        X_train, y_train = get_impute_data(imputation, i)

        X_train, X_test = pd.DataFrame(X_train, columns=X_col), pd.DataFrame(X_test, columns=X_col)
        X_train_cat, X_test_cat = X_train[cat_selected_features], X_test[cat_selected_features]
        X_train_xgb, X_test_xgb = X_train[xgb_selected_features], X_test[xgb_selected_features]
        X_train_svm, X_test_svm = X_train[svm_selected_features], X_test[svm_selected_features]
        # ---------------------------
        
        # imbalanced processing: ClassWeight
        classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
            
        catboost_model = define_model_ClassWeight('catboost', X_train.shape[1], is_TwoModel)
        xgb_model = define_model_ClassWeight('xgb', X_train.shape[1], is_TwoModel)
        svm_model = define_model_ClassWeight('svm', X_train.shape[1], is_TwoModel)

        catboost_model.fit(X_train_cat, y_train, sample_weight=classes_weights)
        xgb_model.fit(X_train_xgb, y_train, sample_weight=classes_weights)
        svm_model.fit(X_train_svm, y_train)
        
        catboost_y_prob = catboost_model.predict_proba(X_test_cat)
        xgb_y_prob = catboost_model.predict_proba(X_test_xgb)
        svm_y_prob = catboost_model.predict_proba(X_test_svm)
        
        y_prob = catboost_y_prob + xgb_y_prob + svm_y_prob
        y_prob /= 3  # 假设有三个模型
        y_pred = np.argmax(y_prob, axis=1)# 依據最高機率變回0/1/2
        
        # for PR curve
        if is_TwoModel == False: # two model沒辦法畫prc
            precisions_forPRC, recalls_forPRC, class_auc_scores = train_tool.PRC_cnt_score(y_test, y_prob)
            all_precisions.append(precisions_forPRC)
            all_recalls.append(recalls_forPRC)
            all_class_auc_scores.append(class_auc_scores)
        
        # cnf
        conf_matrix = confusion_matrix(y_test, y_pred) # each fold
        # print(conf_matrix)
        conf_matrix_total += conf_matrix
        
        # avr f1-score
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        macro_f1List.append(macro_f1)

        # precision、recall、f1score、FPR、specificity
        report = classification_report(y_test, y_pred, output_dict=True)
        for k in range(3):
            PrecisionList[i].append(report[str(k)]['precision'])
            RecallList[i].append(report[str(k)]['recall'])
            F1ScoreList[i].append(report[str(k)]['f1-score'])
            FP = sum(conf_matrix[j][k] for j in range(3) if j != k)
            TN = sum(conf_matrix[m][n] for m in range(3) for n in range(3) if m != k and n != k)
            FPRList[i].append(FP / (FP + TN))
            specificityList[i].append(TN / (TN + FP))# Specificity
        
    train_tool.save_score(folder_name, name, macro_f1List, PrecisionList, RecallList,
                          F1ScoreList, FPRList, specificityList, LabelName)
    
    train_tool.draw_cnf(conf_matrix_total, folder_name, name, LabelName)
    if is_TwoModel == False: # two model沒辦法畫prc
        train_tool.draw_pr(folder_name, name, all_recalls, all_precisions, all_class_auc_scores, LabelName)