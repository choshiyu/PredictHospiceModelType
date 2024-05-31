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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import matplotlib.pyplot as plt
import joblib

def define_model_ClassWeight(base_model):
    
    if base_model == 'lr':
        return LogisticRegression(random_state=0, class_weight = 'balanced',
                                  C=10, max_iter=300, penalty='l2')  
    elif base_model == 'catboost':
        return CatBoostClassifier(random_state=0, depth=6, iterations=500,
                                  l2_leaf_reg=5, learning_rate=0.1, verbose=0)
    elif base_model == 'rf':
        return RandomForestClassifier(random_state=0, class_weight = 'balanced', max_depth=None,
                                      min_samples_leaf=2, min_samples_split=10, n_estimators=100)
    elif base_model == 'xgb':
        return xgb.XGBClassifier(random_state=0, gamma=0.001, learning_rate=0.15, max_depth=3, n_estimators=200)
    elif base_model == 'svm':
        return SVC(random_state=0, probability=True, class_weight = 'balanced',
                   C=1, gamma=0.1, kernel='rbf')

def get_impute_data(imputation, i):
    
    data = pd.read_excel(f'../3_five_fold/3_3_Impute_missing_values_experiment/save_impute_data_5fold/{imputation}/{i}fold.xlsx', engine='openpyxl')
    
    X_train = data.drop('Group', axis=1)
    y_train = data['Group']

    return X_train, y_train

def DT_learning(data_5fold, HSC_redundant, base_model, Num_f, imputation):
    
    folder_name = f'{base_model}_performance'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant = process_function.Split_XY(data_5fold, HSC_redundant)
    X_col = X_5Fold_list[0].columns
    X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant = process_function.To_nparray(X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant)
    
    macro_f1List = []; PrecisionList = [[] for _ in range(5)]
    RecallList = [[] for _ in range(5)]; F1ScoreList = [[] for _ in range(5)]
    FPRList = [[] for _ in range(5)]; specificityList = [[] for _ in range(5)]
    conf_matrix_total = np.zeros((3, 3), dtype=int)
    all_precisions = []; all_recalls = []; all_class_auc_scores = []
    LabelName = ['HHC', 'HIC', 'HSC']
    name = base_model
        
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

        # feature engineering : rfe 
        selected_features = process_function.RFE_FeatureSelect(X_train, y_train, Num_f, X_col, 'svm')
            
        # get impute train
        X_train, y_train = get_impute_data(imputation, i)

        X_train, X_test = pd.DataFrame(X_train, columns=X_col), pd.DataFrame(X_test, columns=X_col)
        X_train, X_test = X_train[selected_features], X_test[selected_features]
        # ---------------------------
        # imbalanced processing: ClassWeight
        model = define_model_ClassWeight('svm')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
    
        # DT_student
        if base_model == 'DT_student':
            soft_labels_train = model.predict(X_train)
            soft_labels_train = pd.DataFrame(soft_labels_train, columns=['Group'])

            tree_student = DecisionTreeClassifier(random_state=42) # student
            tree_student.fit(X_train, soft_labels_train)
            
            # DT_student照理說不能跟svm差太多，檢驗模型表現
            y_pred = tree_student.predict(X_test) # DT的y_pred
            y_prob = tree_student.predict_proba(X_test)
        
        # for PR curve
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
    
    # cnf matrix
    train_tool.draw_cnf(conf_matrix_total, folder_name, name, LabelName)

    # pr curve
    train_tool.draw_pr(folder_name, name, all_recalls, all_precisions, all_class_auc_scores, LabelName)

def Show_feature_importance(data_5fold, HSC_redundant, base_model, Num_f, folder_name):
    
    folder_name_DT = 'DT_tree'
    if not os.path.exists(folder_name_DT):
        os.makedirs(folder_name_DT)
    
    plot_importance_folder_name = 'plot_importance'
    if not os.path.exists(plot_importance_folder_name):
        os.makedirs(plot_importance_folder_name)
    
    X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant = process_function.Split_XY(data_5fold, HSC_redundant)
    X_cols = X_5Fold_list[0].columns
    X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant = process_function.To_nparray(X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant)
    
    X = np.concatenate(X_5Fold_list[0:5])
    y = np.concatenate(y_5Fold_list[0:5])
    X = np.concatenate([X, HSC_X_redundant])
    y = np.concatenate([y, HSC_y_redundant])
        
    X, y = process_function.shuffle_XY(X, y) # shuffle
    
    # feature engineering : rfe
    if base_model != 'DT_student':
        selected_features = process_function.RFE_FeatureSelect(X, y, Num_f, X_cols, base_model)
    else:
        selected_features = process_function.RFE_FeatureSelect(X, y, Num_f, X_cols, 'svm')
    
    # impute missing data
    OneHotColumnsName, cat_ori_cols, data_tobe_impute = process_function.get_data()
    X, y = process_function.MICEImpute(OneHotColumnsName, X_cols, X, y, data_tobe_impute, folder_name, cat_ori_cols)
    
    # keep selected_features
    X = X[selected_features]
    X_cols = X.columns
    
    # class weight
    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y)

    if base_model == 'xgb':
        
        importance = 0 # Initial feature importance
        for _ in range(100):  # Repeat 100 times to get overall results
            # 不set random state隨機跑100次
            model = xgb.XGBClassifier(gamma=0.001, learning_rate=0.15, max_depth=3, n_estimators=200)
            model.fit(X, y, sample_weight=classes_weights)
            importance += model.feature_importances_
            
        importance /= 100
        sorted_idx = importance.argsort()

        # Plot
        plt.figure(figsize=(18, 14))
        # plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
        # plt.yticks(range(len(sorted_idx)), [X_cols[i] for i in sorted_idx])
        plt.barh(range(10), importance[sorted_idx][-10:], align='center') # top 10
        plt.yticks(range(10), [X_cols[i] for i in sorted_idx][-10:], fontsize=16) # top 10
        plt.xlabel('Feature Importance', fontsize=16)
        plt.title('Feature Importance (XGBoost)', fontsize=18)
        plt.subplots_adjust(left=0.3, right=0.8, top=0.9, bottom=0.1)
        plt.savefig(os.path.join(plot_importance_folder_name, 'xgb_importance.png'))

    elif base_model == 'catboost':
        
        importance = 0 # Initial feature importance
        for _ in range(100):  # Repeat 100 times to get overall results
            # 不set random state隨機跑100次
            model = CatBoostClassifier(depth=6, iterations=500,
                                        l2_leaf_reg=5, learning_rate=0.1, verbose=0)
            model.fit(X, y, sample_weight=classes_weights)
            importance += model.get_feature_importance(type='FeatureImportance')
        
        importance /= 100
        feature_names = model.feature_names_
        sorted_idx = importance.argsort()

        # Plot
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(14, 10))  # Smaller figure size
        plt.barh(range(10), importance[sorted_idx][-10:], align='center')
        plt.yticks(range(10), [feature_names[i] for i in sorted_idx][-10:], fontsize=18)  # Larger font size
        plt.xlabel('Feature Importance', fontsize=18)
        plt.title('Feature Importance (CatBoost)', fontsize=20)  # Larger font size for title
        plt.subplots_adjust(left=0.3, right=0.8, top=0.9, bottom=0.1)
        plt.savefig(os.path.join(plot_importance_folder_name, 'CatBoost_importance.png'))
        
    elif base_model == 'DT_student':
        
        # imbalanced processing: ClassWeight
        model = define_model_ClassWeight('svm')
        model.fit(X, y)
        
        # DT_student
        soft_labels_train = model.predict(X)
        soft_labels_train = pd.DataFrame(soft_labels_train, columns=['Group'])
    
        tree_student = DecisionTreeClassifier(random_state=42, max_depth=4) # for plot tree
        tree_student.fit(X, soft_labels_train)
        
        # save model(因為我也只有要畫不超過5層的舉例)
        joblib.dump(tree_student, os.path.join(folder_name_DT, 'tree_student_model.pkl'))

        dot_data = export_graphviz(tree_student, out_file=None, 
                                feature_names=X_cols,
                                class_names=[str(cls) for cls in np.unique(soft_labels_train)],  
                                filled=True, rounded=True,  
                                special_characters=True,)
        dot_data = dot_data.replace('}', 'rankdir=LR;}', 1)
        graph = graphviz.Source(dot_data)
        graph.render(os.path.join(folder_name_DT, 'DT_student'))
        
        # X跟soft_labels_train也存下來
        X_save = pd.DataFrame(X, columns=X_cols)
        y_save = pd.DataFrame(soft_labels_train, columns=['Group'])
        data_save = pd.concat([X_save, y_save], axis=1)
        data_save.to_excel(os.path.join(folder_name_DT, 'data_for_1sample_analysis.xlsx'),index=False)