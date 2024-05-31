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
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

def Run_5_fold(fewdays_5fold, other_5fold, day, HSC_redundant, predict_what):

    folder_name = f'predict_few_days/{day}day'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    X_5Fold_f, y_5Fold_f = process_function.Split_XY(fewdays_5fold)
    X_5Fold_other, y_5Fold_other = process_function.Split_XY(other_5fold)
    
    X_5Fold_f, y_5Fold_f = process_function.To_nparray(X_5Fold_f, y_5Fold_f)
    X_5Fold_other, y_5Fold_other = process_function.To_nparray(X_5Fold_other, y_5Fold_other)
    
    X_more_HSC, y_more_HSC = process_function.preprocess_more_hsc(HSC_redundant)
    
    macro_f1List = []; PrecisionList = [[] for _ in range(5)]
    RecallList = [[] for _ in range(5)]; F1ScoreList = [[] for _ in range(5)]
    FPRList = [[] for _ in range(5)]; specificityList = [[] for _ in range(5)]
    conf_matrix_total = np.zeros((2, 2), dtype=int)
    LabelName = ['HHC', '!=HHC']

    for i in range(5): # 4train 1test
        y_pred = [[] for _ in range(4)]; y_test = [[] for _ in range(4)]
    
        X_train_all = []; y_train_all = []

        for j in range(5):
            if j != i: #如果不是當次的test就是train
                X_train_all.append(X_5Fold_f[j])
                y_train_all.append(y_5Fold_f[j])
                X_train_all.append(X_5Fold_other[j])
                y_train_all.append(y_5Fold_other[j])
                
        # train_all
        X_train_all = np.concatenate((*X_train_all,))
        y_train_all = np.concatenate((*y_train_all,))
        X_train_all = np.concatenate([X_train_all, X_more_HSC]) # 多的HSC
        y_train_all = np.concatenate([y_train_all, y_more_HSC])

        # test_few、test_other
        X_test_f, y_test_f = X_5Fold_f[i], y_5Fold_f[i]
        X_test_other, y_test_other = X_5Fold_other[i], y_5Fold_other[i]
        
        # shuffle
        X_train_all, y_train_all = process_function.shuffle_XY(X_train_all, y_train_all)
        X_test_f, y_test_f = process_function.shuffle_XY(X_test_f, y_test_f)
        X_test_other, y_test_other = process_function.shuffle_XY(X_test_other, y_test_other)

        model = SVC(random_state=0, probability=True, class_weight = 'balanced')
        model.fit(X_train_all, y_train_all)
        
        if predict_what == 'predict_few_days':
            y_pred = model.predict(X_test_f)
            y_test = y_test_f
            
        elif predict_what == 'predict_others':
            y_pred = model.predict(X_test_other)
            y_test = y_test_other
        
        # cnf
        conf_matrix = confusion_matrix(y_test, y_pred) # each fold
        conf_matrix_total += conf_matrix
        
        # avr f1-score
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        macro_f1List.append(macro_f1)

        # precision、recall、f1score、FPR、specificity
        report = classification_report(y_test, y_pred, output_dict=True)
        for k in range(2):
            PrecisionList[i].append(report[str(k)]['precision'])
            RecallList[i].append(report[str(k)]['recall'])
            F1ScoreList[i].append(report[str(k)]['f1-score'])
            FP = sum(conf_matrix[j][k] for j in range(2) if j != k)
            TN = sum(conf_matrix[m][n] for m in range(2) for n in range(2) if m != k and n != k)
            FPRList[i].append(FP / (FP + TN))
            specificityList[i].append(TN / (TN + FP))# Specificity
        
    train_tool.save_score(folder_name, predict_what, macro_f1List, PrecisionList, RecallList,
                          F1ScoreList, FPRList, specificityList, LabelName)
    
    train_tool.draw_cnf(conf_matrix_total, folder_name, predict_what, LabelName)