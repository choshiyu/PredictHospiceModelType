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
    model.add(Dense(32, input_dim=dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    
    if is_TwoModel:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(3, activation='softmax'))

    return model

def define_model(base_model, dim, is_TwoModel):
    
    if base_model == 'lr':
        return LogisticRegression(random_state=0)  
    elif base_model == 'catboost':
        return CatBoostClassifier(random_state=0)
    elif base_model == 'rf':
        return RandomForestClassifier(random_state=0)
    elif base_model == 'xgb':
        return xgb.XGBClassifier(random_state=0)
    elif base_model == 'mlp':
        return model_mlp(dim, is_TwoModel)
    elif base_model == 'svm':
        return SVC(random_state=0, probability=True)

def define_model_ClassWeight(base_model, dim, is_TwoModel):
    
    if base_model == 'lr':
        return LogisticRegression(random_state=0, class_weight = 'balanced')  
    elif base_model == 'catboost':
        return CatBoostClassifier(random_state=0)
    elif base_model == 'rf':
        return RandomForestClassifier(random_state=0, class_weight = 'balanced')
    elif base_model == 'xgb':
        return xgb.XGBClassifier(random_state=0)
    elif base_model == 'mlp':
        return model_mlp(dim, is_TwoModel)
    elif base_model == 'svm':
        return SVC(random_state=0, probability=True, class_weight = 'balanced')

def get_impute_data(imputation, i):
    
    data = pd.read_excel(f'save_impute_data_5fold/{imputation}/{i}fold.xlsx', engine='openpyxl')
    
    X_train = data.drop('Group', axis=1)
    y_train = data['Group']

    return X_train, y_train

def Run_5_fold(data_list_Total5Fold, HSC_redundant, folder_name, base_model,
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
    
    folder_name = f'{base_model}_result_output/{imputation}'
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

    names_dict = {
        isCopy: 'Copy',
        isENN: 'ENN',
        isENNaddSMOTE: 'ENNandSMOTE',
        is_ClassWeight: 'ClassWeight',
        is_TwoModel: 'TwoModel'
    }
    name = names_dict.get(True, '')
        
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
        if feature_engi == 'pca':
            X_train, X_test = process_function.PCA_DimensionalityReduction(X_train, X_test, Num_f, folder_name)
        elif feature_engi == 'rf':
            selected_features = process_function.RandomForest_FeatureSelect(X_train, y_train, Num_f, folder_name, X_col, i)
        elif feature_engi == 'rfe':
            selected_features = process_function.RFE_FeatureSelect(X_train, y_train, Num_f, X_col, base_model)
        elif feature_engi == 'vt':
            selected_features = process_function.VT_FeatureSelect(X_train, Num_f, X_col)
        elif feature_engi == 'Statistical_Analysis' and Num_f == 51:
            selected_features = process_function.Statistical_FeatureSelect(X_col)
            
        # get impute train
        X_train, y_train = get_impute_data(imputation, i)

        if feature_engi != 'pca': # 不是PCA的特徵選擇保留selected_features
            X_train, X_test = pd.DataFrame(X_train, columns=X_col), pd.DataFrame(X_test, columns=X_col)
            X_train, X_test = X_train[selected_features], X_test[selected_features]
        # ---------------------------

        if is_ClassWeight:

            classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
            
            class_weights = {}
            for class_label in np.unique(y_train):
                class_weights[class_label] = classes_weights[y_train == class_label].mean()
            
            model = define_model_ClassWeight(base_model, X_train.shape[1], is_TwoModel)
            
        else:
            model = define_model(base_model, X_train.shape[1], is_TwoModel)
        
        if is_TwoModel:
            
            # 1st stage
            FirstPredict_y_train = np.where(np.isin(y_train, [0, 1]), 0, 1)
            
            if isinstance(model, Sequential):
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_m])            
                model.fit(X_train, FirstPredict_y_train, epochs=50, batch_size=32, validation_split=0.2)
                y_pred = model.predict(X_test) # 為機率，要再轉換
                y_pred = (y_pred >= 0.5).astype(int)
            else:
                model.fit(X_train, FirstPredict_y_train) #fit第一階段的label
                y_pred = model.predict(X_test) # 第一階段預測
            
            y_pred = np.where(y_pred == 1, 2, 1)
            hsc_indices = np.where(y_pred == 2)[0] # 紀錄預測是hsc的index
            hhchic_indices = np.where(y_pred == 1)[0] # 紀錄預測不是hsc的index
            
            y_pred_hsc = y_pred[hsc_indices] #只記錄是hsc的y_pred
            
            # X_train, y_train保留y_train不是hsc的
            selected_indices = y_train != 2
            X_train = X_train[selected_indices]
            y_train = y_train[selected_indices]
            
            hsc_y_test = y_test[hsc_indices] #只記錄預測成hsc的y_test
            hhchic_X_test, hhchic_y_test = X_test[hhchic_indices], y_test[hhchic_indices]#只記錄預測成hhc或hic的

            # second stage
            model = define_model(base_model, X_train.shape[1], is_TwoModel)
            
            if isinstance(model, Sequential):
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_m])        
                model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
                hhchic_y_pred = model.predict(hhchic_X_test) #fit第二階段的label，裡面還是有可能預測成hsc
                hhchic_y_pred = (hhchic_y_pred >= 0.5).astype(int)
                
            else:
                model.fit(X_train, y_train) #fit第二階段的label
                hhchic_y_pred = model.predict(hhchic_X_test) #fit第二階段的label，裡面還是有可能預測成hsc
            
            y_test = np.concatenate((hsc_y_test, hhchic_y_test))
            y_pred = np.concatenate((y_pred_hsc, hhchic_y_pred))
            
        else: 
            if isinstance(model, Sequential):  # mlp
                y_train_forMLP = tf.keras.utils.to_categorical(y_train) # 轉成三個0/1，而不是一個0/1/2
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1_m])
                if is_ClassWeight:
                    model.fit(X_train, y_train_forMLP, epochs=50, batch_size=32, validation_split=0.2, class_weight=class_weights)
                else:
                    model.fit(X_train, y_train_forMLP, epochs=50, batch_size=32, validation_split=0.2)
                y_pred = model.predict(X_test) # y_pred為三種類別的機率，要再做轉換
                y_prob = y_pred
                y_pred = np.argmax(y_pred, axis=1) # 依據最高機率變回0/1/2
            
            else: #如果是ml
                if is_ClassWeight and (base_model == 'xgb' or base_model == 'catboost'):
                    model.fit(X_train, y_train, sample_weight=classes_weights)
                else:
                    model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)
        
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