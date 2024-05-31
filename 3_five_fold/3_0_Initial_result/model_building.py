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
from keras import backend as K

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

def model_mlp(dim):

    model = Sequential()
    model.add(Dense(32, input_dim=dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    return model

def define_model(base_model, dim):
    
    if base_model == 'lr':
        return LogisticRegression(random_state=0)  
    elif base_model == 'catboost':
        return CatBoostClassifier(random_state=0)
    elif base_model == 'rf':
        return RandomForestClassifier(random_state=0)
    elif base_model == 'xgb':
        return xgb.XGBClassifier(random_state=0)
    elif base_model == 'mlp':
        return model_mlp(dim)
    elif base_model == 'svm':
        return SVC(random_state=0, probability=True)

def Run_5_fold(data_list_Total5Fold, HSC_redundant, folder_name, base_model):
    
    folder_name = f'{base_model}_result_output'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant = process_function.Split_XY(data_list_Total5Fold, HSC_redundant)
    X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant = process_function.To_nparray(X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant)
    
    LabelName = ['HHC', 'HIC', 'HSC']
    macro_f1List = []
    PrecisionList = [[] for _ in range(5)]
    RecallList = [[] for _ in range(5)]
    F1ScoreList = [[] for _ in range(5)]
    FPRList = [[] for _ in range(5)]
    specificityList = [[] for _ in range(5)]
    conf_matrix_total = np.zeros((3, 3), dtype=int)
    all_precisions = []; all_recalls = []; all_class_auc_scores = []

    with open(os.path.join(folder_name, f'Result.txt'), 'w') as f:
    
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

            model = define_model(base_model, X_train.shape[1])
            
            # 如果是mlp才需要，因為我其他都是傳統ml模型所以用Sequential來判斷
            if isinstance(model, Sequential):
                y_train_forMLP = tf.keras.utils.to_categorical(y_train) # 轉成三個0/1，而不是一個0/1/2
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1_m])
                model.fit(X_train, y_train_forMLP, epochs=50, batch_size=32, validation_split=0.2)
                y_pred = model.predict(X_test) # y_pred為三種類別的機率，要再做轉換
                y_prob = y_pred
                y_pred = np.argmax(y_pred, axis=1) # 依據最高機率變回0/1/2
            
            else: #如果是ml
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)
            
            # for PR curve
            precisions_forPRC = []; recalls_forPRC = []; class_auc_scores = []
            for j in range(3):
                precision_forPRC, recall_forPRC, _ = precision_recall_curve(y_test == j, y_prob[:, j])
                precisions_forPRC.append(precision_forPRC)
                recalls_forPRC.append(recall_forPRC)
                class_auc = auc(recall_forPRC, precision_forPRC)
                class_auc_scores.append(class_auc)
                
            all_precisions.append(precisions_forPRC)
            all_recalls.append(recalls_forPRC)
            all_class_auc_scores.append(class_auc_scores)
            
            # cnf
            conf_matrix = confusion_matrix(y_test, y_pred) # 1 fold的 cnf
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
        
        # f1score的平均、最高、最低
        avr_highest_lowest_f1score = train_tool.CountScore(macro_f1List)
        
        # precision、recall、f1score的平均、最高、最低
        precision_CountDoneList = [[] for _ in range(3)]
        recall_CountDoneList = [[] for _ in range(3)]
        f1score_CountDoneList = [[] for _ in range(3)]
        FPR_CountDoneList = [[] for _ in range(3)]
        specificity_CountDoneList = [[] for _ in range(3)]

        for i in range(3):
            precision_CountDoneList[i] = train_tool.Format_2f(train_tool.CountScore([item[i] for item in PrecisionList[0:5]]))
            recall_CountDoneList[i] = train_tool.Format_2f(train_tool.CountScore([item[i] for item in RecallList[0:5]]))
            f1score_CountDoneList[i] = train_tool.Format_2f(train_tool.CountScore([item[i] for item in F1ScoreList[0:5]]))
            FPR_CountDoneList[i] = train_tool.Format_2f(train_tool.CountScore([item[i] for item in FPRList[0:5]]))
            specificity_CountDoneList[i] = train_tool.Format_2f(train_tool.CountScore([item[i] for item in specificityList[0:5]]))
        avr_highest_lowest_f1score = train_tool.Format_2f(avr_highest_lowest_f1score)

        print(' '*5+'precision'+' '*15+'recall(TPR)'+' '*15+'f1-score'+' '*15+'FPR'+' '*15+'specificity', file=f)

        for i in range(3):
            print(LabelName[i],
                '  %s(H:%s/ L:%s) ' % (precision_CountDoneList[i][0]
                , precision_CountDoneList[i][1], precision_CountDoneList[i][2]),
                '%s(H:%s/ L:%s) ' % (recall_CountDoneList[i][0]
                , recall_CountDoneList[i][1], recall_CountDoneList[i][2]),
                '%s(H:%s/ L:%s) ' % (f1score_CountDoneList[i][0]
                , f1score_CountDoneList[i][1], f1score_CountDoneList[i][2]),
                '%s(H:%s/ L:%s) ' % (FPR_CountDoneList[i][0]
                , FPR_CountDoneList[i][1], FPR_CountDoneList[i][2]),
                '%s(H:%s/ L:%s)\n' % (specificity_CountDoneList[i][0]
                , specificity_CountDoneList[i][1], specificity_CountDoneList[i][2])
                , file=f)
        print('Mean of Macro F1-score:%s(H:%s/ L:%s) ' % (avr_highest_lowest_f1score[0], avr_highest_lowest_f1score[1], avr_highest_lowest_f1score[2]), file=f)

        # 5fold統合的cnf
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_total, annot=True, fmt="d", cmap="Blues", linewidths=.5, square=True)
        sns.heatmap(conf_matrix_total, annot=True, fmt="d", cmap="Blues", linewidths=.5, square=True,
                xticklabels=LabelName, yticklabels=LabelName)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix Heatmap')
        plt.savefig(os.path.join(folder_name, 'Confusion Matrix.png'))

        #PRcurve
        plt.figure(figsize=(15, 8))

        for fold in range(3):
            plt.subplot(2, 3, fold + 1)
            for i in range(3):
                plt.plot(all_recalls[fold][i], all_precisions[fold][i], lw=2, label=LabelName[i])

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Fold {fold + 1}')
            plt.legend(loc="upper right")

            plt.text(0.04, 0.25, f'AUC-PR:\n' + '\n'.join([f"{LabelName[i]}: {all_class_auc_scores[fold][i]:.3f}" for i in range(3)]),
                    horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)


        for fold in range(3, 5):
            plt.subplot(2, 3, fold + 1)
            for i in range(3):
                plt.plot(all_recalls[fold][i], all_precisions[fold][i], lw=2, label=f'Class {i}')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Fold {fold + 1}')
            plt.legend(loc="upper right")

            plt.text(0.04, 0.25, f'AUC-PR:\n' + '\n'.join([f"{LabelName[i]}: {all_class_auc_scores[fold][i]:.3f}" for i in range(3)]),
                    horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)


        plt.suptitle('Precision-Recall Curves for Multiclass (One-vs-Rest) - 5 Folds', y=1.02)  # 添加总标题
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, 'PRcurve.png'))