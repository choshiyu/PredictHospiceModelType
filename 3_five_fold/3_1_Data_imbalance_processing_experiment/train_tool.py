import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve, auc
import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

def enn_augmentation(X_train, y_train, folder_name, i):
    '''
    ENNL:剔除被認為是噪声或重疊樣本的樣本，保留與多數鄰居樣本類别一致的樣本
    '''
    mode = 'a' if i != 0 else 'w'
    
    with open(os.path.join(folder_name, f'ENN.txt'), mode) as f:
    
        print("\n\n\nbefore ENN: len(label 0)", len(np.where(y_train == 0)[0]),
            ', len(label 1)', len(np.where(y_train == 1)[0]),
            ', len(label 2)', len(np.where(y_train == 2)[0]), file=f)
        
        enn = EditedNearestNeighbours(sampling_strategy='majority', n_neighbors=16)
        X_train, y_train = enn.fit_resample(X_train, y_train)
        
        print("after ENN: len(label 0)", len(np.where(y_train == 0)[0]),
            ', len(label 1)', len(np.where(y_train == 1)[0]),
            ', len(label 2)', len(np.where(y_train == 2)[0]), file=f)

    return X_train, y_train

def ENNaddSMOTE_augmentation(X_train, y_train, folder_name, i):
    '''
    先enn再smote
    '''
    mode = 'a' if i != 0 else 'w'
    
    with open(os.path.join(folder_name, f'ENN_and_SMOTE.txt'), mode) as f:
    
        print("\n\n\nbefore SMOTE: len(label 0)", len(np.where(y_train == 0)[0]),
            ', len(label 1)', len(np.where(y_train == 1)[0]),
            ', len(label 2)', len(np.where(y_train == 2)[0]), file=f)

        enn = EditedNearestNeighbours(sampling_strategy='majority', n_neighbors=16)
        X_train, y_train = enn.fit_resample(X_train, y_train)
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

        print("after SMOTE: len(label 0)", len(np.where(y_train == 0)[0]),
            ', len(label 1)', len(np.where(y_train == 1)[0]),
            ', len(label 2)', len(np.where(y_train == 2)[0]), file=f)
    
    return X_train, y_train

def CopyAugmentation(X_train, y_train, folder_name, i):
    '''
    hhc、hic分別複製成1.4跟1.5倍
    '''
    mode = 'a' if i != 0 else 'w'
    with open(os.path.join(folder_name, f'Copy.txt'), mode) as f:
        hhc_indices = np.where(y_train == 0)[0] # 紀錄hhc的index
        hic_indices = np.where(y_train == 1)[0] # 紀錄hic的index
        hsc_indices = np.where(y_train == 2)[0]
        
        X_train_hhc = X_train[hhc_indices]; y_train_hhc = y_train[hhc_indices]
        X_train_hic = X_train[hic_indices]; y_train_hic = y_train[hic_indices]
        X_train_hsc = X_train[hsc_indices]; y_train_hsc = y_train[hsc_indices]
        
        print('\n\n\nbefore CopyAugmentation: len(hhc)', len(y_train_hhc),
            ', len(hic)', len(y_train_hic),
            ', len(hsc)', len(y_train_hsc), file=f)

        k = 10
        kmeansModel = KMeans(n_clusters=k, random_state=46)
        clusters_pre_hhc = kmeansModel.fit_predict(X_train_hhc)
        clusters_pre_hic = kmeansModel.fit_predict(X_train_hic)

        X_train_hhc_aug = []; X_train_hic_aug = []
        for i in range(k):
            X_train_hhc_aug.append(X_train_hhc[(clusters_pre_hhc == i)][:23])
            X_train_hic_aug.append(X_train_hic[(clusters_pre_hic == i)][:26])
        
        X_train_hhc_aug = np.concatenate(X_train_hhc_aug, axis=0)
        X_train_hic_aug = np.concatenate(X_train_hic_aug, axis=0)
        
        y_train_hhc_aug = np.array([0] * len(X_train_hhc_aug))
        y_train_hic_aug = np.array([1] * len(X_train_hic_aug))

        print('\nafter CopyAugmentation: len(hhc)', len(y_train_hhc)+len(y_train_hhc_aug),
            ', len(hic)', len(y_train_hic)+len(y_train_hic_aug),
            ', len(hsc)', len(y_train_hsc), file=f)

        X_train = np.concatenate((X_train_hhc, X_train_hic, X_train_hsc, X_train_hhc_aug, X_train_hic_aug))
        y_train = np.concatenate((y_train_hhc, y_train_hic, y_train_hsc, y_train_hhc_aug, y_train_hic_aug))

        shuffle_idx = np.random.permutation(len(X_train)) # shuffle
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]

    return X_train, y_train

def CountScore(ScoreList):
    '''
    獲得一個list中的平均值、最大值、最小值
    '''
    avr_Score = sum(ScoreList) / len(ScoreList)
    highest_Score = max(ScoreList)
    lowest_Score = min(ScoreList)
    avr_highest_lowest_score = [avr_Score, highest_Score, lowest_Score]
    
    return avr_highest_lowest_score

def Format_2f(my_list):
    '''
    list不能直接在輸出的時候轉format小數點後2位
    所以值算好後先一併轉
    '''
    formatted_list = ["{:.2f}".format(round(item+0.001, 2)) for item in my_list]

    return formatted_list

def draw_cnf(conf_matrix_total, folder_name, name, LabelName):
    
    # 5fold統合的cnf
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_total, annot=True, fmt="d", cmap="Blues", linewidths=.5, square=True)
    sns.heatmap(conf_matrix_total, annot=True, fmt="d", cmap="Blues", linewidths=.5, square=True,
            xticklabels=LabelName, yticklabels=LabelName)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap')
    plt.savefig(os.path.join(folder_name, f'{name}_cnf_Matrix.png'))

def draw_pr(folder_name, name, all_recalls, all_precisions, all_class_auc_scores, LabelName):

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
    plt.savefig(os.path.join(folder_name, f'{name}_PRcurve.png'))
    
def PRC_cnt_score(y_test, y_prob):
    
    precisions_forPRC = []; recalls_forPRC = []; class_auc_scores = []
    for j in range(3):
        precision_forPRC, recall_forPRC, _ = precision_recall_curve(y_test == j, y_prob[:, j])
        precisions_forPRC.append(precision_forPRC)
        recalls_forPRC.append(recall_forPRC)
        class_auc = auc(recall_forPRC, precision_forPRC)
        class_auc_scores.append(class_auc)
    
    return precisions_forPRC, recalls_forPRC, class_auc_scores
        
def save_score(folder_name, name, macro_f1List, PrecisionList, RecallList, F1ScoreList, FPRList, specificityList, LabelName):
    
    with open(os.path.join(folder_name, f'{name}_result.txt'), 'w') as f:

        # f1score的平均、最高、最低
        avr_highest_lowest_f1score = CountScore(macro_f1List)
        
        # precision、recall、f1score的平均、最高、最低
        precision_CountDoneList = [[] for _ in range(3)]
        recall_CountDoneList = [[] for _ in range(3)]
        f1score_CountDoneList = [[] for _ in range(3)]
        FPR_CountDoneList = [[] for _ in range(3)]
        specificity_CountDoneList = [[] for _ in range(3)]

        for i in range(3):
            precision_CountDoneList[i] = Format_2f(CountScore([item[i] for item in PrecisionList[0:5]]))
            recall_CountDoneList[i] = Format_2f(CountScore([item[i] for item in RecallList[0:5]]))
            f1score_CountDoneList[i] = Format_2f(CountScore([item[i] for item in F1ScoreList[0:5]]))
            FPR_CountDoneList[i] = Format_2f(CountScore([item[i] for item in FPRList[0:5]]))
            specificity_CountDoneList[i] = Format_2f(CountScore([item[i] for item in specificityList[0:5]]))
        avr_highest_lowest_f1score = Format_2f(avr_highest_lowest_f1score)

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
        print('Mean of Macro F1-score:%s(H:%s/ L:%s) ' % (avr_highest_lowest_f1score[0],avr_highest_lowest_f1score[1], avr_highest_lowest_f1score[2]), file=f)