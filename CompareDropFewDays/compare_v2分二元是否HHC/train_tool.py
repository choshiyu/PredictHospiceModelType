import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve, auc
import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

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

        for i in range(2):
            precision_CountDoneList[i] = Format_2f(CountScore([item[i] for item in PrecisionList[0:5]]))
            recall_CountDoneList[i] = Format_2f(CountScore([item[i] for item in RecallList[0:5]]))
            f1score_CountDoneList[i] = Format_2f(CountScore([item[i] for item in F1ScoreList[0:5]]))
            FPR_CountDoneList[i] = Format_2f(CountScore([item[i] for item in FPRList[0:5]]))
            specificity_CountDoneList[i] = Format_2f(CountScore([item[i] for item in specificityList[0:5]]))
        avr_highest_lowest_f1score = Format_2f(avr_highest_lowest_f1score)

        print(' '*5+'precision'+' '*15+'recall(TPR)'+' '*15+'f1-score'+' '*15+'FPR'+' '*15+'specificity', file=f)

        for i in range(2):
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