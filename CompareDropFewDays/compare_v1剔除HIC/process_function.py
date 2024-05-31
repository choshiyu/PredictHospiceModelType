import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def Split_Into_Kcluster(data_list):

    K_cluster_List = []; K_list = []
    for i in range(2):
        
        data_len = len(data_list[i])
        if data_len < 100:
            k = 2
        elif data_len >= 100 and data_len < 700:
            k = 4
        else:
            k = 6
        
        K_list.append(k)
        kmeansModel = KMeans(n_clusters=k, random_state=46)
        clusters_predicted = kmeansModel.fit_predict(data_list[i])
        K_cluster_List.append(clusters_predicted)

    return K_cluster_List, K_list

def Equally_Partition_Data(data_list, K_list, K_cluster_List):

    cluster_list = [[] for _ in range(2)]
    cluster_list = [[[] for _ in range(5)] for _ in range(2)]

    for j in range(2):
        for i in range(K_list[j]): # k群各切五份
            filt = (K_cluster_List[j] == i) # 依據分群紀錄K_cluster_List設一個filter
            # 要切5份所以設定一個0.2的point
            point = int(0.2*len(data_list[j][filt]))
            # append到第j個group的五個fold中
            cluster_list[j][0].append(data_list[j][filt][:point])
            cluster_list[j][1].append(data_list[j][filt][point:2*point])
            cluster_list[j][2].append(data_list[j][filt][2*point:3*point])
            cluster_list[j][3].append(data_list[j][filt][3*point:4*point])
            cluster_list[j][4].append(data_list[j][filt][4*point:])

        for i in range(5): # 第j個group中的第i個fold裡面的k群資料合併起來
            cluster_list[j][i] = pd.concat(cluster_list[j][i])

    return cluster_list

def Split_Into_5Fold(data, folder_name):
    
    data_hsc = data.loc[(data['Group'] == 1)] #hsc
    data_hhc = data.loc[(data['Group'] == 0)] #hhc
    data_list = [data_hsc, data_hhc]
    
    k_cluster_list, K_list = Split_Into_Kcluster(data_list)
    data_list_5FoldEachGroup = Equally_Partition_Data(data_list, K_list, k_cluster_list)

    data_list_Total5Fold = []
    for i in range(5):
        data_list_Total5Fold.append(pd.concat([data_list_5FoldEachGroup[0][i],data_list_5FoldEachGroup[1][i]]))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))# 畫圖
    for i in range(5):
        row = i // 3; col = i % 3
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data_list_Total5Fold[i])
        axes[row, col].scatter(data_2d[:, 0], data_2d[:, 1], s=10, alpha=0.7, label=f'Fold {i}')
        axes[row, col].set_title(f'Split_Into_5Fold - Fold {i}')
        axes[row, col].legend()
    fig.delaxes(axes[1, 2])
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'Split_Into_5Fold.png'))

    return data_list_Total5Fold

def Split_XY(data_list_Total5Fold):
    '''
    把5個fold分X跟y
    '''
    X_5Fold_list = []; y_5Fold_list = []
    for i in range(5):
        X_5Fold_list.append(data_list_Total5Fold[i].drop('Group', axis = 'columns'))
        y_5Fold_list.append(data_list_Total5Fold[i]['Group'])

    return X_5Fold_list, y_5Fold_list

def To_nparray(X_5Fold_list, y_5Fold_list):
    
    for i in range(5):
        X_5Fold_list[i] = np.array(X_5Fold_list[i])
        y_5Fold_list[i] = np.array(y_5Fold_list[i])
    
    return X_5Fold_list, y_5Fold_list

def shuffle_XY(X, y):
    '''
    用shuffle_idx來打亂資料
    '''
    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(X)) # shuffle
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    return X, y

def preprocess_more_hsc(data):
    
    X = data.drop('Group', axis = 'columns')
    y = data['Group']
    X, y = np.array(X), np.array(y)

    return X, y