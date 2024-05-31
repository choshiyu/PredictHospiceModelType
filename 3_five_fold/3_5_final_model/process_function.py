import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight
    
def Split_Into_3Group(data, folder_name):
    '''
    把資料分依據Group類別分成三份
    '''
    data_hsc = data.loc[(data['Group'] == 2)] #hsc
    data_hpc = data.loc[(data['Group'] == 1)] #hpc
    data_hhc = data.loc[(data['Group'] == 0)] #hhc
    data_list = [data_hsc, data_hpc, data_hhc]
    
    for i, data in enumerate(data_list):
        file_name = f'01({i + 1})_Split_Into_3Group_{i + 1}.xlsx'
        file_path = os.path.join(folder_name, file_name)
        data.to_excel(file_path, index=False)
        
    return data_list

def Split_Into_Kcluster(data_list, K):
    '''
    三份data依據K來分成K群，把對資料點的分群記錄下來
    舉例 --> a點被分到第2群，a的位置紀錄:2  回傳一個list，裡面是三群分別的紀錄
    '''
    K_cluster_List = []
    for i in range(3):
        kmeansModel = KMeans(n_clusters=K, random_state=46)
        clusters_predicted = kmeansModel.fit_predict(data_list[i])
        K_cluster_List.append(clusters_predicted)

    return K_cluster_List

def Equally_Partition_Data(data_list, K, K_cluster_List):
    '''
    取得分群資料後，把三份資料各自依據他們分K群的資料平均分成5份
    在最後回傳的cluster_list中有三個群(依據group)
    每個group中有5個List，其中一個list裡面有K群(已合併好)
    '''
    cluster_list = [[] for _ in range(3)]
    cluster_list = [[[] for _ in range(5)] for _ in range(3)]

    for j in range(3):
        for i in range(K): # k群各切五份
            filt = (K_cluster_List[j] == i) # 依據分群紀錄K_cluster_List設一個filter
            # 要切5份所以設定一個0.2的point，+0.4是我發現這樣分完會比較平均
            point = int(0.2*len(data_list[j][filt])+0.4)
            # append到第j個group的五個fold中
            cluster_list[j][0].append(data_list[j][filt][:point])
            cluster_list[j][1].append(data_list[j][filt][point:2*point])
            cluster_list[j][2].append(data_list[j][filt][2*point:3*point])
            cluster_list[j][3].append(data_list[j][filt][3*point:4*point])
            cluster_list[j][4].append(data_list[j][filt][4*point:])

        for i in range(5): # 第j個group中的第i個fold裡面的k群資料合併起來
            cluster_list[j][i] = pd.concat(cluster_list[j][i])

    return cluster_list

def Split_Into_5Fold(main_data, folder_name, K):
    
    data_list = Split_Into_3Group(main_data, folder_name)
    k_cluster_list = Split_Into_Kcluster(data_list, K)
    
    with open(os.path.join(folder_name, f'02_Split_Into_Kcluster.txt'), 'w') as f:
        print(k_cluster_list, file=f)
        
    data_list_5FoldEachGroup = Equally_Partition_Data(data_list, K, k_cluster_list)

    data_list_Total5Fold = []
    for i in range(5):
        data_list_Total5Fold.append(pd.concat([data_list_5FoldEachGroup[0][i],data_list_5FoldEachGroup[1][i],data_list_5FoldEachGroup[2][i]]))

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
    plt.savefig(os.path.join(folder_name, '03_Split_Into_5Fold.png'))

    return data_list_Total5Fold

def Split_XY(data_list_Total5Fold, HSC_redundant):
    '''
    把5個fold分X跟y
    '''
    X_5Fold_list = []; y_5Fold_list = []
    for i in range(5):
        X_5Fold_list.append(data_list_Total5Fold[i].drop('Group', axis=1))
        y_5Fold_list.append(data_list_Total5Fold[i]['Group'])
    
    HSC_X_redundant = HSC_redundant.drop('Group', axis=1)
    HSC_y_redundant = HSC_redundant['Group']

    return X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant

def To_nparray(X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant):
    
    for i in range(5):
        X_5Fold_list[i] = np.array(X_5Fold_list[i])
        y_5Fold_list[i] = np.array(y_5Fold_list[i])
    HSC_X_redundant = np.array(HSC_X_redundant)
    HSC_y_redundant = np.array(HSC_y_redundant)
    
    return X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant

def shuffle_XY(X, y):
    '''
    用shuffle_idx來打亂資料
    '''
    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(X)) # shuffle
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    return X, y

def PCA_DimensionalityReduction(X_train, X_test, n_Dimension, folder_name):

    pca = PCA(n_components=n_Dimension)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], s=10, alpha=0.7)
    plt.title('PCA')
    plt.savefig(os.path.join(folder_name, 'PCA.png'))

    return X_train_pca, X_test_pca

def RandomForest_FeatureSelect(X_train, y_train, threshold, folder_name, X_col, i):
    
    x = pd.DataFrame(X_train, columns=X_col)
    y = pd.DataFrame(y_train, columns=['Group'])

    importance = 0
    rf = RandomForestClassifier(n_estimators=300, random_state=26)
    for _ in range(100):  # Repeat 100 times to get overall results
        rf.fit(x, y)
        importance += rf.feature_importances_
    
    importance_map = dict(zip(x.columns, importance))
    sorted_importance = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)
    sorted_features = [feature for feature, _ in sorted_importance]
    selected_features = sorted_features[:threshold]

    # Plot feature importances
    plt.figure(figsize=(16, 6))
    plt.bar(x.columns, importance)
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.subplots_adjust(bottom=0.6)
    plt.savefig(os.path.join(folder_name, f'{i}fold_RF_feature_importances.png'))

    return selected_features

def RFE_FeatureSelect(X_train, y_train, threshold, X_col, base_model):

    x = pd.DataFrame(X_train, columns=X_col)
    if base_model == 'catboost':
        y = y_train
    else:
        y = pd.DataFrame(y_train, columns=['Group'])
    
    # decide est
    if base_model == 'lr':
        est = LogisticRegression(random_state=42, class_weight = 'balanced')  
    elif base_model == 'catboost':
        classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y)
        est = CatBoostClassifier(random_state=42)
    elif base_model == 'rf':
        est = RandomForestClassifier(random_state=42, class_weight = 'balanced')
    elif base_model == 'xgb':
        classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y)
        est = xgb.XGBClassifier(random_state=42)
    elif base_model == 'mlp' or 'svm':
        est = SVC(kernel='linear', random_state=42, class_weight = 'balanced')

    # build RFE
    rfe = RFE(estimator=est, n_features_to_select=threshold)
    
    if base_model == 'xgb' or base_model == 'catboost':
        rfe_selector = rfe.fit(x, y, sample_weight=classes_weights)
    else:
        rfe_selector = rfe.fit(x, y)

    selected_feature_indices = rfe_selector.get_support(indices=True)
    selected_features = (x.columns[selected_feature_indices]).tolist()

    return selected_features

def VT_FeatureSelect(X_train, threshold, X_col):

    x = pd.DataFrame(X_train, columns=X_col)
    
    feature_variances = np.var(x, axis=0)
    sorted_feature_indices = np.argsort(feature_variances)[::-1]
    selected_feature_indices = sorted_feature_indices[:threshold]
    selected_features = x.columns[selected_feature_indices].tolist()
    
    return selected_features

def VT_FeatureSelect(X_train, threshold, X_col):

    x = pd.DataFrame(X_train, columns=X_col)
    
    feature_variances = np.var(x, axis=0)
    sorted_feature_indices = np.argsort(feature_variances)[::-1]
    selected_feature_indices = sorted_feature_indices[:threshold]
    selected_features = x.columns[selected_feature_indices].tolist()
    
    return selected_features

def Statistical_FeatureSelect(X_col):

    remove_list = [
    'sex_Female', 'sex_Male', 'new_year_LunarNewYear',
    'new_year_NonChineseNewYear', 'ECOG', 'tumor_ulcer_wound']
    
    selected_features = list(filter(lambda x: x not in remove_list, X_col))
    
    return selected_features