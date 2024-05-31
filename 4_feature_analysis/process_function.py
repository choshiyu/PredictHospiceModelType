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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import math
    
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

def Statistical_FeatureSelect(X_col):

    remove_list = [
    'sex_Female', 'sex_Male', 'new_year_LunarNewYear',
    'new_year_NonChineseNewYear', 'ECOG', 'tumor_ulcer_wound']
    
    selected_features = list(filter(lambda x: x not in remove_list, X_col))
    
    return selected_features

def get_data():
    
    OneHotColumnsName = []
    with open('../2_Data_preprocess(second)/Output/05(2)_OneHotColumnsName.txt', 'r', encoding='utf-8') as file:
        for line in file:
            OneHotColumnsName.append(line.strip())
    
    cat_ori_cols = []
    with open('../2_Data_preprocess(second)/Output/05(3)_cat_ori_cols.txt', 'r', encoding='utf-8') as file:
        for line in file:
            cat_ori_cols.append(line.strip())
            
    data_tobe_impute = pd.read_excel('../2_Data_preprocess(second)/Output/04(1)_data_tobe_impute.xlsx', engine='openpyxl')
    data_tobe_impute = pd.DataFrame(data_tobe_impute)
    
    data_tobe_impute = data_tobe_impute[data_tobe_impute['Group'] != 2] # HSC不用補
    data_tobe_impute = data_tobe_impute.reset_index(drop=True)
    
    missing_values_per_row_of_data_tobe_impute = data_tobe_impute.isnull().sum(axis=1)# 判斷每一列中缺失值的數量
    data_tobe_impute.drop(data_tobe_impute[missing_values_per_row_of_data_tobe_impute > 10].index, inplace=True)
    data_tobe_impute.reset_index(drop=True, inplace=True)
            
    return OneHotColumnsName, cat_ori_cols, data_tobe_impute

def reinstatement_onehot(OneHotColumnsName, X_train):
    
    X_train_ToBeReinstatement = X_train[OneHotColumnsName] # 取得轉過onehot的X_train(cat)
    X_train.drop(OneHotColumnsName, axis=1, inplace=True) # 保留con data
    
    prefix_dict = {} # 遍歷列名，將具有相同prefix的列合併成新的dataframe，存在prefix_dict
    for col in X_train_ToBeReinstatement: # 遍歷列
        prefix = col.rsplit('_', 1)[0] # 取得以_切割字串後的前面部分
        
        if prefix in prefix_dict: # prefix已出現過的話
            prefix_dict[prefix].append(col)  # 將當前col跟已出現的相同prefix合併成list
        else:
            prefix_dict[prefix] = [col] # prefix還沒出現的話將col直接放到字典中

    # 現在prefix_dict中有多個list，每個list要在後面被合併回一column

    prefix_order = []
    df_list = [] # 遍歷字典，將具有相同prefix的列合併成dataframe，存在df_list中
    for prefix, cols in prefix_dict.items():
        df_subset = X_train_ToBeReinstatement[cols] # 一個df_subset要合併成一個col
        df_subset.columns = [col.rsplit('_', 1)[1] for col in df_subset.columns]  # 重新命名列為_後的字串
        df_list.append(df_subset)
        prefix_order.append(prefix)
        
    # df_list中有多個df_subset，df_subset中的列名是_後的字串
    
    df_n_list = []
    for df in df_list:
        df_n = df.apply(lambda x: x.idxmax(), axis = 1) # df_n 是一個 Series，其中包含了每行的最大值所在的欄位名
        df_n_list.append(df_n)
    
    combined_df = pd.concat(df_n_list, axis=1)
    combined_df.columns = prefix_order
    data = pd.concat([X_train, combined_df], axis=1)
    
    return data

def convert_to_str(value):
    
    if isinstance(value, str):  # 如果已是str
        return value # 不用改直接回傳
    elif not math.isnan(value):  # 如果不是缺值
        return str(value)  # 轉str後回傳
    
    return value

def OneHotEncoding(data, cat_ori_columns):
    
    cat_data = data[cat_ori_columns]
    other = data.drop(columns=cat_ori_columns)
    
    OneHotDone_data = pd.get_dummies(cat_data) # 做OneHotEncoding
    data = pd.concat([other, OneHotDone_data], axis=1)

    return data

def MICEImpute(OneHotColumnsName, X_col, X_train, y_train,
               data_tobe_impute, folder_name, cat_ori_columns):

    with open(os.path.join(folder_name, 'MICEImpute_alldata.txt'), 'w') as f:
        
        print("\n\n\nbefore MICEImpute: len(label 0)", len(np.where(y_train == 0)[0]),
            ', len(label 1)', len(np.where(y_train == 1)[0]),
            ', len(label 2)', len(np.where(y_train == 2)[0]), file=f)
    
        X_train = pd.DataFrame(X_train, columns=X_col)
        y_train = pd.DataFrame(y_train, columns=['Group'])

        maindata = reinstatement_onehot(OneHotColumnsName, X_train)
        maindata['Group'] = y_train['Group']
        
        combined_list = []
        for col in maindata.columns:# 把有缺值跟沒缺值的資料合併一起
            maindata_column = maindata[col]
            data_tobe_impute_column = data_tobe_impute[col]
            combined_list.append(pd.concat([maindata_column, data_tobe_impute_column], ignore_index=True))
        data = pd.concat(combined_list, axis=1)

        for column_name in cat_ori_columns:
            data[column_name] = data[column_name].map(convert_to_str)

        data = OneHotEncoding(data, cat_ori_columns) # one hot回傳進來時的狀態
        
        imputer = IterativeImputer(random_state=58, max_iter=10)
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        X_train = data.drop('Group', axis=1)
        y_train = data['Group']
         
        print("after MICEImpute: len(label 0)", len(np.where(y_train == 0)[0]),
            ', len(label 1)', len(np.where(y_train == 1)[0]),
            ', len(label 2)', len(np.where(y_train == 2)[0]), file=f)
    
    return X_train, y_train