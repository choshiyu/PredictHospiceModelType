import pandas as pd
import numpy as np
import process_function
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn.impute import KNNImputer
from missingpy import MissForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os
import math

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
               data_tobe_impute, folder_name, i, cat_ori_columns):

    mode = 'a' if i != 0 else 'w'
    with open(os.path.join(folder_name, 'MICEImpute.txt'), mode) as f:
        
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
    
    return data

def MissForestImpute(OneHotColumnsName, X_col, X_train, y_train,
               data_tobe_impute, folder_name, i, cat_ori_columns):
    
    mode = 'a' if i != 0 else 'w'
    
    with open(os.path.join(folder_name, 'MissForestImpute.txt'), mode) as f:
    
        X_train = pd.DataFrame(X_train, columns=X_col)
        y_train = pd.DataFrame(y_train, columns=['Group'])
        
        print("\n\n\nbefore MissForestImpute: len(label 0)", len(np.where(y_train == 0)[0]),
            ', len(label 1)', len(np.where(y_train == 1)[0]),
            ', len(label 2)', len(np.where(y_train == 2)[0]), file=f)
        
        maindata = reinstatement_onehot(OneHotColumnsName, X_train)
        maindata['Group'] = y_train['Group']

        combined_list = []
        for col in maindata.columns:# 把有缺值跟沒缺值的資料合併一起
            maindata_column = maindata[col]
            data_tobe_impute_column = data_tobe_impute[col]
            combined_list.append(pd.concat([maindata_column, data_tobe_impute_column], ignore_index=True))
        data = pd.concat(combined_list, axis=1)
        
        for col in data.columns:
            if data[col].dtype == 'O':# 將含有字串的列轉換為數值
                data[col] = pd.factorize(data[col])[0]

        # MissForest
        imputer = MissForest(random_state=52)
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        for col in maindata.columns:
            if maindata[col].dtype == 'O':
                data[col] = maindata[col].astype(str)# 將數值型態的值轉換回原始字串形式
                
        data = OneHotEncoding(data, cat_ori_columns) # one hot回傳進來時的狀態
        
        X_train = data.drop('Group', axis=1)
        y_train = data['Group']
        
        print("after MissForestImpute: len(label 0)", len(np.where(y_train == 0)[0]),
            ', len(label 1)', len(np.where(y_train == 1)[0]),
            ', len(label 2)', len(np.where(y_train == 2)[0]), file=f)

    return data

def KNNImpute(OneHotColumnsName, X_col, X_train, y_train,
               data_tobe_impute, folder_name, i, cat_ori_columns):

    mode = 'a' if i != 0 else 'w'
    
    with open(os.path.join(folder_name, 'KNNImpute.txt'), mode) as f:
    
        X_train = pd.DataFrame(X_train, columns=X_col)
        y_train = pd.DataFrame(y_train, columns=['Group'])
        
        print("\n\n\nbefore KNNImpute: len(label 0)", len(np.where(y_train == 0)[0]),
            ', len(label 1)', len(np.where(y_train == 1)[0]),
            ', len(label 2)', len(np.where(y_train == 2)[0]), file=f)
        
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
        
        data = OneHotEncoding(data, cat_ori_columns)
        
        knn_imputer = KNNImputer()
        data = pd.DataFrame(knn_imputer.fit_transform(data), columns=data.columns)

        X_train = data.drop('Group', axis=1)
        y_train = data['Group']
        
        print("after MICEImpute: len(label 0)", len(np.where(y_train == 0)[0]),
            ', len(label 1)', len(np.where(y_train == 1)[0]),
            ', len(label 2)', len(np.where(y_train == 2)[0]), file=f)
    
    return data

def get_data():
    
    OneHotColumnsName = []
    with open('../../2_Data_preprocess(second)/Output/05(2)_OneHotColumnsName.txt', 'r', encoding='utf-8') as file:
        for line in file:
            OneHotColumnsName.append(line.strip())
    
    cat_ori_cols = []
    with open('../../2_Data_preprocess(second)/Output/05(3)_cat_ori_cols.txt', 'r', encoding='utf-8') as file:
        for line in file:
            cat_ori_cols.append(line.strip())
            
    data_tobe_impute = pd.read_excel('../../2_Data_preprocess(second)/Output/04(1)_data_tobe_impute.xlsx', engine='openpyxl')
    data_tobe_impute = pd.DataFrame(data_tobe_impute)
    
    data_tobe_impute = data_tobe_impute[data_tobe_impute['Group'] != 2] # HSC不用補
    data_tobe_impute = data_tobe_impute.reset_index(drop=True)
    
    missing_values_per_row_of_data_tobe_impute = data_tobe_impute.isnull().sum(axis=1)# 判斷每一列中缺失值的數量
    data_tobe_impute.drop(data_tobe_impute[missing_values_per_row_of_data_tobe_impute > 10].index, inplace=True)
    data_tobe_impute.reset_index(drop=True, inplace=True)
            
    return OneHotColumnsName, cat_ori_cols, data_tobe_impute


def save_impute_data(data_list_Total5Fold, HSC_redundant, imputation):
    
    folder_name = f'save_impute_data_5fold/{imputation}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant = process_function.Split_XY(data_list_Total5Fold, HSC_redundant)
    X_col = X_5Fold_list[0].columns
    X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant = process_function.To_nparray(X_5Fold_list, y_5Fold_list, HSC_X_redundant, HSC_y_redundant)
        
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

        OneHotColumnsName, cat_ori_cols, data_tobe_impute = get_data()

        # Impute
        if imputation == 'mice':
            data = MICEImpute(OneHotColumnsName, X_col, X_train, y_train,
                                    data_tobe_impute, folder_name, i, cat_ori_cols)
        elif imputation == 'missforest':
            data = MissForestImpute(OneHotColumnsName, X_col, X_train, y_train,
                                    data_tobe_impute, folder_name, i, cat_ori_cols)
        elif imputation == 'knn':
            data = KNNImpute(OneHotColumnsName, X_col, X_train, y_train,
                                    data_tobe_impute, folder_name, i, cat_ori_cols)
        
        # save data
        data.to_excel(os.path.join(folder_name, f'{i}fold.xlsx'), index=False)