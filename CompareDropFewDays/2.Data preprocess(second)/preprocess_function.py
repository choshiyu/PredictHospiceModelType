import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.cluster import KMeans
import random

def StrToInt(data):
    '''
    程度的特徵value直接轉成數字型態，作為連續變項
    '''  
    data['ECOG'] = data['ECOG'].map({'0-2': 0, '3-4': 1})
    data['pain'] = data['pain'].map({'0-3': 0, '4-6': 1, '7-10': 2})
    symptom_mapping = {'0': 0, '1-2': 1, '3-4': 2}
    columns_to_map = ['dyspnea', 'nausea', 'constipation', 'dysphagia', 'tumor_ulcer_wound',
                      'ascites', 'lymphedema', 'general_edema', 'fatigue', 'insomnia', 'incontinence']

    for column in columns_to_map:
        data[column] = data[column].map(symptom_mapping)

    return data

def OneHotEncoding(data):
    '''
    將類別變項做OneHotEncoding
    '''
    data_ToBeOneHot = data[['sex','DNR','pain_medication','dyspnea_medication',
                        'psychosocial_issues_patient','psychosocial_issues_family',
                        'spirituality_religion_patient','spirituality_religion_family',
                        'referral','diagnosis','religious_beliefs','new_year',
                        'patient_awareness_diagnosis','patient_awareness_prognosis',
                        'family_awareness_prognosis']]
    cat_ori_cols = data_ToBeOneHot.columns
    data.drop(data_ToBeOneHot.columns, axis=1, inplace=True)
    
    OneHotDone_data = pd.get_dummies(data_ToBeOneHot) # 做OneHotEncoding
    OneHotColumnsName = OneHotDone_data.columns
    data = pd.concat([data, OneHotDone_data], axis=1) # 把OneHotDone_data跟其他data合併回來
    
    return data, OneHotColumnsName, cat_ori_cols

def age_MinMaxNormalization(data):
    '''
    age做MinMaxNormalization，因為其他數值特徵有缺，後面還要補資料，先不能做normalization
    '''
    continuous_data = data[['age']]
    data.drop(['age'], axis=1, inplace=True)
    
    columns = continuous_data.columns.values
    scaler = MinMaxScaler()
    continuous_data = scaler.fit_transform(continuous_data.astype(np.float64))
    continuous_data = pd.DataFrame(continuous_data)
    continuous_data.columns = columns
    data = pd.concat([continuous_data, data], axis=1) # 把continuous_data跟categorical_data

    return data

def MinMaxNormalization(data):

    continuous_data = data[['ECOG','pain','dyspnea','nausea',
            'constipation','dysphagia','tumor_ulcer_wound','ascites',
            'lymphedema','general_edema','fatigue','insomnia','incontinence']]
    # print(continuous_data.head())
    data.drop(continuous_data.columns, axis=1, inplace=True)
    
    columns = continuous_data.columns.values
    scaler = MinMaxScaler()
    continuous_data = scaler.fit_transform(continuous_data.astype(np.float64))
    continuous_data = pd.DataFrame(continuous_data)
    continuous_data.columns = columns
    # print(continuous_data.head())
    data = pd.concat([continuous_data, data], axis=1) # 把continuous_data跟categorical_data

    return data

def Split_HSC_Test(data):
    
    data_tmp = data.drop(['period'], axis=1)
    tmp_hsc = data_tmp.loc[(data_tmp['Group'] == 2)] #hsc
    
    data_hsc = data.loc[(data['Group'] == 2)] #hsc
    data = data.loc[(data['Group'] != 2)] #!=hsc

    k = 50 # 肉眼觀察分的較平均的k
    kmeansModel = KMeans(n_clusters=k, random_state=46)
    clusters_predicted = kmeansModel.fit_predict(tmp_hsc)
    
    HSC_redundant = []; HSC_in5fold = []
    Decide_point = random.sample([0]*34 + [1]*16, k)
    for i in range(k):
        filt = (clusters_predicted == i) # 依據分群紀錄K_cluster_List設一個filter
        point = 14 if Decide_point[i] == 0 else 15
        HSC_in5fold.append(data_hsc[filt][:point])
        HSC_redundant.append(data_hsc[filt][point:])
        
    HSC_in5fold = pd.concat(HSC_in5fold, ignore_index=True)
    print('HSC_in5fold', len(HSC_in5fold))
    data = pd.concat([HSC_in5fold, data], ignore_index=True)
    HSC_redundant = pd.concat(HSC_redundant, ignore_index=True)
    print('HSC_redundant', len(HSC_redundant))
    
    return data, HSC_redundant