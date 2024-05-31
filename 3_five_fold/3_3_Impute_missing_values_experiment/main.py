# -*- coding:utf-8 -*-
import pandas as pd
import warnings
import process_function
import model_building
import impute_missing_data
warnings.filterwarnings('ignore')
import os

folder_name = 'Output'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

#read data
main_data = pd.read_excel('../../2_Data_preprocess(second)/Output/08_shuffle.xlsx', engine='openpyxl')
main_data = pd.DataFrame(main_data)
main_data = main_data.reset_index(drop=True)

HSC_redundant = pd.read_excel('../../2_Data_preprocess(second)/Output/07(2)_Split_HSC_Test(HSC_redundant).xlsx', engine='openpyxl')
HSC_redundant = pd.DataFrame(HSC_redundant)
HSC_redundant = HSC_redundant.reset_index(drop=True)

# Split_Into_5Fold
K = 30
data_5fold = process_function.Split_Into_5Fold(main_data, folder_name, K)

# base_model
base_model_list = ['lr', 'catboost', 'rf', 'xgb', 'mlp', 'svm']

# imbalance processing
is_ClassWeight = True # set ClassWeight
isCopy = False
isENN = False
isENNaddSMOTE = False
is_TwoModel = False

# feature engineering
# feature_engi_list = ['pca', 'rf', 'rfe', 'vt', 'Statistical_Analysis']
# Num_f_list = [20,30,40,51]
feature_engi = 'rfe'; Num_f = 51 # set feature_engi &　Num_f

# imputation - 目前補完的資料不適用isCopy、isENN、isENNaddSMOTE、'pca'，要的話要在impute_missing_data中調整
imputation_method_list = ['mice', 'missforest', 'knn']
for imputation in imputation_method_list:
    impute_missing_data.save_impute_data(data_5fold, HSC_redundant, imputation)

# run 5 fold (all experiments)
for base_model in base_model_list:
    for imputation in imputation_method_list:
        model_building.Run_5_fold(data_5fold, HSC_redundant, folder_name, base_model,
                    is_ClassWeight, isCopy, isENN, isENNaddSMOTE, is_TwoModel,
                    feature_engi, Num_f, imputation)