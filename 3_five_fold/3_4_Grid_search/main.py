# -*- coding:utf-8 -*-
import pandas as pd
import warnings
import process_function
import model_building
warnings.filterwarnings('ignore')
import os
from grid_search import Grid_Search

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

# set imbalance processing
is_ClassWeight = True
isCopy = False
isENN = False
isENNaddSMOTE = False
is_TwoModel = False

# set feature engineering
feature_engi = 'rfe'; Num_f = 51

# set imputation method
imputation = 'mice'

# grid search - 這個版本只適用目前選擇的處理方式:ClassWeight、RFE feature select、MICE impute
# for base_model in base_model_list:
#     Grid_Search(data_5fold, HSC_redundant, base_model, is_ClassWeight,
#                 feature_engi, Num_f, imputation)

# final result - 把cnf等等其他performance也跑出來
for base_model in base_model_list:
    model_building.Run_5_fold(data_5fold, HSC_redundant, folder_name, base_model,
                is_ClassWeight, isCopy, isENN, isENNaddSMOTE, is_TwoModel,
                feature_engi, Num_f, imputation)