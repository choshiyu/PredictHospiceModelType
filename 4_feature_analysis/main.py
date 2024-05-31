# -*- coding:utf-8 -*-
import pandas as pd
import warnings
import process_function
import model_building
warnings.filterwarnings('ignore')
import os

folder_name = 'Output'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

#read data
main_data = pd.read_excel('../2_Data_preprocess(second)/Output/08_shuffle.xlsx', engine='openpyxl')
main_data = pd.DataFrame(main_data)
main_data = main_data.reset_index(drop=True)

HSC_redundant = pd.read_excel('../2_Data_preprocess(second)/Output/07(2)_Split_HSC_Test(HSC_redundant).xlsx', engine='openpyxl')
HSC_redundant = pd.DataFrame(HSC_redundant)
HSC_redundant = HSC_redundant.reset_index(drop=True)

# Split_Into_5Fold
K = 30
data_5fold = process_function.Split_Into_5Fold(main_data, folder_name, K)

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

# teacher student model - DT student performance
# 先用5 fold驗證DT學習SVM的能力
# model_building.DT_learning(data_5fold, HSC_redundant, 'svm', Num_f, imputation)
# model_building.DT_learning(data_5fold, HSC_redundant, 'DT_student', Num_f, imputation)

# plot DT student
# catboost、xgb importances
# base_model_list = ['DT_student', ]
base_model_list = ['catboost', 'xgb']
for base_model in base_model_list:
    model_building.Show_feature_importance(data_5fold, HSC_redundant, base_model, Num_f, folder_name)