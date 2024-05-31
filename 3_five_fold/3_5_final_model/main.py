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
main_data = pd.read_excel('../../2_Data_preprocess(second)/Output/08_shuffle.xlsx', engine='openpyxl')
main_data = pd.DataFrame(main_data)
main_data = main_data.reset_index(drop=True)

HSC_redundant = pd.read_excel('../../2_Data_preprocess(second)/Output/07(2)_Split_HSC_Test(HSC_redundant).xlsx', engine='openpyxl')
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

# final model
model_building.Run_5_fold(data_5fold, HSC_redundant, folder_name,
            is_ClassWeight, isCopy, isENN, isENNaddSMOTE, is_TwoModel,
            feature_engi, Num_f, imputation)