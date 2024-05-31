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
main_data = pd.read_excel('../2.Data preprocess(second)/Output/08_shuffle.xlsx', engine='openpyxl')
main_data = pd.DataFrame(main_data)
main_data = main_data.reset_index(drop=True)

HSC_redundant = pd.read_excel('../2.Data preprocess(second)/Output/07(2)_Split_HSC_Test(HSC_redundant).xlsx', engine='openpyxl')
HSC_redundant = pd.DataFrame(HSC_redundant)
HSC_redundant = HSC_redundant.reset_index(drop=True)

# 保留容易弄混的HHC HSC，並把HSC原本是2改成1
main_data = main_data[main_data['Group'] != 1]
main_data.loc[main_data['Group'] == 2, 'Group'] = 1
HSC_redundant.loc[HSC_redundant['Group'] == 2, 'Group'] = 1

# 幾天
day = 7
few_day_data = main_data[(main_data['period'] >= 0) & (main_data['period'] <= day)]
main_data = main_data[(main_data['period'] < 0) | (main_data['period'] > day)]
few_day_data.drop(['period'], axis=1, inplace=True)
main_data.drop(['period'], axis=1, inplace=True)
HSC_redundant.drop(['period'], axis=1, inplace=True)

print(f'len(period <= {day}):', len(few_day_data))
print('HHC:', len(few_day_data[few_day_data['Group'] == 0]))
print('HSC:', len(few_day_data[few_day_data['Group'] == 1]))

print('len(other):', len(main_data))
print('HHC:', len(main_data[main_data['Group'] == 0]))
print('HSC:', len(main_data[main_data['Group'] == 1]))

# Split_Into_5Fold
fewdays_5fold = process_function.Split_Into_5Fold(few_day_data, folder_name)
other_5fold = process_function.Split_Into_5Fold(main_data, folder_name)

# base_model
base_model = 'svm'
model_building.Run_5_fold(fewdays_5fold, other_5fold, day, HSC_redundant, 'predict_few_days')
model_building.Run_5_fold(fewdays_5fold, other_5fold, day, HSC_redundant, 'predict_others')