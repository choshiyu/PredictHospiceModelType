# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import os
import preprocess_function
import joblib
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

folder_name = 'Output'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

#read data
data = pd.read_excel('../../0_raw_data/original.xlsx', engine='openpyxl')
data = pd.DataFrame(data)
data = data.reset_index(drop=True)

# 讀入OneHotCols，要確保等下X_test要有每個OneHotCols
OneHotCols = []
with open('../../2_Data_preprocess(second)/Output/05(2)_OneHotColumnsName.txt', 'r', encoding='utf-8') as file:
    for line in file:
        OneHotCols.append(line.strip())

# random select one row data
data.dropna(axis=0, inplace=True) # 避免取到含缺值的資料點
data = data.sample(n=1, random_state=32)

# do all the preprocess
data = preprocess_function.all_preprocess(data, OneHotCols)
X_test = data.drop('Group', axis=1)
y_test = data['Group']

# load train data
train = pd.read_excel('../DT_tree/data_for_1sample_analysis.xlsx', engine='openpyxl')
X = train.drop('Group', axis=1)

X_test = X_test[X.columns] # X_train有特徵選擇過，所以要確保X_test特徵與X_train一致

# load model
model = joblib.load('../DT_tree/tree_student_model.pkl')
with open(os.path.join(folder_name, f'TrueAnswer.txt'), 'w') as f:
    print(f'###y_true -->{y_test}\n', file=f) # true answer

# visualize_tree_with_path
filepath = os.path.join(folder_name, 'decision_tree_with_path')
preprocess_function.visualize_tree_with_path(model, X, X_test, filepath)