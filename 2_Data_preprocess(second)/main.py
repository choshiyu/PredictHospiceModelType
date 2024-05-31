# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import preprocess_function
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
import os

# 存放各階段Output的資料夾
folder_name = 'Output'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

#read data
data = pd.read_excel('../1_Data_preprocess(first)/Output/08_DiseaseDiagnosis_Transcoding.xlsx', engine='openpyxl')
data = pd.DataFrame(data)
data = data.reset_index(drop=True)

# 把0-1、2-3那種改為連續變項(且字串轉數字)
data = preprocess_function.StrToInt(data)
data.to_excel((os.path.join(folder_name, '01_StrToInt.xlsx')), index=False)

# 把結果三種group轉為1、2、3
data['Group'] = data['Group'].map({'HHC': 0, 'HPC': 1, 'HSC': 2})
data.to_excel((os.path.join(folder_name, '02_GroupTo_012.xlsx')), index=False)

# 只有age先做normalization
data = preprocess_function.age_MinMaxNormalization(data)
data.to_excel((os.path.join(folder_name, '03_Only_age_MinMaxNormalization.xlsx')), index=False)

# 無缺值的作為main data，有缺值的5fold中跟當次train data合併，預測填補缺值後作為當次train data
data_tobe_impute = data[data.isna().any(axis=1)].copy()
missing_values_per_row_of_data_tobe_impute = data_tobe_impute.isnull().sum(axis=1)# 判斷每一列中缺失值的數量
data_tobe_impute.to_excel((os.path.join(folder_name, '04(1)_data_tobe_impute.xlsx')), index=False)

main_data = data.dropna() # main會是沒有缺值的
main_data.reset_index(drop=True, inplace=True)
main_data.to_excel((os.path.join(folder_name, '04(2)_main_data.xlsx')), index=False)

# one hot encoding
main_data, OneHotColumnsName, cat_ori_cols = preprocess_function.OneHotEncoding(main_data)
with open(os.path.join(folder_name, '05(2)_OneHotColumnsName.txt'), 'w') as f:
    for item in OneHotColumnsName:
        f.write("%s\n" % item)
with open(os.path.join(folder_name, '05(3)_cat_ori_cols.txt'), 'w') as f:
    for item in cat_ori_cols:
        f.write("%s\n" % item)
main_data.to_excel((os.path.join(folder_name, '05(1)_OneHotEncoding_maindata.xlsx')), index=False)

# MinMaxNormalization
main_data = preprocess_function.MinMaxNormalization(main_data)
main_data.to_excel((os.path.join(folder_name, '06_MinMaxNormalization_maindata.xlsx')), index=False)

# Split_HSC_Test
main_data, HSC_redundant = preprocess_function.Split_HSC_Test(main_data)
main_data.to_excel((os.path.join(folder_name, '07(1)_Split_HSC_Test(main_data).xlsx')), index=False)
HSC_redundant.to_excel((os.path.join(folder_name, '07(2)_Split_HSC_Test(HSC_redundant).xlsx')), index=False)
# 畫圖
pca_A = PCA(n_components=2)
data_2d_A = pca_A.fit_transform(main_data.loc[(main_data['Group'] == 2)])
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(data_2d_A[:, 0], data_2d_A[:, 1], s=10, alpha=0.7, label='HSC_in5fold')
plt.title('HSC_in5fold')
plt.legend()
pca_B = PCA(n_components=2)
data_2d_B = pca_B.fit_transform(HSC_redundant)
plt.subplot(1, 2, 2)
plt.scatter(data_2d_B[:, 0], data_2d_B[:, 1], s=10, alpha=0.7, label='HSC_redundant')
plt.title('HSC_redundant')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(folder_name, '07(3)_Split_HSC_Test.png'))

# shuffle
main_data = main_data.sample(frac=1, random_state=24).reset_index(drop=True)
main_data.to_excel((os.path.join(folder_name, '08_shuffle.xlsx')), index=False)
