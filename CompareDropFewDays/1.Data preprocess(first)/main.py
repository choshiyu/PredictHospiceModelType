# -*- coding:utf-8 -*-
import pandas as pd
import warnings
import preprocess_function
import os
warnings.filterwarnings('ignore')

# 存放各階段Output的資料夾
folder_name = 'Output'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

#read data
data = pd.read_excel('../0.raw_data/original.xlsx', engine='openpyxl')
data = pd.DataFrame(data)
data = data.reset_index(drop=True)

# Rename_ChineseToEnglish
data = preprocess_function.Rename_ChineseToEnglish(data)
data.to_excel((os.path.join(folder_name, '01_Rename_ChineseToEnglish.xlsx')), index=False)

# 保留結案原因是死亡、瀕死跟症狀改善的(代表符合研究目標)
data = preprocess_function.delete_NotDeath(data)
data.to_excel((os.path.join(folder_name, '02_delete_NotDeath.xlsx')), index=False)

# “religious_beliefs”中的缺值歸類至民間信仰裡面
data = preprocess_function.religion_replace_na_with_folk(data)
data.to_excel((os.path.join(folder_name, '03_religion_replace_na_with_folk.xlsx')), index=False)

# 把'IsSignDNR', 'DNR_timing'合併成(No/before/after)
data['DNR'] = data.apply(preprocess_function.Combine_DNRtiming_and_IsSignDNR, axis=1)
data = data.drop(['IsSignDNR', 'DNR_timing'], axis = 'columns')
data.to_excel((os.path.join(folder_name, '04_Combine_DNRtiming_and_IsSignDNR.xlsx')), index=False)

# 'pain_medication'、'dyspnea_medication'分為opioid跟non-opioid
data = preprocess_function.medicine_morphine(data)
data.to_excel((os.path.join(folder_name, '05_medicine_morphine.xlsx')), index=False)

# 收案日期改為春節或非春節
data = preprocess_function.StartDay_To_lunar_New_Year(data)
data.to_excel((os.path.join(folder_name, '06_StartDay_To_lunar_New_Year.xlsx')), index=False)

# create period
period_time = pd.DataFrame(((data['discharge_date'] - data['admission_date']).dt.days))
data = pd.concat([data, period_time], axis='columns')
data.rename(columns={0:'period'}, inplace=True)

# 把研究中不會用到的特徵drop掉
data.drop(['discharge_date', 'discharge_reason','IsCancer','patient_ID_number',
            'birthday','discharge_reason_1','primary_caregiver','admission_date'], axis=1, inplace=True)
data.to_excel((os.path.join(folder_name, '07_Delete_features_not_be_used.xlsx')), index=False)

# 疾病診斷處理(轉碼)
data = preprocess_function.DiseaseDiagnosis_Transcoding(data)
data.to_excel((os.path.join(folder_name, '08_DiseaseDiagnosis_Transcoding.xlsx')), index=False)