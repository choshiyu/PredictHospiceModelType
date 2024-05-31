import pandas as pd
import numpy as np
from scipy.stats import kruskal
from scipy.stats import chi2_contingency
import os

def Kruskal(List_3, all):

    stats, p = kruskal(List_3[0],List_3[1],List_3[2])
    all = all.values # df型態改成np型態才能%.3f
    print('p=%.3f' % p, file=f) # 統計量、p值
    alpha = 0.05
    print("fail to reject H0" if p > alpha else "reject H0", file=f)
    
    print('all data\'s mean(sd)=%.3f(%.3f)' % (np.mean(all), np.std(all)), file=f)
    print('HHC\'s mean(sd)=%.3f(%.3f)' % (np.mean(List_3[0]) , np.std(List_3[0])), file=f) # mean/sd
    print('HPC\'s mean(sd)=%.3f(%.3f)' % (np.mean(List_3[1]) , np.std(List_3[1])), file=f) # mean/sd
    print('HSC\'s mean(sd)=%.3f(%.3f)\n' % (np.mean(List_3[2]) , np.std(List_3[2])), file=f) # mean/sd

def ChiContingency(table):

    obs = (np.array(table.iloc[0][0:].values) ,np.array(table.iloc[1][0:].values), np.array(table.iloc[2][0:].values))
    chi2, p, dof, expected = chi2_contingency(obs, correction = False) # 2*2列聯表才要=True(葉慈修正)
    print('p=%.3f' % p, file=f)

    alpha = 0.05
    print("fail to reject H0" if p > alpha else "reject H0", file=f)

# 存放各階段Output的資料夾
folder_name = 'Output'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

#read data
df = pd.read_excel('../Output/08_DiseaseDiagnosis_Transcoding.xlsx', engine='openpyxl')
data = pd.DataFrame(df)
data = data.reset_index(drop=True)

data = data.dropna()
data.reset_index(drop=True, inplace=True)

continuous_data = data[['Group', 'age']]# 連續特徵跑KruskalWallis
categorical_data = data.drop(['age'], axis = 'columns')# 類別特徵跑卡方

# 連續資料依據group分成三部分
Continuous_List = [group.drop('Group', axis='columns') for _, group in continuous_data.groupby('Group')]
continuous_data_no_Group = continuous_data.drop('Group', axis='columns')

with open(os.path.join(folder_name, 'output.txt'), 'w') as f:
    
    # --------------------Kruskal-Wallis------------------------
    for ColumnName in continuous_data_no_Group.columns:
        
        KW_InputList = [[], [], []]
        for i in range(3):
            KW_InputList[i] = np.array(Continuous_List[i][ColumnName])

        print('##', ColumnName, file=f)
        Kruskal(KW_InputList, continuous_data_no_Group)
    '''
    # example
    ex_data = []
    ex_num_list = np.array([21.2, 22.9, 30.9, 32.1, 37.5, 43.2, 47.0, 52.8, 50.4, 51.5, 47.1, 54.8])
    for i in range(4):
        ex_data.append(ex_num_list)
        if i == 3:
            ex_data.append(ex_num_list + 50)

    kruskal(ex_data[0], ex_data[1], ex_data[2], 0) #0、1、2
    kruskal(ex_data[0], ex_data[1], ex_data[2], 0) # 0、1、3
    '''
    # --------------------chi2_contingency------------------------
    list_table = pd.DataFrame() # 每次迴圈都會把新資料concat上去

    for columns_name in categorical_data.columns:
        
        table = pd.crosstab(categorical_data['Group'], categorical_data[columns_name], margins=True)
        ChiContingency(table)
        
        opposite_table = pd.crosstab(categorical_data[columns_name], categorical_data['Group'], margins=True, normalize='columns')  
        opposite_table = (opposite_table * 100).applymap(lambda x: "({:.1f}%)".format(x))
        transposed_table = (table.T).iloc[:-1, :]

        table = transposed_table.astype(str) + opposite_table.astype(str)
        list_table = pd.concat([list_table, table]) # 把新資料concat上去
        print(table, '\n', file=f)

    list_table.to_excel((os.path.join(folder_name, 'output.xlsx')), index=False)
    '''
    # example 
    ex_data = data[['Group', 'sex', 'referral']][0:5]

    for ex_columns_name in ex_data.columns:
        
        ex_table = pd.crosstab(ChiContingency_data['Group'], ChiContingency_data[columns_name], margins=True)
        ex_opposite_table = pd.crosstab(ChiContingency_data[columns_name], ChiContingency_data['Group'], margins=True, normalize='columns')  
        print('-'*15, ex_columns_name, '-'*15)
        ChiContingency(ex_table)
        print(ex_opposite_table, '\n')
    '''