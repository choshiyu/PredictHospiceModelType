import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import graphviz
from sklearn.tree import export_graphviz

def Rename_ChineseToEnglish(data):
    '''
    Rename all Chinese to English
    '''
    data.rename(columns={
        '病歷號': 'patient_ID_number',
        '主要照顧者': 'primary_caregiver',
        '生日': 'birthday',
        '收案日期': 'admission_date',
        '性別': 'sex',
        '疾病診斷': 'diagnosis',
        '結案日期': 'discharge_date',
        '癌症病人': 'IsCancer',
        '簽署DNR同意書': 'IsSignDNR',
        '簽署DNR時間為收案前或後': 'DNR_timing',
        '症狀_疼痛_收案時': 'pain',
        '症狀_疼痛_藥物_收案時': 'pain_medication',
        '症狀_呼吸困難_收案時': 'dyspnea',
        '症狀_呼吸困難_藥物_收案時': 'dyspnea_medication',
        '症狀_噁心嘔吐_收案時': 'nausea',
        '症狀_便秘_收案時': 'constipation',
        '症狀_吞嚥困難_收案時': 'dysphagia',
        '症狀_腫瘤潰瘍傷口_收案時': 'tumor_ulcer_wound',
        '症狀_腹水_收案時': 'ascites',
        '症狀_淋巴水腫_收案時': 'lymphedema',
        '症狀_一般水腫_收案時': 'general_edema',
        '症狀_軟弱疲倦_收案時': 'fatigue',
        '症狀_失眠_收案時': 'insomnia',
        '症狀_大小便失禁_收案時': 'incontinence',
        '症狀_病人對診斷的認知_收案時': 'patient_awareness_diagnosis',
        '症狀_病人對病情及預後的認知_收案時': 'patient_awareness_prognosis',
        '症狀_家屬對病情及預後的認知_收案時': 'family_awareness_prognosis',
        '心理社會問題_病人': 'psychosocial_issues_patient',
        '心理社會問題_家屬': 'psychosocial_issues_family',
        '靈性宗教_病人': 'spirituality_religion_patient',
        '靈性宗教_家屬': 'spirituality_religion_family',
        '相關轉介': 'referral',
        '結案原因': 'discharge_reason',
        '結案原因1': 'discharge_reason_1',
        '宗教信仰': 'religious_beliefs'
    }, inplace=True)
    
    religious_beliefs_mapping = {
        '一般民間': 'folk',
        '佛教': 'Buddhism',
        '基督/天主教': 'Christ',
        '其他': 'other',
        '道教': 'Taoism'}
    data['religious_beliefs'].replace(religious_beliefs_mapping, inplace=True)
    
    DNR_timing_mapping = {
        '前': 'before',
        '後': 'after'}
    data['DNR_timing'].replace(DNR_timing_mapping, inplace=True)
    
    known_mapping = {
        '知': 'Wellknown',
        '半知': 'Partial',
        '不知': 'Unknown',}
    data['patient_awareness_diagnosis'].replace(known_mapping, inplace=True)
    data['patient_awareness_prognosis'].replace(known_mapping, inplace=True)
    data['family_awareness_prognosis'].replace(known_mapping, inplace=True)
    
    return data

def religion_replace_na_with_folk(data):
    '''
    Features: In religious beliefs,
    folk (general folk) is used to fill in missing values.
    '''
    data['religious_beliefs'].fillna(value='folk', inplace=True)

    return data

def Combine_DNRtiming_and_IsSignDNR(data):
    '''
    Merge 'IsSignDNR', 'DNR_timing' into (No/before/after)
    '''
    if data['IsSignDNR'] == 'No':
        return 'No'
    else:
        if data['DNR_timing'] == 'before':
            return 'before'
        elif data['DNR_timing'] == 'after':
            return 'after'
        else:
            return None

def medicine_morphine(data):
    '''
    Divide pain medications and dyspnea medications into "opioid" and "non-opioid"
    Only after converting to str type can we get_dummies directly.
    '''
    pain_medication_mapping = {
        'demeroal,morphine,phentalyine': 'opioid',
        'Non': 'non-opioid',
        'codenine,tramadol': 'opioid'
    }

    dyspnea_medication_mapping = {
        'Morphine': 'opioid',
        'O2': 'non-opioid',
        'Inhlation,Steroid': 'non-opioid'
    }

    data['pain_medication'] = data['pain_medication'].map(pain_medication_mapping)
    data['dyspnea_medication'] = data['dyspnea_medication'].map(dyspnea_medication_mapping)
    data['pain_medication'].fillna(value='non-opioid', inplace=True)
    data['dyspnea_medication'].fillna(value='non-opioid', inplace=True)

    return data

def StartDay_To_lunar_New_Year(data):
    '''
    Change the start date of hospice services to "LunarNewYear" and "NonChineseNewYear" according to the month
    Check table 2005-2020
    The 15 days before and the 15 days after the Spring Festival are counted as the Spring Festival
    '''
    data.reset_index(drop=True, inplace=True)
    start_date = []
    Is_NewYear = 0
    for i in data['admission_date']:
        year = int(str(i)[0:4])
        month = int(str(i)[5:7])
        day = int(str(i)[8:10])
        if year == 2005: # 0209
            if (month == 1 and day >= 25) or (month == 2 and day <= 24):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'
        elif year == 2006: #0129
            if (month == 1 and day >= 14) or (month == 2 and day <= 13):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'
        elif year == 2007: #0218, 2007年的2月有28天
            if (month == 2 and day >= 3) or (month == 3 and day <= 5):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'     
        elif year == 2008: #0207
            if (month == 1 and day >= 23) or (month == 2 and day <= 22):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'
        elif year == 2009: #0126
            if (month == 1 and day >= 11) or (month == 2 and day <= 10):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'
        elif year == 2010: #0214
            if (month == 1 and day >= 30) or (month == 2) or (month == 3 and day <= 1):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'            
        elif year == 2011: #0203
            if (month == 1 and day >= 19) or (month == 2 and day <= 18):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'
        elif year == 2012: #0123
            if (month == 1 and day >= 8) or (month == 2 and day <= 7):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'
        elif year == 2013: #0210
            if (month == 1 and day >= 26) or (month == 2 and day <= 25):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'
        elif year == 2014: #0131
            if (month == 1 and day >= 16) or (month == 2 and day <= 15):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'
        elif year == 2015: #0219
            if (month == 2 and day >= 4) or (month == 3 and day <= 6):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'
        elif year == 2016: #0208
            if (month == 1 and day >= 24) or (month == 2 and day <= 23):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'
        elif year == 2017: #0128
            if (month == 1 and day >= 13) or (month == 2 and day <= 12):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'
        elif year == 2018: #0216
            if (month == 2 and day >= 1) or (month == 3 and day <= 1):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'
        elif year == 2019: #0205
            if (month == 1 and day >= 21) or (month == 2 and day <= 20):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'
        elif year == 2020: #0125
            if (month == 1 and day >= 10) or (month == 2 and day <= 9):
                Is_NewYear = 'LunarNewYear'
            else:
                Is_NewYear = 'NonChineseNewYear'
        start_date.append(Is_NewYear)
    start_date = pd.DataFrame(start_date)
    data = pd.concat([data, start_date], axis='columns')
    data.rename(columns={0:'new_year'}, inplace=True)

    return data

def DiseaseDiagnosis_Transcoding(data):
    '''
    2023 icd10:https://www.icd10data.com/ICD10CM/Codes/C00-D49
    icd9(no longer in use):https://en.wikipedia.org/wiki/List_of_ICD-9_codes_140%E2%80%93239:_neoplasms
    
    In feature diagnosis:
    1. Replace missing values ​​with "Other"
    2. Delete the decimal point in the value (just judge by numbers)
    
    icd10:
        If it starts with English==C:
            C00-C14 Malignant neoplasms of lip, oral cavity and pharynx (唇、口腔和咽部惡性腫瘤)
            C15-C26 Malignant neoplasms of digestive organs (消化器官惡性腫瘤)
            C30-C39 Malignant neoplasms of respiratory and intrathoracic organs (呼吸系統和胸腔內器官惡性腫瘤)
            C40-C41 Malignant neoplasms of bone and articular cartilage (骨和關節軟骨惡性腫瘤)
            C43-C44 Melanoma and Other malignant neoplasms of skin (黑色素瘤和其他皮膚惡性腫瘤)
            C45-C49 Malignant neoplasms of mesothelial and soft tissue (中皮和軟組織惡性腫瘤)
            C50-C50 Malignant neoplasms of breast (乳腺惡性腫瘤)
            C51-C58 Malignant neoplasms of female genital organs (女性生殖器官惡性腫瘤)
            C60-C63 Malignant neoplasms of male genital organs (男性生殖器官惡性腫瘤)
            C64-C68 Malignant neoplasms of urinary tract (泌尿系統惡性腫瘤)
            C69-C72 Malignant neoplasms of eye, brain and Other parts of central nervous system (眼、腦和中樞神經系統其他部位惡性腫瘤)
            C73-C75 Malignant neoplasms of thyroid and Other endocrine glands (甲狀腺和其他內分泌腺惡性腫瘤)
            C76-C80 Malignant neoplasms of ill-defined, Other secondary and unspecified sites (未明確定義的、其他次發性和未指定部位的惡性腫瘤)
            C7A-C7A Malignant neuroendocrine tumors (神經內分泌腫瘤)
            C7B-C7B Secondary neuroendocrine tumors (二次神經內分泌腫瘤)
            C81-C96 Malignant neoplasms of lymphoid, hematopoietic and related tissue (淋巴、血液和相關組織的惡性腫瘤)
            
        If it starts with English==D:(較多良性腫瘤，因此多數會歸類到其他)
            D00-D09 In situ neoplasms (原位腫瘤)
            D10-D36 Benign neoplasms, except benign neuroendocrine tumors (良性腫瘤)
            D37-D48 Neoplasms of uncertain behavior, polycythemia vera and myelodysplastic syndromes (行為不確定的腫瘤，原發性紅血球增多症和骨髓增生異常綜合徵)
            D3A-D3A Benign neuroendocrine tumors (良性神經內分泌腫瘤)
            D49-D49 Neoplasms of unspecified behavior (行為未指定的腫瘤)
    
    icd9(140~239: neoplasms)
        (140~149) Malignant neoplasm of lip, oral cavity, and pharynx (唇、口腔和咽部惡性腫瘤)
        (150~159) Malignant neoplasm of digestive organs and peritoneum (消化器官和腹膜惡性腫瘤)
        (160~165) Malignant neoplasm of respiratory and intrathoracic organs (呼吸系統和胸腔內器官惡性腫瘤)
        (170_175) Malignant neoplasm of bone, connective tissue, skin, and breast (骨、結締組織、皮膚和乳房惡性腫瘤)
        (176~176) Kaposi's sarcoma (卡波西氏肉瘤)
        (179~189) Malignant neoplasm of genitourinary organs (泌尿生殖器官惡性腫瘤)
        (190~199) Malignant neoplasm of Other and unspecified sites (其他和未指定部位的惡性腫瘤)
        (200~208) Malignant neoplasm of lymphatic and hematopoietic tissue (淋巴和造血組織惡性腫瘤)
        (209~209) Neuroendocrine tumors (神經內分泌腫瘤)
        (210~229) Benign neoplasms (良性腫瘤)
        (230~234) Carcinoma in situ (原位癌)
        (235~238) Neoplasms of uncertain behavior (行為不確定的腫瘤)
        (239~239) Neoplasms of unspecified nature (未指定性質的腫瘤)
    '''
    data['diagnosis'].fillna(value = 'OtherCancer',inplace=True) # Replace missing values ​​with "Other"

    for i in range(len(data['diagnosis'].values)):

        data['diagnosis'].values[i] = str(data['diagnosis'].values[i])
        dele = '.'
        data['diagnosis'].values[i] = ''.join( x for x in data['diagnosis'].values[i] if x not in dele)

        # icd10
        if data['diagnosis'].values[i][0] == 'C':
            if data['diagnosis'].values[i][1:3] == '7A' or data['diagnosis'].values[i][1:3] == '7B':
                # Malignant neuroendocrine tumors
                data['diagnosis'].values[i] = 'OtherCancer'
                continue

            if data['diagnosis'].values[i][1:].isdigit(): #如果c後面都是數字(因為有些有字母)
                if (int(data['diagnosis'].values[i][1:3]) >= 0 and int(data['diagnosis'].values[i][1:3]) <= 14) or int(data['diagnosis'].values[i][1:3]) == 73 :
                    # Malignant neoplasms of lip, oral cavity and pharynx
                    # C73 --> Malignant neoplasm of thyroid gland(甲狀腺的，醫師說放到頭頸)
                    data['diagnosis'].values[i] = 'HeadAndNeck'
                    continue
                elif int(data['diagnosis'].values[i][1:3]) >= 30 and int(data['diagnosis'].values[i][1:3]) <= 32:
                    # C30 鼻腔和中耳惡性腫瘤、C31 副鼻竇惡性腫瘤、C32 喉惡性腫瘤
                    data['diagnosis'].values[i] = 'HeadAndNeck'
                    continue 
                elif int(data['diagnosis'].values[i][1:3]) >= 15 and int(data['diagnosis'].values[i][1:3]) <= 26:
                    # Malignant neoplasms of digestive organs
                    data['diagnosis'].values[i] = 'GI'
                    continue
                elif int(data['diagnosis'].values[i][1:3]) >= 33 and int(data['diagnosis'].values[i][1:3]) <= 39:
                    # C33 氣管惡性腫瘤、C34 支氣管和肺部惡性腫瘤、C37 胸腺惡性腫瘤
                    # C38 心臟、縱隔和胸膜惡性腫瘤、C39 呼吸系統和胸腔內器官其他和不明確部位的惡性腫瘤
                    data['diagnosis'].values[i] = 'Respiratory'
                    continue 
                elif int(data['diagnosis'].values[i][1:3]) >= 40 and int(data['diagnosis'].values[i][1:3]) <= 49 and int(data['diagnosis'].values[i][1:3]) != 42:
                    # C40-C41  Malignant neoplasms of bone and articular cartilag
                    # C43-C44  Melanoma and Other malignant neoplasms of skin
                    # C45-C49  Malignant neoplasms of mesothelial and soft tissue
                    # 沒有C42這個代碼!!
                    data['diagnosis'].values[i] = 'OtherCancer'
                    continue
                elif int(data['diagnosis'].values[i][1:3]) == 50:
                    # Malignant neoplasms of breast
                    data['diagnosis'].values[i] = 'Breast'
                    continue
                elif int(data['diagnosis'].values[i][1:3]) >= 51 and int(data['diagnosis'].values[i][1:3]) <= 58:
                    # Malignant neoplasms of female genital organs
                    data['diagnosis'].values[i] = 'GYN'
                    continue
                elif int(data['diagnosis'].values[i][1:3]) >= 60 and int(data['diagnosis'].values[i][1:3]) <= 68:
                    # C60-C63  Malignant neoplasms of male genital organs(男性生殖器)
                    # C64-C68  Malignant neoplasms of urinary tract(腎和泌尿道)
                    data['diagnosis'].values[i] = 'GU'
                    continue
                elif int(data['diagnosis'].values[i][1:3]) >= 69 and int(data['diagnosis'].values[i][1:3]) <= 72:
                    # Malignant neoplasms of eye, brain and Other parts of central nervous system
                    data['diagnosis'].values[i] = 'HeadAndNeck'
                    continue
                elif int(data['diagnosis'].values[i][1:3]) >= 74 and int(data['diagnosis'].values[i][1:3]) <= 97:
                    # C74以後，且除了C7A-C7A、C7B-C7B的
                    data['diagnosis'].values[i] = 'OtherCancer'
                    continue

        if data['diagnosis'].values[i][0] == 'D':
            if int(data['diagnosis'].values[i][1:3]) == 42 or int(data['diagnosis'].values[i][1:3]) == 43:
                # D42  Neoplasm of uncertain behavior of meninges
                # D43  Neoplasm of uncertain behavior of brain and central nervous system
                data['diagnosis'].values[i] = 'HeadAndNeck'
                continue
            elif int(data['diagnosis'].values[i][1:3]) == 0 or int(data['diagnosis'].values[i][1:3]) == 1:
                # D00  Carcinoma in situ of oral cavity, esophagus and stomach
                # D01  Carcinoma in situ of Other and unspecified digestive organs
                data['diagnosis'].values[i] = 'GI'
                continue
            elif int(data['diagnosis'].values[i][1:3]) == 5:
                # D05  Carcinoma in situ of breast
                data['diagnosis'].values[i] = 'Breast'
                continue
            elif int(data['diagnosis'].values[i][1:3]) == 6 or int(data['diagnosis'].values[i][1:3]) == 39:
                # D06  Carcinoma in situ of cervix uteri
                # D39  Neoplasm of uncertain behavior of female genital organs
                data['diagnosis'].values[i] = 'GYN'
                continue
            elif int(data['diagnosis'].values[i][1:3]) == 40 or int(data['diagnosis'].values[i][1:3]) == 41:
                # D40  Neoplasm of uncertain behavior of male genital organs
                # D41  Neoplasm of uncertain behavior of urinary organs
                data['diagnosis'].values[i] = 'GU'
                continue
            else:
                data['diagnosis'].values[i] = 'OtherCancer'
                continue

        # icd9
        if data['diagnosis'].values[i].isdigit(): #如果都是字串都是數字(因為icd9沒有英文開頭，全是數字)
            if int(data['diagnosis'].values[i][0:2]) == 14 or int(data['diagnosis'].values[i][0:3]) == 193:
                # 140–149:Malignant neoplasm of lip, oral cavity, and pharynx
                # 193 Malignant neoplasm of thyroid gland(甲狀腺的)
                data['diagnosis'].values[i] = 'HeadAndNeck'
                continue
            elif int(data['diagnosis'].values[i][0:2]) == 15 and int(data['diagnosis'].values[i][2]) != 8:
                # Malignant neoplasm of digestive organs and peritoneum(腹膜跟消化)
                # !=158 : Malignant neoplasm of retroperitoneum and peritoneum(把腹膜排除掉)
                data['diagnosis'].values[i] = 'GI'
                continue
            elif int(data['diagnosis'].values[i][0:3]) >= 160 and int(data['diagnosis'].values[i][0:3]) <= 165:
                # Malignant neoplasm of respiratory and intrathoracic(呼吸、胸腔內)
                data['diagnosis'].values[i] = 'Respiratory'
                continue
            elif (int(data['diagnosis'].values[i][0:3]) >= 170 and int(data['diagnosis'].values[i][0:3]) <= 173) or int(data['diagnosis'].values[i][0:3]) == 158:
                # 170 Malignant neoplasm of bone and articular cartilage、171 Malignant neoplasm of connective and Other soft tissue Rhabdomyosarcoma
                # 172 Malignant melanoma of skin、173 Other malignant neoplasm of skin
                # 一些皮膚相關的，158是腹膜的
                data['diagnosis'].values[i] = 'OtherCancer'
                continue
            elif int(data['diagnosis'].values[i][0:3]) == 174 or int(data['diagnosis'].values[i][0:3]) == 175:
                # 分別是男女乳腺腫瘤的
                data['diagnosis'].values[i] = 'Breast'
                continue
            elif int(data['diagnosis'].values[i][0:3]) >= 179 and int(data['diagnosis'].values[i][0:3]) <= 184:
                # 179 Malignant neoplasm of uterus, part unspecified
                # 180 Malignant neoplasm of cervix uteri
                # 181 Malignant neoplasm of placenta
                # 182 Malignant neoplasm of body of uterus
                # 182.0 Corpus uteri, except isthmus
                # Endometrial cancer
                # 183 Malignant neoplasm of ovary and Other uterine adnexa
                # 184 Malignant neoplasm of Other and unspecified female genital organs
                # 就是一些婦科的腫瘤
                data['diagnosis'].values[i] = 'GYN'
                continue
            elif int(data['diagnosis'].values[i][0:3]) >= 185 and int(data['diagnosis'].values[i][0:3]) <= 189:
                # 185 Malignant neoplasm of prostate
                # 186 Malignant neoplasm of testis
                # 187 Malignant neoplasm of penis and Other male genital organs
                # 188 Malignant neoplasm of bladder
                # 189 Malignant neoplasm of kidney and Other and unspecified urinary organs
                data['diagnosis'].values[i] = 'GU'
                continue
            elif int(data['diagnosis'].values[i][0:3]) >= 190 and int(data['diagnosis'].values[i][0:3]) <= 193 and int(data['diagnosis'].values[i][0:3]) != 192:
                # 190~193但排除192(神經的:Malignant neoplasm of Other and unspecified parts of nervous system)
                # 190 Malignant neoplasm of eye、191 Malignant neoplasm of brain、193 Malignant neoplasm of thyroid gland
                data['diagnosis'].values[i] = 'HeadAndNeck'
                continue
            elif int(data['diagnosis'].values[i][0:3]) >= 192 and int(data['diagnosis'].values[i][0:3]) <= 239 and int(data['diagnosis'].values[i][0:3]) != 193:      
                # 192~239 但排除193(甲狀腺歸到HeadAndNeck)
                data['diagnosis'].values[i] = 'OtherCancer'
                continue
            else:
                # 已經全部是腫瘤的可能性都放上去了，剩下的icd9碼就是非癌症
                data['diagnosis'].values[i] = 'NotCancer'
                continue

        else:
            # 已經把icd10的腫瘤可能都放上去了，剩下的就不是癌症了
            data['diagnosis'].values[i] = 'NotCancer'

    return data

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

def MinMaxNormalization(data):

    continuous_data = data[['age','ECOG','pain','dyspnea','nausea',
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

def OneHotEncoding(data, OneHotCols):

    data_ToBeOneHot = data[['sex','DNR','pain_medication','dyspnea_medication',
                        'psychosocial_issues_patient','psychosocial_issues_family',
                        'spirituality_religion_patient','spirituality_religion_family',
                        'referral','diagnosis','religious_beliefs','new_year',
                        'patient_awareness_diagnosis','patient_awareness_prognosis',
                        'family_awareness_prognosis']]
    cat_ori_cols = data_ToBeOneHot.columns
    data.drop(data_ToBeOneHot.columns, axis=1, inplace=True)
    
    OneHotDone_data = pd.get_dummies(data_ToBeOneHot) # 做OneHotEncoding
    
    # 判斷如果OneHotCols有OneHotDone_data.columns沒有的column
    missing_cols = [col for col in OneHotCols if col not in OneHotDone_data.columns]
    # OneHotDone_data就加上這個column，值為0
    for col in missing_cols:
        OneHotDone_data[col] = 0
    
    # 整理一下column順序
    OneHotDone_data = OneHotDone_data[OneHotCols]
    
    data = pd.concat([data, OneHotDone_data], axis=1) # 把OneHotDone_data跟其他data合併回來
    
    return data, cat_ori_cols

def all_preprocess(data, OneHotCols):
    
    # Rename_ChineseToEnglish
    data = Rename_ChineseToEnglish(data)

    # “religious_beliefs”中的缺值歸類至民間信仰裡面
    data = religion_replace_na_with_folk(data)

    # 把'IsSignDNR', 'DNR_timing'合併成(No/before/after)
    data['DNR'] = data.apply(Combine_DNRtiming_and_IsSignDNR, axis=1)
    data = data.drop(['IsSignDNR', 'DNR_timing'], axis = 'columns')

    # 'pain_medication'、'dyspnea_medication'分為opioid跟non-opioid
    data = medicine_morphine(data)

    # 收案日期改為春節或非春節
    data = StartDay_To_lunar_New_Year(data)

    # 把研究中不會用到的特徵drop掉
    data.drop(['discharge_date', 'discharge_reason','IsCancer','patient_ID_number',
                'birthday','discharge_reason_1','primary_caregiver','admission_date'], axis=1, inplace=True)

    # 疾病診斷處理(轉碼)
    data = DiseaseDiagnosis_Transcoding(data)

    # 把0-1、2-3那種改為連續變項(且字串轉數字)
    data = StrToInt(data)

    # 把結果三種group轉為1、2、3
    data['Group'] = data['Group'].map({'HHC': 0, 'HPC': 1, 'HSC': 2})

    # MinMaxNormalization
    data = MinMaxNormalization(data)

    # one hot encoding
    data, _ = OneHotEncoding(data, OneHotCols)
    
    return data

def visualize_tree_with_path(tree_model, X, data_point, filepath):

    feature_names = [str(feature) for feature in X.columns]
    class_names = [str(cls) for cls in tree_model.classes_]
    
    # Generate decision tree visualization
    dot_data = export_graphviz(tree_model, out_file=None, 
                               feature_names=feature_names,
                               class_names=class_names,  
                               filled=True, rounded=True,  
                               special_characters=True)

    # Find the decision path for the given data point
    node_indicator = tree_model.decision_path(data_point)
    leaf_id = tree_model.apply(data_point)
    
    # Get the nodes involved in the decision path
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    # Highlight the decision path in the dot_data
    lines = dot_data.split('\n')
    for i, line in enumerate(lines):
        if 'fillcolor' in line:
            node_id_end = line.find(' ')  # Node ID is before the first space
            node_id = int(line[:node_id_end])
            # print(node_id)
            if node_id not in node_index:
                lines[i] = line.split(', fillcolor="')[0] + ', fillcolor="#FFFFFF"]'  # Remove existing fillcolor property and add empty fillcolor
    
    # Create new dot data with modified fillcolor
    modified_dot_data = '\n'.join(lines)
    
    # Visualize the decision tree with path 直的
    graph = graphviz.Source(modified_dot_data)
    graph.render(filepath)
    
    modified_dot_data = modified_dot_data.replace('}', 'rankdir=LR;}', 1)
    graph = graphviz.Source(modified_dot_data)
    graph.render(f'{filepath}_LR')