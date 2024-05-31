### Project Overview ###
This project focuses on developing machine learning models to predict individualized hospice care needs.
--------------------------------------------------------------------------------
## 0_raw_data
  Due to confidentiality reasons, the raw data cannot be disclosed. Files containing data content within the entire project will also not be provided.

## 1_Data_preprocess(first)
  1.1 Data Filtering:
    Excluded non-death related case closures.
    Ensured each patient received only one type of hospice care.
  1.2 Handling Some Missing Values:
    Classified missing religious data as general folk beliefs.
  1.3 Feature Engineering:
    Combined DNR signing time and existence into one feature.
    Reclassified pain medications into opioids and non-opioids.
  1.4 Data Coverage:
    Included data from 2005 to 2020, marking Lunar New Year periods.
  1.5 Final Dataset:
    Total patients: 3,468;  Total features: 29
    HHC: 713 cases;  HIC: 642 cases;  HSC: 2,113 cases

## 2_Data_preprocess
  2.1 Converted certain variables into continuous ones and transformed 'Group' into numerical values.
  2.2 Separated data with missing values for future imputation.
  2.3 Retained a clean dataset without missing values as the main dataset.
  2.4 Applied one-hot encoding to categorical variables and normalized all features.
  2.5 Split the main dataset for training and testing, ensuring balanced representation of all groups.
  2.6 Data Shuffling

## 3_five_fold
  '''
  The data is split into 5 folds(using K-means) and trained using 6 models
  (logistic regression, random forest, SVM, XGBoost, CatBoost, MLP). 
  Each fold undergoes 5-fold cross-validation, with different experiments conducted in each folder.
  '''
  3_0_Initial_result

  3_1_Data_imbalance_processing_experiment
    Compares [Balanced Class Weight,
              ENN,
              ENN + SMOTE,
              Copy(K-means Sampling),
              Two-Stage Model(First predicting HSC suitability, then classifying non-HSC patients as HIC or HHC.)]

  3_2_Feature_engineering
    Compares [PCA, RFE, Statistical_Analysis, Variance threshold]

  3_3_Impute_missing_values_experiment
    Compares [MICE, Miss forest, KNN impute]

  3_4_Grid_search
    Grid search with 5-fold.

  3_5_final_model
    Combine 3 models with the best performance.

## 4_feature_analysis
  4.1 Teacher Student models
    Teacher-Student models train a simpler model (student) to mimic a more complex one (teacher).
    Here, the teacher is an accurate SVM, and the student is an interpretable Decision Tree.
    The goal is to create an explanatory model suitable for clinical use.
  4.2 Plot importance
    Plotting feature importance with CatBoost and XGBoost.
  4.3 Give an example to classify and visualize the tree student model

## CompareDropFewDays
  Compare model performance after deleting data within n days of receiving service.