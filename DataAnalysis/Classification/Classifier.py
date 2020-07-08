import pandas as pd
import sys
import numpy as np
from numpy.random import seed
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import export_text
import xgboost as xgb
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
from skopt import BayesSearchCV
from random import randrange
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus

warnings.filterwarnings("ignore")

#################################################################################################
# Let's see if we can do label propagation with the subclusters we have (semi-supervised ML)
# Also try the quasi-Newton SVM (this takes time!)
# LABEL SPREADING --> TRANSDUCTION
# We are going to consider the labeled data first, to test label prop accuracy

# --------------------------------------------------------------------------------------------------------

# Read data with responsive units and validated units from unsupervised clustering
df = pd.read_csv('Data_v1.csv')

# Create key Sample-Unit
df['key'] = df['Sample'] + '-' + df['Unit'].astype(str)

# Set key as index
df = df.set_index('key')


# Transform label names to numbers
def labels(row):
    if row['SubCluster'] == 'Pk' or row['RespSS'] == 1:
        return 1
    elif (row['SubCluster'] == 'CS' and row['Corr_CS'] != 1) or row['RespCS'] == 1:
        return 2
    elif row['SubCluster'] == 'Gr':
        return 3
    elif row['SubCluster'] == 'Go':
        return 4
    elif row['SubCluster'] == 'MF':
        return 5
    # elif row['MF_baseline'] == 1 or row['RespMF'] == 1:
    #   return 5
    elif row['SubCluster'] == 'MLI':
        return 6
    else:
        return -1  # Unidentified observations


df['Label'] = df.apply(labels, axis=1)
print(df.groupby('Label').count())

# def only_responsive_labels(row):
#     if row['RespPk'] == 1 or row['RespSS'] == 1:
#         return 1
#     elif row['RespCS'] == 1:
#         return 2
#     elif row['RespGr'] == 1:
#         return 3
#     elif row['RespGo'] == 1:
#         return 4
#     elif row['RespMF'] == 1:
#         return 5
#     elif row['RespMLI'] == 1:
#         return 6
#     else:
#         return -1  # Unidentified observations

# df['Label'] = df.apply(only_responsive_labels, axis=1)
# print(df.groupby('Label').count())

# Select only labeled data to train a classifier
labeled = df[df.Label != -1]

# -------------------------------------------------------------------------------------------------------
# Baseline model

seed(1)


def baseline(row):
    if row['MFRBlockHz'] >= 60:
        return 1  # Pk
    elif 0.5 <= row['MFRBlockHz'] <= 2:
        return 2  # CS
    elif 2 <= row['MFRBlockHz'] <= 8:
        return 3  # Gr
    elif row['MF_baseline'] == 1:  # To identify extra sags commonly found in MF
        return 5
    elif 10 <= row['MFRBlockHz'] <= 50:
        n = np.random.rand()
        if n >= 0.5:
            return 4
        else:
            return 6
    else:
        return randrange(1, 7)  # Unidentified observations, which include Golgi and MLI cells


labeled['Baseline'] = labeled.apply(baseline, axis=1)
print(labeled.groupby('Baseline').count())

# -------------------------------------------------------------------------------------------------------

# Class weights
Pk_p = round(labeled[labeled['Label'] == 1].shape[0] / labeled.shape[0], 2)
CS_p = round(labeled[labeled['Label'] == 2].shape[0] / labeled.shape[0], 2)
Gr_p = round(labeled[labeled['Label'] == 3].shape[0] / labeled.shape[0], 2)
Go_p = round(labeled[labeled['Label'] == 4].shape[0] / labeled.shape[0], 2)
MF_p = round(labeled[labeled['Label'] == 5].shape[0] / labeled.shape[0], 2)
MLI_p = round(labeled[labeled['Label'] == 6].shape[0] / labeled.shape[0], 2)

weights = {1: Pk_p, 2: CS_p, 3: Gr_p, 4: Go_p, 5: MF_p, 6: MLI_p}

# C values to test for regularization
C_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 10, 100, 150]

# K-fold splits
k = 5
inner_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
outer_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

# Extract features and target
X_all = labeled[['MFRBlockHz',
                 'MeanAmpBlock',
                 'tf_MIFRBlockHz',
                 'tf_MedIsi',
                 'tf_ModeIsi',
                 'tf_Perc5Isi',
                 'tf_Entropy',
                 'tf_CV2Mean',
                 'tf_CV2Median',
                 'tf_CV',
                 'tf_Ir',
                 'tf_Lv',
                 'tf_LvR',
                 'tf_Si',
                 'tf_skw',
                 'RespPk',
                 'RespGo',
                 'RespGr',
                 'RespMF',
                 'RespMLI',
                 'wf_RiseTime',
                 'wf_PosDecayTime',
                 'wf_FallTime',
                 'wf_NegDecayTime',
                 'wf_MaxAmpNorm',
                 'wf_Duration',
                 'wf_PosHwDuration',
                 'wf_NegHwDuration',
                 'wf_Onset',
                 'wf_End',
                 'wf_Crossing',
                 'wf_Pk10',
                 'wf_Pk90',
                 'wf_Pk50',
                 'wf_PkTrRatio',
                 'wf_DepolarizationSlope',
                 'wf_RepolarizationSlope',
                 'wf_RecoverySlope',
                 'wf_EndSlopeTau'
                 ]]

X = labeled[['MFRBlockHz',
             'MeanAmpBlock',
             'tf_MIFRBlockHz',
             'tf_MedIsi',
             'tf_ModeIsi',
             'tf_Perc5Isi',
             'tf_Entropy',
             'tf_CV2Mean',
             'tf_CV2Median',
             'tf_CV',
             'tf_Ir',
             'tf_Lv',
             'tf_LvR',
             'tf_Si',
             'tf_skw',
             'RespPk',
             'RespGo',
             'RespGr',
             'RespMF',
             'RespMLI',
             'wf_RiseTime',
             'wf_PosDecayTime',
             'wf_FallTime',
             'wf_NegDecayTime',
             'wf_MaxAmpNorm',
             'wf_Duration',
             'wf_PosHwDuration',
             'wf_NegHwDuration',
             'wf_Onset',
             'wf_End',
             'wf_Crossing',
             'wf_Pk10',
             'wf_Pk90',
             'wf_Pk50',
             'wf_PkTrRatio',
             'wf_DepolarizationSlope',
             'wf_RepolarizationSlope',
             'wf_RecoverySlope',
             'wf_EndSlopeTau',
             'Baseline'
             ]]

y = labeled[['Label']]
y1 = labeled[['Label', 'Baseline']]

# Train data
X_train, X_rest, y_train, y_rest = train_test_split(X, y1,
                                                    test_size=0.4,
                                                    random_state=0,
                                                    stratify=y)

# Test and validation data
X_test, X_validation, y_test, y_validation = train_test_split(X_rest, y_rest[['Label']],
                                                              test_size=0.50,
                                                              random_state=0,
                                                              stratify=y_rest[['Label']])

# Select model and hyper-param based on test data performance
# Report final accuracy on validation data

# -------------------------------------------------------------------------------------------------------
# How does baseline perform? We can only judge known prior classes
# Baseline accuracy for Go and MLI is 0

target_names = ['Pk', 'CS', 'Gr', 'Go', 'MF', 'MLI']
y1 = y1[y1['Baseline'] != -1]
y_test1 = y1[['Label']]
y_baseline = y1[['Baseline']]

print('Baseline model - Test Set (Model Selection) \n',
      classification_report(y_test1, y_baseline, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

# -------------------------------------------------------------------------------------------------------
# For the rest of the algorithms

y_train = y_train[['Label']]

cols_to_use = ['MFRBlockHz',
               'MeanAmpBlock',
               'tf_MIFRBlockHz',
               'tf_MedIsi',
               'tf_ModeIsi',
               'tf_Perc5Isi',
               'tf_Entropy',
               'tf_CV2Mean',
               'tf_CV2Median',
               'tf_CV',
               'tf_Ir',
               'tf_Lv',
               'tf_LvR',
               'tf_Si',
               'tf_skw'
               ]

X = X[cols_to_use]
X_train = X_train[cols_to_use]
X_test = X_test[cols_to_use]
X_validation = X_validation[cols_to_use]

# SMOTE oversampling on training data (not on test data)
# print(y_train.reset_index().groupby('Label').count())

# d = {1: 258, 2: 73, 3: 31, 4: 20, 5: 36, 6: 20}
#
# ros = RandomOverSampler(random_state=42, sampling_strategy=d)
# X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

# print(X_train_ros[y_train_ros.Label == 4].reset_index().groupby('MFRBlockHz').count())

# d2 = {1: 558, 2: 373, 3: 331, 4: 320, 5: 336, 6: 320}

# sm = SMOTE(random_state=42, sampling_strategy=d2, k_neighbors=2)
# X_train_sm, y_train_sm = sm.fit_resample(X_train_ros, y_train_ros)

# print(y_train_sm.reset_index().groupby('Label').count())
# print(X_train_sm[y_train_sm.Label == 6])
# print(X_train_sm[y_train_sm.Label == 4].reset_index().groupby('MFRBlockHz').count())

# df_Pk = labeled[labeled.Label == 1]
# df_CS = labeled[labeled.Label == 2]
# df_Gr = labeled[labeled.Label == 3]
# df_Go = labeled[labeled.Label == 4]
# df_MF = labeled[labeled.Label == 5]
# df_MLI = labeled[labeled.Label == 6]

# print(y_train_sm.reset_index().groupby('Label').count())
# print(y_test.reset_index().groupby('Label').count())
# print(y_validation.reset_index().groupby('Label').count())

true_class = []
pred_class = []


def classification_report_with_f_score(y_true, y_pred):
    true_class.extend(y_true)
    pred_class.extend(y_pred)
    # precision, recall, f_score, support = score(y_true, y_pred)

    print(f'CV Classification report \n',
          classification_report(y_true, y_pred, zero_division=1))

    return f1_score(y_true, y_pred, average='macro')


# Excluding MF and MLI > LogReg again

df_Pk = labeled[labeled.Label == 1]
df_CS = labeled[labeled.Label == 2]
df_Gr = labeled[labeled.Label == 3]
df_MF = labeled[labeled.Label == 5]

target_names = ['Pk', 'CS', 'Gr', 'MF']

new_X = pd.concat([df_Pk, df_CS, df_Gr, df_MF])

new_y = new_X[['Label']]

new_X = new_X[cols_to_use]

# Train data >> 60%
X_train, X_rest, y_train, y_rest = train_test_split(new_X, new_y,
                                                    test_size=0.4,
                                                    random_state=0,
                                                    stratify=new_y)

# Test and validation data >> 20% y 20%
X_test, X_validation, y_test, y_validation = train_test_split(X_rest, y_rest,
                                                              test_size=0.50,
                                                              random_state=0,
                                                              stratify=y_rest)

# print('test')
# print(y_train.reset_index().groupby('Label').count())

# d2 = {1: 400, 2: 200, 3: 100, 5: 100}
d2 = {1: 310, 2: 74, 3: 37, 5: 43}

sm = SMOTE(random_state=42, sampling_strategy=d2, k_neighbors=5)
# sm = SMOTE(random_state=42, sampling_strategy='not majority', k_neighbors=5)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# # print(y_train_sm.reset_index().groupby('Label').count())
# # print(X_train_sm[y_train_sm.Label == 4])

# GLM
# -------------------------------------------------------------------------------------------------------
# Logistic Regression


parameters = {'solver': ['saga', 'liblinear'],
              'penalty': ['l1', 'l2'],
              'multi_class': ['multinomial', 'ovr'],
              'tol': [1e-5, 1e-3],
              'C': [1e-6, 1e+6],
              'class_weight': [weights, 'balanced']
              }

lr = LogisticRegression(random_state=1234)

# my_cv = LeaveOneOut()

# lr_opt = BayesSearchCV(
#     lr,
#     parameters,
#     n_iter=10,
#     random_state=1234,
#     verbose=False,
#     scoring='f1_macro',
#     cv=inner_cv
# )

# lr_opt = GridSearchCV(
#     lr,
#     parameters,
#     scoring='f1_macro',
#     cv=inner_cv
# )
#
# lr_opt.fit(X, y)

# lr_test = lr_opt.best_estimator_.predict(X_test)
#
# print('LogisticRegression Classifier - Test Set (Oversampling) \n',
#       classification_report(y_test, lr_test, target_names=target_names, zero_division=1),
#       '------------------------------------------------------------------------------------------------- \n')
# lr_validation = lr_opt.best_estimator_.predict(X_validation)
#
# print('LogisticRegression Classifier - Validation Set (Oversampling) \n',
#       classification_report(y_validation, lr_validation, target_names=target_names, zero_division=1),
#       '------------------------------------------------------------------------------------------------- \n')

# Method 2 >> Use cross-val-predict on the entire original data
# from sklearn.model_selection import cross_val_predict
#
# lr_predictions_2 = cross_val_predict(lr_opt.best_estimator_, X, y, cv=5)
# print('LogisticRegression Classifier - (Cross-val-predict) \n',
#       classification_report(y, lr_predictions_2, target_names=target_names, zero_division=1),
#       '------------------------------------------------------------------------------------------------- \n')

# Classes
# print('Actual classes \n', np.unique(y_test0))
# print('Predicted classes \n', np.unique(lr_predictions))

# AUROC
from sklearn.metrics import plot_roc_curve

# probabilities = lr_opt.best_estimator_.predict_proba(X_test)
# print('AUROC \n', roc_auc_score(y_test, probabilities, multi_class="ovr"), '\n')

# ax = plt.gca()
# plot_roc_curve(lr_opt.best_estimator_, X_train[y_train.Label == 6], y_train[y_train.Label == 6], ax=ax)
# plot_roc_curve(lr_opt.best_estimator_, X_test, y_test, ax=ax)
# plot_roc_curve(lr_opt.best_estimator_, X_test, y_test, ax=ax)
# plt.show()

# Nested CV with parameter optimization
# nested_score = np.mean(cross_val_score(lr_opt.best_estimator_, X=X_test, y=y_test.values.ravel(), cv=outer_cv,
#                        scoring=make_scorer(classification_report_with_f_score)))
#
# print('Cross-Validated F-Score:', round(float(nested_score), 2))

# print('Cross-Validated LogisticRegression \n',
#       classification_report(true_class, pred_class, target_names=target_names, zero_division=1),
#       '------------------------------------------------------------------------------------------------- \n')

# true_class.clear()
# pred_class.clear()

# --------------------------------------------------------------------------------------------------------------------

lr_opt = GridSearchCV(
    lr,
    parameters,
    scoring='f1_macro',
    cv=inner_cv
)

lr_opt.fit(X_train_sm, y_train_sm)

lr_test = lr_opt.best_estimator_.predict(X_test)

print('LogisticRegression Classifier - Test Set (Oversampling) \n',
      classification_report(y_test, lr_test, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')
lr_validation = lr_opt.best_estimator_.predict(X_validation)

print('LogisticRegression Classifier - Validation Set (Oversampling) \n',
      classification_report(y_validation, lr_validation, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

probabilities = lr_opt.best_estimator_.predict_proba(X_test)
y_test['LogReg_prob_Pk'] = probabilities[:, 0]
y_test['LogReg_prob_CS'] = probabilities[:, 1]
y_test['LogReg_probs_Gr'] = probabilities[:, 2]
y_test['LogReg_probs_MF'] = probabilities[:, 3]
np.set_printoptions(suppress=True)

df_out_lr = pd.merge(X_test, y_test, how='left', left_index=True, right_index=True)


"""
# print('AUROC \n', roc_auc_score(y_test, probabilities, multi_class="ovr"), '\n')


# -------------------------------------------------------------------------------------------------------
# Probit

# -------------------------------------------------------------------------------------------------------
# Passive-Aggressive Classifier

pac = PassiveAggressiveClassifier()

parameters = {'max_iter': [1000],
              'random_state': [0],
              'tol': [1e-3],
              # 'early_stopping': [True, False],
              # 'validation_fraction': [0.1, 0.2, 0.3],
              # 'shuffle': [True, False],
              'loss': ['hinge', 'squared_hinge'],
              'C': C_values,
              'class_weight': [weights, 'balanced']
              }

# Non_nested parameter search and scoring
pac_opt = GridSearchCV(estimator=pac, param_grid=parameters, cv=inner_cv)

pac_opt.fit(X_train_sm, y_train_sm)

pac_test = pac_opt.best_estimator_.predict(X_test)

print('Passive-Aggressive Classifier - Test Set (Oversampling) \n',
      classification_report(y_test, pac_test, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')
pac_validation = pac_opt.best_estimator_.predict(X_validation)

print('Passive-Aggressive Classifier - Validation Set (Oversampling) \n',
      classification_report(y_validation, pac_validation, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

# # Method 2 >> Use cross-val-predict on the entire original data
# pac_predictions = cross_val_predict(pac_opt.best_estimator_, X, y, cv=5)
# print('Passive-Aggressive Classifier - (Cross-val-predict) \n',
#       classification_report(y, pac_predictions, target_names=target_names, zero_division=1),
#       '------------------------------------------------------------------------------------------------- \n')

# -------------------------------------------------------------------------------------------------------
# SVM

# scale = StandardScaler()
# X_train = scale.fit_transform(X_train)

svm = SVC()

parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              # 'max_iter': [1000],
              'degree': [2, 3],
              'gamma': ['scale', 'auto'],
              'break_ties': [False, True],
              'random_state': [0],
              'tol': [1e-3],
              'C': C_values,
              'class_weight': [weights, 'balanced']
              }

svm_opt = GridSearchCV(estimator=svm, param_grid=parameters, cv=inner_cv, refit=True)

svm_opt.fit(X_train_sm, y_train_sm)

svm_test = svm_opt.best_estimator_.predict(X_test)

print('SVM Classifier - Test Set (Oversampling) \n',
      classification_report(y_test, svm_test, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')
svm_validation = svm_opt.best_estimator_.predict(X_validation)

print('SVM Classifier - Validation Set (Oversampling) \n',
      classification_report(y_validation, svm_validation, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

# # Method 2 >> Use cross-val-predict on the entire original data
# svm_predictions = cross_val_predict(svm_opt.best_estimator_, X, y, cv=5)
# print('SVM Classifier - (Cross-val-predict) \n',
#       classification_report(y, svm_predictions, target_names=target_names, zero_division=1),
#       '------------------------------------------------------------------------------------------------- \n')
#
print('SVM best parameters \n', svm_opt.best_params_)

# Tree-based models
# -------------------------------------------------------------------------------------------------------
# Decision Tree

dt = DecisionTreeClassifier()

parameters = {'max_depth': [4, 5, 6, 7],
              'criterion': ['entropy', 'gini'],
              'min_samples_split': [5, 6, 7, 8, 9, 10],
              'min_samples_leaf': [5, 6, 7, 8, 9, 10],
              'class_weight': [weights, 'balanced']
              }

dt_opt = GridSearchCV(estimator=dt, param_grid=parameters, cv=inner_cv, refit=True)

dt_opt.fit(X_train_sm, y_train_sm)

dt_test = dt_opt.best_estimator_.predict(X_test)

print('DecisionTree Classifier - Test Set (Oversampling) \n',
      classification_report(y_test, dt_test, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')
dt_validation = dt_opt.best_estimator_.predict(X_validation)

print('DecisionTree Classifier - Validation Set (Oversampling) \n',
      classification_report(y_validation, dt_validation, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

# # Method 2 >> Use cross-val-predict on the entire original data
# dt_predictions = cross_val_predict(dt_opt.best_estimator_, X, y, cv=5)
# print('Decision Tree - (Cross-val-predict) \n',
#       classification_report(y, dt_predictions, target_names=target_names, zero_division=1),
#       '------------------------------------------------------------------------------------------------- \n')
#
# r = export_text(dt_opt.best_estimator_, feature_names=cols_to_use)
# print(r)
#
# import collections
#
# dot_data = StringIO()
# export_graphviz(dt_opt.best_estimator_, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True, feature_names=cols_to_use, class_names=target_names)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#
# colors =  ('deeppink', 'dodgerblue', 'darkorange', 'gold', 'white')
#
# nodes = graph.get_node_list()
#
# for node in nodes:
#     if node.get_name() not in ('node', 'edge'):
#         values = dt_opt.best_estimator_.tree_.value[int(node.get_name())][0]
#         #color only nodes where only one class is present
#         if max(values) >= 0.9*sum(values):
#             node.set_fillcolor(colors[np.argmax(values)])
#         #mixed nodes get the default color
#         else:
#             node.set_fillcolor(colors[-1])
#
# graph.write_png('NeuroPixelsTree.png')
# Image(graph.create_png())


# -------------------------------------------------------------------------------------------------------
# Random Forest + Decision Tree

rf = RandomForestClassifier()

parameters = {'max_depth': [4, 5, 6, 7],
              'criterion': ['entropy', 'gini'],
              'min_samples_split': [5, 6, 7, 8, 9, 10],
              'min_samples_leaf': [5, 6, 7, 8, 9, 10],
              'class_weight': [weights, 'balanced']
              }

rf_opt = GridSearchCV(estimator=rf, param_grid=parameters, cv=inner_cv, refit=True)

rf_opt.fit(X_train_sm, y_train_sm)

# rf_predictions = cross_val_predict(rf, X, y, cv=5)
# print('Random Forest - (Cross-val-predict) \n',
#       classification_report(y, rf_predictions, target_names=target_names, zero_division=1),
#       '------------------------------------------------------------------------------------------------- \n')

rf_test = rf_opt.best_estimator_.predict(X_test)

print('Random Forest Classifier - Test Set (Oversampling) \n',
      classification_report(y_test, rf_test, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')
rf_validation = rf_opt.best_estimator_.predict(X_validation)

print('Random Forest Classifier - Validation Set (Oversampling) \n',
      classification_report(y_validation, rf_validation, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')


"""
# -------------------------------------------------------------------------------------------------------
# XGBOOST


def xgboost(row):
    if row['Label'] == 1:
        return 1  # Pk
    elif row['Label'] == 2:
        return 2  # CS
    elif row['Label'] == 3:
        return 3  # Gr
    elif row['Label'] == 5:
        return 4


y_train_sm['Label_2'] = y_train_sm.apply(xgboost, axis=1)
y_train_sm_xg = y_train_sm[['Label_2']]
y_train_sm_xg.rename(columns={"Label_2": "Label"})

y_test['Label_2'] = y_test.apply(xgboost, axis=1)
y_test_xg = y_test[['Label_2']]
y_test_xg.rename(columns={"Label_2": "Label"})

y_validation['Label_2'] = y_validation.apply(xgboost, axis=1)
y_validation_xg = y_validation[['Label_2']]
y_validation_xg.rename(columns={"Label_2": "Label"})

# print(y_train_sm_xg.reset_index().groupby('Label_2').count())

D_train = xgb.DMatrix(X_train_sm, label=y_train_sm_xg)
D_test = xgb.DMatrix(X_test, label=y_test_xg)
D_validation = xgb.DMatrix(X_validation, label=y_validation_xg)

params = {'eta': 0.3, 'max_depth': 6, 'min_child_weight': 1, 'subsample': 1, 'colsample_bytree': 1,
          'objective': 'multi:softprob', 'num_class': 5, 'eval_metric': "mlogloss"}

num_boost_round = 999

early_stopping_rounds = 10

# XGBoost tuning
# Grid search
# gridsearch_params = [
#     (max_depth, min_child_weight)
#     for max_depth in range(9, 12)
#     for min_child_weight in range(5, 8)
# ]
#
# min_mlogloss = float("Inf")
# best_params = None
#
# for max_depth, min_child_weight in gridsearch_params:
#
#     print("CV with max_depth={}, min_child_weight={}".format(
#         max_depth,
#         min_child_weight))
#
#     # Update our parameters
#     params['max_depth'] = max_depth
#     params['min_child_weight'] = min_child_weight
#
#     # Run CV
#     cv_results = xgb.cv(
#         params,
#         D_train,
#         num_boost_round=num_boost_round,
#         seed=42,
#         nfold=5,
#         metrics='mlogloss',
#         early_stopping_rounds=early_stopping_rounds
#     )
#
#     # Update best multi-class log loss
#     mean_mlogloss = cv_results['test-mlogloss-mean'].min()
#     boost_rounds = cv_results['test-mlogloss-mean'].argmin()
#     print("\tMulti-class log loss {} for {} rounds".format(mean_mlogloss, boost_rounds))
#
#     if mean_mlogloss < min_mlogloss:
#         min_mlogloss = mean_mlogloss
#         best_params = (max_depth, min_child_weight)
#
# print("Best params: {}, {}, Multi-class log loss: {}".format(best_params[0], best_params[1], min_mlogloss))

# Tuning results
# max_depth = 9
# min_child_weight = 5

params['max_depth'] = 9
params['min_child_weight'] = 5

# gridsearch_params = [
#     (subsample, colsample)
#     for subsample in [i / 10. for i in range(7, 11)]
#     for colsample in [i / 10. for i in range(7, 11)]
# ]
#
# min_mlogloss = float("Inf")
# best_params = None
#
# for subsample, colsample in reversed(gridsearch_params):
#
#     print("CV with subsample={}, colsample={}".format(
#           subsample,
#           colsample))
#
#     # We update our parameters
#     params['subsample'] = subsample
#     params['colsample_bytree'] = colsample
#
#     # Run CV
#     cv_results = xgb.cv(
#         params,
#         D_train,
#         num_boost_round=num_boost_round,
#         seed=42,
#         nfold=5,
#         metrics='mlogloss',
#         early_stopping_rounds=early_stopping_rounds
#     )
#
#     # Update best multi-class log loss
#     mean_mlogloss = cv_results['test-mlogloss-mean'].min()
#     boost_rounds = cv_results['test-mlogloss-mean'].argmin()
#     print("\tMulti-class log loss {} for {} rounds".format(mean_mlogloss, boost_rounds))
#
#     if mean_mlogloss < min_mlogloss:
#         min_mlogloss = mean_mlogloss
#         best_params = (subsample, colsample)
#
# print("Best params: {}, {}, Multi-class log loss: {}".format(best_params[0], best_params[1], min_mlogloss))

# Tuning results
# subsample = 1.0
# colsample = 0.8

params['subsample'] = 1.0
params['colsample_bytree'] = 0.8

# min_mlogloss = float("Inf")
# best_params = None
#
# for eta in [.3, .2, .1, .05, .01, .005]:
#
#     print("CV with eta={}".format(eta))
#
#     # We update our parameters
#     params['eta'] = eta
#
#     # Run CV
#     cv_results = xgb.cv(
#         params,
#         D_train,
#         num_boost_round=num_boost_round,
#         seed=42,
#         nfold=5,
#         metrics='mlogloss',
#         early_stopping_rounds=early_stopping_rounds
#     )
#
#     # Update best multi-class log loss
#     mean_mlogloss = cv_results['test-mlogloss-mean'].min()
#     boost_rounds = cv_results['test-mlogloss-mean'].argmin()
#     print("\tMulti-class log loss {} for {} rounds".format(mean_mlogloss, boost_rounds))
#
#     if mean_mlogloss < min_mlogloss:
#         min_mlogloss = mean_mlogloss
#         best_params = eta
#
# print("Best params: {}, Multi-class log loss: {}".format(best_params, min_mlogloss))

# Tuning results
# eta = 0.3

params['eta'] = 0.1

# print('Best params from hyper-param tuning: \n', params)

xgboost = xgb.train(params, D_train, num_boost_round=num_boost_round,
                    evals=[(D_test, "Test")], early_stopping_rounds=early_stopping_rounds)

xgboost.save_model("best_xgboost.model")

# print("Best LogLoss: {:.2f} in {} rounds".format(xgboost.best_score, xgboost.best_iteration+1))

xgboost_preds_test = xgboost.predict(D_test)
xgboost_preds_test = np.asarray([np.argmax(line) for line in xgboost_preds_test])

print('XGBOOST - Test Set (Oversampling) \n',
      classification_report(y_test_xg, xgboost_preds_test, target_names=target_names, zero_division=1),
      '\n -------------------------------------------------------------------------------------------------')

xgboost_preds_validation = xgboost.predict(D_validation)
xgboost_preds_validation = np.asarray([np.argmax(line) for line in xgboost_preds_validation])

print('XGBOOST - Validation Set (Oversampling) \n',
      classification_report(y_validation_xg, xgboost_preds_validation, target_names=target_names, zero_division=1),
      '\n -------------------------------------------------------------------------------------------------')

sys.exit()
# -------------------------------------------------------------------------------------------------------
# KNN

knn = KNeighborsClassifier()

parameters = {'n_neighbors': [2, 3, 4, 5, 6, 7],
              'weights': ['uniform', 'distance'],
              'algorithm': ['kd_tree', 'brute', 'auto'],
              'p': [1, 2]
              }

knn_opt = GridSearchCV(estimator=knn, param_grid=parameters, cv=inner_cv, refit=True)

knn_opt.fit(X_train_sm, y_train_sm)

knn_predictions_test = knn_opt.best_estimator_.predict(X_test)

print('KNN - Test Set (Oversampling) \n',
      classification_report(y_test, knn_predictions_test, target_names=target_names, zero_division=1),
      '\n -------------------------------------------------------------------------------------------------')

knn_predictions_validation = knn_opt.best_estimator_.predict(X_validation)

print('KNN - Validation Set (Oversampling) \n',
      classification_report(y_test, knn_predictions_validation, target_names=target_names, zero_division=1),
      '\n -------------------------------------------------------------------------------------------------')

# -------------------------------------------------------------------------------------------------------
# Gaussian Naive-Bayes

gnb = GaussianNB()

parameters = {'var_smoothing': [1e-9, 1e-5, 1, 1.1, 1.2, 1.5]}

gnb_opt = GridSearchCV(estimator=gnb, param_grid=parameters, cv=inner_cv, refit=True)

gnb_opt.fit(X_train_sm, y_train_sm)

gnb_predictions_test = gnb_opt.best_estimator_.predict(X_test)

print('Gaussian Naive-Bayes - Test Set (Oversampling) \n',
      classification_report(y_test, gnb_predictions_test, target_names=target_names, zero_division=1),
      '\n -------------------------------------------------------------------------------------------------')

gnb_predictions_validation = gnb_opt.best_estimator_.predict(X_validation)

print('Gaussian Naive-Bayes - Validation Set (Oversampling) \n',
      classification_report(y_test, gnb_predictions_validation, target_names=target_names, zero_division=1),
      '\n -------------------------------------------------------------------------------------------------')

# -------------------------------------------------------------------------------------------------------
# NN >> Keras!

# -------------------------------------------------------------------------------------------------------
# Gaussian Process Classifier

kernel = 1.0 * RBF(1.0)

gpc = GaussianProcessClassifier(kernel=kernel,
                                random_state=0,
                                max_iter_predict=1000).fit(X_train, y_train.values.ravel())

gpc_predictions_test = gpc.predict(X_test)

print('GPC - Test Set (Oversampling) \n',
      classification_report(y_test, gpc_predictions_test, target_names=target_names, zero_division=1),
      '\n -------------------------------------------------------------------------------------------------')

gpc_predictions_validation = gpc.predict(X_validation)

print('GPC - Validation Set (Oversampling) \n',
      classification_report(y_test, gpc_predictions_validation, target_names=target_names, zero_division=1),
      '\n -------------------------------------------------------------------------------------------------')

# -------------------------------------------------------------------------------------------------------

# Ideas to explore
# Data scaling before all ML algos?
# How to introduce cross validation on all of these
# KERAS for NN
# Continue to explore label prop and semi-supervised algorithms
# Effects of upsampling in performance? How much to upsample?
# Probit implementation stats models


"""
# Transduction model
n_total_samples = len(y)
n_labeled_points = len(y_train)

labeled_indices = np.arange(n_total_samples)[:n_labeled_points]
unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]

max_iterations = 3

X_ = X.values
min_max_scaler = preprocessing.MinMaxScaler()
X_ = min_max_scaler.fit_transform(X_)


for i in range(max_iterations):
    if len(unlabeled_indices) == 0:
        print("No unlabeled items left to label.")
        break

    y_train = np.copy(np.ravel(np.array(y).reshape(len(y),)))
    y_train[unlabeled_indices] = -1

    lp_model = LabelSpreading(gamma=0.25, max_iter=20)
    # lp_model.fit(X.to_numpy(), y_train)
    lp_model.fit(X_, y_train)
    predicted_labels = lp_model.transduction_[unlabeled_indices]
    true_labels = np.array(y)[unlabeled_indices]

    cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)

    print("Iteration %i %s" % (i, 70 * "_"))
    print("Label Spreading model: %d labeled & %d unlabeled (%d total)"
           % (n_labeled_points, n_total_samples - n_labeled_points,
              n_total_samples))

    print(classification_report(true_labels, predicted_labels))

    print("Confusion matrix")
    print(cm)
"""
