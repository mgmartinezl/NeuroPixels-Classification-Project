"""
Project: Neuropixels classification pipeline
Author: Gabriela Martinez
Script: Classifier.py
Description: supervised classification on top of cerebellar cortical cells augmented via SMOTE interpolation

"""

import pandas as pd
import numpy as np
from numpy.random import seed
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.gaussian_process.kernels import RBF
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import warnings
from random import randrange
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

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

# print(df.groupby('Label').count())


def only_responsive_labels(row):
    if row['RespPk'] == 1 or row['RespSS'] == 1:
        return 1
    elif row['RespCS'] == 1 or row['SubCluster'] == 'CS':
        return 2
    elif row['RespGr'] == 1 or row['SubCluster'] == 'Gr':
        return 3
    elif row['RespGo'] == 1:
        return 4
    elif row['RespMF'] == 1 or row['MF_baseline'] == 1:
        return 5
    elif row['RespMLI'] == 1:
        return 6
    else:
        return -1  # Unidentified observations


def only_responsive_labels_truly(row):
    if row['RespPk'] == 1 or row['RespSS'] == 1:
        return 1
    elif row['RespCS'] == 1:
        return 2
    elif row['RespGr'] == 1:
        return 3
    elif row['RespGo'] == 1:
        return 4
    elif row['RespMF'] == 1:
        return 5
    elif row['RespMLI'] == 1:
        return 6
    else:
        return -1  # Unidentified observations


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


df['GroundLabel'] = df.apply(only_responsive_labels, axis=1)
df['GroundLabel_Truly'] = df.apply(only_responsive_labels_truly, axis=1)
df['Baseline'] = df.apply(baseline, axis=1)
# print(df.groupby('GroundLabel').count())

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

# cols_to_use = ['MFRBlockHz',
#                'tf_MedIsi',
#                'tf_Perc5Isi',
#                'tf_Entropy',
#                'tf_CV2Mean',
#                'tf_LvR'
#                ]

target_names = ['Pk', 'CS', 'Gr', 'Go', 'MF']
# -------------------------------------------------------------------------------------------------------

# Baseline model
ground_truth = df[df.GroundLabel_Truly != -1]
y_groundtruth = ground_truth['GroundLabel_Truly']
y_baseline = ground_truth['Baseline']

print('Baseline model  \n',
      classification_report(y_groundtruth, y_baseline, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')


# ---------------------------------------------------------------------------------------------------------------------
# Discard unknowns
ground_truth = df[df.GroundLabel != -1]

# MLI will not be considered
ground_truth = ground_truth[ground_truth.GroundLabel != 6]
# print(ground_truth.groupby('GroundLabel').count())

# Class weights
Pk_p = round(ground_truth[ground_truth['GroundLabel'] == 1].shape[0] / ground_truth.shape[0], 2)
CS_p = round(ground_truth[ground_truth['GroundLabel'] == 2].shape[0] / ground_truth.shape[0], 2)
Gr_p = round(ground_truth[ground_truth['GroundLabel'] == 3].shape[0] / ground_truth.shape[0], 2)
Go_p = round(ground_truth[ground_truth['GroundLabel'] == 4].shape[0] / ground_truth.shape[0], 2)
MF_p = round(ground_truth[ground_truth['GroundLabel'] == 5].shape[0] / ground_truth.shape[0], 2)
# MLI_p = round(ground_truth[ground_truth['GroundLabel'] == 6].shape[0] / ground_truth.shape[0], 2) # not considered

weights = {1: Pk_p, 2: CS_p, 3: Gr_p, 4: Go_p, 5: MF_p}

# Number of samples to synthetically generate
size = 500

# SMOTE implementation
t2 = {1: size, 2: size, 3: size, 4: size, 5: size}
sm = SMOTE(random_state=42, sampling_strategy=t2, k_neighbors=8)

X_training = ground_truth[cols_to_use]
y_training = ground_truth['GroundLabel']

X_training_smote, y_training_smote = sm.fit_resample(X_training, y_training)

# General values of regularization strength and tolerance criterion to tune hyper-parameters
C_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100]
tol_values = [1e-5, 1e-4, 1e-3]
random_state = 0

# Logistic Regression
# ---------------------------------------------------------------------------------------------------------------------

parameters = {'solver': ['saga', 'liblinear'],
              'penalty': ['l1', 'l2'],
              'multi_class': ['multinomial', 'ovr'],
              'tol': tol_values,
              'C': C_values,
              'class_weight': [weights, 'balanced']
              }

lr = LogisticRegression(random_state = random_state)

# Hyper-parameter tuning via grid search
lr_opt = GridSearchCV(lr, parameters,scoring = 'f1_macro',cv=3)

# Cross-validation on augmented data set
lr_opt.fit(X_training_smote, y_training_smote)

print('Best LogReg parameters \n',
      lr_opt.best_params_)

# Cross-validation for singleton predictions
y_pred = cross_val_predict(lr_opt, X_training_smote, y_training_smote, cv=5)

print('LogisticRegression - Cross-Validated Classification Report (augmented data) \n',
      classification_report(y_training_smote, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

# Cross-validation on ground-truth data
lr_opt.fit(X_training, y_training)

print('Best LogReg parameters \n',
      lr_opt.best_params_)

# {'C': 0.9, 'class_weight': 'balanced', 'multi_class': 'ovr', 'penalty': 'l1', 'solver': 'liblinear', 'tol': 0.001}

y_pred = cross_val_predict(lr_opt, X_training, y_training, cv=5)

print('LogisticRegression - Cross-Validated Classification Report (ground-truth data) \n',
      classification_report(y_training, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

# RFE
tuned_lr = lr = LogisticRegression(random_state=1234, C=0.9, class_weight='balanced', multi_class='ovr',
                                   penalty='l1', solver='liblinear', tol=0.001)

selector = RFE(tuned_lr, n_features_to_select=5, step=1)
pipeline = Pipeline(steps=[('s', selector), ('m', tuned_lr)])
selector = selector.fit(X_training, y_training)

y_pred = cross_val_predict(selector, X_training, y_training, cv=5)

print('LogisticRegression - Cross-Validated Classification Report (RFE) \n',
      classification_report(y_training, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

# print(selector.support_)
# print(selector.ranking_)


# -------------------------------------------------------------------------------------------------------
# Passive-Aggressive Classifier

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
              
pac = PassiveAggressiveClassifier(random_state = random_state)

# Hyper-parameter tuning via grid search
pac_opt = GridSearchCV(estimator=pac, param_grid=parameters, scoring='f1_macro', cv=3)

# Cross-validation for singleton predictions
y_pred = cross_val_predict(pac_opt, X_training_smote, y_training_smote, cv=5)

print('PassiveAggressiveClassifier - Cross-Validated Classification Report \n',
      classification_report(y_training_smote, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')


# Tree-based models
# -------------------------------------------------------------------------------------------------------
# Decision Tree

parameters = {'max_depth': [4, 5, 6, 7],
              'criterion': ['entropy', 'gini'],
              'min_samples_split': [5, 6, 7, 8, 9, 10],
              'min_samples_leaf': [5, 6, 7, 8, 9, 10],
              'class_weight': [weights, 'balanced']
              }

dt = DecisionTreeClassifier()

dt_opt = GridSearchCV(estimator=dt, param_grid=parameters, scoring='f1_macro', cv=3)

# Cross-validation for augmented data
dt_opt.fit(X_training_smote, y_training_smote)

print('Best DecisionTree parameters \n',
      dt_opt.best_params_)

y_pred = cross_val_predict(dt_opt, X_training_smote, y_training_smote, cv=5)

print('DecisionTreeClassifier - Cross-Validated Classification Report (augmented data) \n',
      classification_report(y_training_smote, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

# Cross-validation for ground-truth data
dt_opt.fit(X_training, y_training)

print('Best DecisionTree parameters \n',
      dt_opt.best_params_)

# {'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 7, 'min_samples_split': 5}

y_pred = cross_val_predict(dt_opt, X_training, y_training, cv=5)

print('DecisionTreeClassifier - Cross-Validated Classification Report (ground-truth data) \n',
      classification_report(y_training, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')


# -------------------------------------------------------------------------------------------------------
# Random Forest

rf = RandomForestClassifier()

parameters = {'max_depth': [4, 5, 6, 7],
              'criterion': ['entropy', 'gini'],
              'min_samples_split': [5, 6, 7, 8, 9, 10],
              'min_samples_leaf': [5, 6, 7, 8, 9, 10],
              'class_weight': [weights, 'balanced']
              }

rf_opt = GridSearchCV(estimator=rf, param_grid=parameters, scoring='f1_macro', cv=3)

# Cross-validation for augmented data
rf_opt.fit(X_training_smote, y_training_smote)

print('Best RandomForest parameters \n',
      rf_opt.best_params_)

y_pred = cross_val_predict(rf_opt, X_training_smote, y_training_smote, cv=5)

print('RandomForestClassifier - Cross-Validated Classification Report (augmented data) \n',
      classification_report(y_training_smote, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

# Cross-validation for ground-truth data
rf_opt.fit(X_training, y_training)

print('Best RandomForest parameters \n',
      rf_opt.best_params_)

# {'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 9, 'min_samples_split': 8}

y_pred = cross_val_predict(rf_opt, X_training, y_training, cv=5)

print('RandomForestClassifier - Cross-Validated Classification Report (ground-truth data) \n',
      classification_report(y_training, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')


# -------------------------------------------------------------------------------------------------------
# SVM

# SVMs are sensitive to data scaling!
# scale = StandardScaler()
# X_train = scale.fit_transform(X_train)

svm = SVC()

parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'degree': [2, 3],
              'gamma': ['scale', 'auto'],
              'break_ties': [False, True],
              'random_state': [0],
              'tol': [1e-3],
              'C': C_values,
              'class_weight': [weights, 'balanced']
              }

svm_opt = GridSearchCV(estimator=svm, param_grid=parameters, scoring='f1_macro', cv=3)

# Cross-validation on augmented data
svm_opt.fit(X_training_smote, y_training_smote)

print('Best SVC parameters \n',
      svm_opt.best_params_)

y_pred = cross_val_predict(svm_opt, X_training_smote, y_training_smote, cv=5)

print('SVC - Cross-Validated Classification Report (augmented data) \n',
      classification_report(y_training_smote, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

# Cross-validation on ground-truth data
svm_opt.fit(X_training, y_training)

print('Best SVC parameters \n',
      svm_opt.best_params_)
      
# For ground-truth data >> {'C': 0.05, 'break_ties': True, 'class_weight': 'balanced', 'degree': 2, 'gamma': 'scale',
# 'kernel': 'linear', 'random_state': 0, 'tol': 0.001}

y_pred = cross_val_predict(svm_opt, X_training, y_training, cv=5)

print('SVC - Cross-Validated Classification Report (ground-truth data) \n',
      classification_report(y_training, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')


# -------------------------------------------------------------------------------------------------------
# KNN

knn = KNeighborsClassifier()

parameters = {'n_neighbors': [2, 3, 4, 5, 6, 7],
              'weights': ['uniform', 'distance'],
              'algorithm': ['kd_tree', 'brute', 'auto'],
              'p': [1, 2]
              }

knn_opt = GridSearchCV(estimator=knn, param_grid=parameters, scoring='f1_macro', cv=3)

# Cross-validation on augmented data
knn_opt.fit(X_training_smote, y_training_smote)

print('Best KNN parameters \n',
      knn_opt.best_params_)

y_pred = cross_val_predict(knn_opt, X_training_smote, y_training_smote, cv=5)

print('KNN - Cross-Validated Classification (augmented data) \n',
      classification_report(y_training_smote, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

# Cross-validation on ground-truth data
knn_opt.fit(X_training, y_training)

print('Best KNN parameters \n',
      knn_opt.best_params_)

# {'algorithm': 'kd_tree', 'n_neighbors': 3, 'p': 1, 'weights': 'uniform'}

y_pred = cross_val_predict(knn_opt, X_training, y_training, cv=5)

print('KNN - Cross-Validated Classification Report (ground-truth data) \n',
      classification_report(y_training, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')


# -------------------------------------------------------------------------------------------------------
# Gaussian Naive-Bayes

gnb = GaussianNB()

parameters = {'var_smoothing': [1e-9, 1e-5, 1, 1.1, 1.2, 1.5]}

gnb_opt = GridSearchCV(estimator=gnb, param_grid=parameters, scoring='f1_macro', cv=3)

# Cross-validation on augmented data
y_pred = cross_val_predict(gnb_opt, X_training_smote, y_training_smote, cv=5)

print('GaussianNB - Cross-Validated Classification Report (augmented data) \n',
      classification_report(y_training_smote, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

# Cross-validation on ground-truth data
gnb_opt.fit(X_training, y_training)

print('Best NB parameters \n',
      gnb_opt.best_params_)

y_pred = cross_val_predict(gnb_opt, X_training, y_training, cv=5)

print('GaussianNB - Cross-Validated Classification Report (ground-truth data) \n',
      classification_report(y_training, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')


# -------------------------------------------------------------------------------------------------------
# Gaussian Process Classifier

kernel = 1.0 * RBF(1.0)

gpc = GaussianProcessClassifier(kernel=kernel,
                                random_state=0,
                                max_iter_predict=1000)

y_pred = cross_val_predict(gpc, X_training_smote, y_training_smote, cv=5)

print('GaussianProcessClassifier - Cross-Validated Classification Report \n',
      classification_report(y_training_smote, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

# -------------------------------------------------------------------------------------------------------
# Method 1: XGBoost tuning (not straightforward process via grid search)

X_train, X_test, y_train, y_test = train_test_split(X_training_smote, y_training_smote,
                                                    test_size=0.4,
                                                    random_state=0,
                                                    stratify=y_training_smote)

D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)

params = {'eta': 0.3, 'max_depth': 6, 'min_child_weight': 1, 'subsample': 1, 'colsample_bytree': 1,
          'objective': 'multi:softprob', 'num_class': 5, 'eval_metric': "mlogloss"}

num_boost_round = 999
early_stopping_rounds = 10

# >> First parameters to tune >> max_depth, min_child_weight

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

# >> Second set of parameters to tune >> subsample, colsample_bytree

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

# Finally, tuning of the learning rate
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

print('XGBoost parameter tuning performance - Test Set (augmented data) \n',
      classification_report(y_test, xgboost_preds_test, target_names=target_names, zero_division=1),
      '\n -------------------------------------------------------------------------------------------------')

# -------------------------------------------------------------------------------------------------------
# Method 2: XGBoost optimization via sciki-learn embedding

xgboost = xgb.XGBClassifier()

parameters = {'eta': [.3, .2, .1, .05, .01, .005],
              'max_depth': [9, 10, 11, 12],
              'min_child_weight': [5, 6, 7, 8],
              'objective': ['multi:softprob'],
              'num_class': [6],
              'eval_metric': ["mlogloss"],
              'subsample': [i / 10. for i in range(7, 11)],
              'colsample_bytree': [i / 10. for i in range(7, 11)]
              }

xgboost_opt = GridSearchCV(xgboost, parameters, scoring='f1_macro', cv=3)

# Cross-validation on augmented data
xgboost_opt.fit(X_training_smote, y_training_smote)

print('Best XGBoost parameters \n',
      xgboost_opt.best_params_)

y_pred = cross_val_predict(xgboost_opt, X_training_smote, y_training_smote, cv=5)

print('XGBoost - Cross-Validated Classification Report (augmented data) \n',
      classification_report(y_training_smote, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

# Cross-validation on ground-truth data
xgboost_opt.fit(X_training, y_training)

print('Best XGBoost parameters \n',
      xgboost_opt.best_params_)

y_pred = cross_val_predict(xgboost_opt, X_training, y_training, cv=5)
# {'colsample_bytree': 0.9, 'eta': 0.2, 'eval_metric': 'mlogloss', 'max_depth': 9, 'min_child_weight': 7,
# 'num_class': 6, 'objective': 'multi:softprob', 'subsample': 1.0}

print('XGBoost - Cross-Validated Classification Report (ground-truth data) \n',
     classification_report(y_training, y_pred, target_names=target_names, zero_division=1),
     '------------------------------------------------------------------------------------------------- \n')

# RFE
tuned_xgboost = xgb.XGBClassifier(colsample_bytree=0.9, eta=0.2, eval_metric='mlogloss', max_depth=9, min_child_weight=7,
                                  num_class=6, objective='multi:softprob', subsample=1.0)

selector = RFECV(estimator=tuned_xgboost)
pipeline = Pipeline(steps=[('s', selector), ('m', tuned_xgboost)])
selector = selector.fit(X_training, y_training)

y_pred = cross_val_predict(selector, X_training, y_training, cv=5)

print('XGBoost - Cross-Validated Classification Report (RFE) \n',
      classification_report(y_training, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')

# print(selector.ranking_)


# -------------------------------------------------------------------------------------------------------
# GradientBoostingClassifier

GBC = GradientBoostingClassifier(random_state=0)

parameters = {'loss': ['deviance', 'exponential'],
              'learning_rate': [.3, .2, .1, .05, .01, .005],
              'subsample': [i / 10. for i in range(7, 11)],
              'criterion': ['friedman_mse', 'mse', 'mae'],
              'max_depth': [9, 10, 11, 12],
              'min_samples_split': [5, 6, 7, 8, 9, 10],
              'min_samples_leaf': [5, 6, 7, 8, 9, 10],
              }

GBC_opt = GridSearchCV(GBC, parameters, scoring='f1_macro', cv=3)

GBC_opt.fit(X_training_smote, y_training_smote)

print('Best GradientBoosting parameters \n',
      GBC_opt.best_params_)

y_pred = cross_val_predict(GBC, X_training_smote, y_training_smote, cv=5)

print('GradientBoostingClassifier - Cross-Validated Classification Report (augmented data) \n',
      classification_report(y_training_smote, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')


# -------------------------------------------------------------------------------------------------------
#  Models trained on ground-truth data only

# Hyper-parameters tuned
# xgboost_params = {'eta': 0.3,
#                   'max_depth': 9,
#                   'min_child_weight': 5,
#                   'subsample': 1,
#                   'colsample_bytree': 0.8,
#                   'random_state': [0],
#                   'objective': 'multi:softprob',
#                   'num_class': 6,
#                   'eval_metric': "mlogloss"
#                   }

# xgboost = xgb.XGBClassifier(parameters=xgboost_params)
xgboost = xgb.XGBClassifier(scale_pos_weight=0.1)

# svm = SVC(kernel='poly', degree=3,
#           break_ties=False,
#           random_state=0,
#           tol=0.001,
#           C=0.2,
#           class_weight='balanced',
#           probability=True)

svm = SVC()

# knn = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=2, p=1, weights='distance')
knn = KNeighborsClassifier(n_neighbors=2)

# lr = LogisticRegression(random_state=0,
#                         C=100,
#                         class_weight='balanced',
#                         multi_class='ovr',
#                         penalty='l1',
#                         solver='liblinear',
#                         tol=1e-05)

# lr = LogisticRegression(C= 0.9, class_weight= 'balanced', multi_class= 'ovr', penalty= 'l1', solver= 'liblinear',
# tol= 0.001)
lr2 = LogisticRegression()

RF = RandomForestClassifier()

gnb = GaussianNB()

# Ensemble model
eclf = VotingClassifier(estimators=[('XGBoost', xgboost), ('LogReg', lr2), ('GnB', gnb), ('SVC', svm)],
                        voting='hard')

# eclf1 = VotingClassifier(estimators=[('XGBoost', xgboost), ('GBC', GBC)], voting='soft')
eclf = eclf.fit(X_training, y_training)
y_pred = cross_val_predict(eclf, X_training, y_training, cv=5)

print('VotingClassifier - Cross-Validated Classification Report (ground-truth data) \n',
      classification_report(y_training, y_pred, target_names=target_names, zero_division=1),
      '------------------------------------------------------------------------------------------------- \n')