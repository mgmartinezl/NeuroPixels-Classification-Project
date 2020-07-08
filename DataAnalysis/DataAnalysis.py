import pandas as pd
import numpy as np
import seaborn
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read data
df = pd.read_csv('Neuropixels_FeaturesWithNonQualResponsiveUnits.csv')

# Features
features = [
            'SpikesBlock',
            # 'PercMissingSpikesBlock',
            'MFRBlockHz',
            'MeanAmpBlock',
            # 'ACGViolationRatio',
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
            'tf_LcV',
            'tf_Si',
            'tf_skw',
            # 'wf_MaxAmpNorm',
            # 'wf_Duration',
            # 'wf_PosHwDuration',
            # 'wf_NegHwDuration',
            # 'wf_Onset',
            # 'wf_End',
            # 'wf_Crossing',
            # 'wf_Pk10',
            # 'wf_Pk90',
            # 'wf_Pk50',
            # 'wf_PkTrRatio'
            # 'wf_DepolarizationSlope',
            # 'wf_RepolarizationSlope',
            # 'wf_RecoverySlope',
            # 'wf_RiseTime',
            # 'wf_PosDecayTime',
            # 'wf_FallTime',
            # 'wf_NegDecayTime'
            # 'wf_EndSlopeTau',
            # 'ResponsiveUnit_Pk'
            # 'ResponsiveUnit_GrC',
            # 'ResponsiveUnit_MF'
            # 'ResponsiveUnit_-1'
            ]

# print(df.head())

# Separating out the features
x = df.loc[:, features].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

# Apply PCA
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(x)
# principalDf = pd.DataFrame(data=principalComponents,
#                            columns=['principal component 1',
#                                     'principal component 2'])


# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlabel('Principal Component 1', fontsize=15)
# ax.set_ylabel('Principal Component 2', fontsize=15)
# ax.set_title('2 component PCA', fontsize=20)
# ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'])
# plt.show()

# print(principalDf.head())
#
#
# from mpl_toolkits.mplot3d import Axes3D
#
# # fig = plt.figure(1, figsize=(4, 3))
# # ax = Axes3D(fig)
# # ax.scatter(principalComponents['principal component 1'],
# #            principalComponents['principal component 2'],
# #            principalComponents['principal component 3'])
#
print(pca.explained_variance_ratio_)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
#
# ax.scatter(principalComponents[:, 'principal component 1'],
#            principalComponents[:, 'principal component 2'],
#            principalComponents[:, 'principal component 3'],
#            c=principalComponents['principal component 3'],
#            cmap='Greens')
# plt.show()
#
# fig = plt.figure()
# ax = Axes3D(fig)
