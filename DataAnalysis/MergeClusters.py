import pandas as pd

# df1 = pd.read_csv('WardClustering_WithNonQualityResponsiveUnits_3Classes_5Responsive_AllFeatures.csv')
# df2 = pd.read_csv('WardClustering_GranularMolecularLayer_4clusters.csv')
#
# print(len(df1.columns))
# print(len(df2.columns))
#
# m = pd.merge(df1, df2[['Sample', 'Unit', 'Cluster']], how='outer', on=['Sample', 'Unit'])
#
# def f(row):
#     if row['Cluster_x'] == 2 and row['Cluster_y'] == 1:
#         val = 3
#         return val
#     elif row['Cluster_x'] == 2 and row['Cluster_y'] == 2:
#         val = 4
#         return val
#     elif row['Cluster_x'] == 2 and row['Cluster_y'] == 3:
#         val = 5
#         return val
#     elif row['Cluster_x'] == 2 and row['Cluster_y'] == 4:
#         val = 6
#         return val
#     elif row['Cluster_x'] == 3:
#         val = 2
#         return val
#     elif row['Cluster_x'] == 1:
#         val = 1
#         return val
#
# m['SmallCluster'] = m.apply(f, axis=1)
#
#
# def f1(row):
#     if row['SmallCluster'] == 1:
#         return 'Purkinke'
#     elif row['SmallCluster'] == 2:
#         return 'Complex Spike'
#     elif row['SmallCluster'] == 3:
#         return 'Potentially MLI'
#     elif row['SmallCluster'] == 4:
#         return 'Granule'
#     elif row['SmallCluster'] == 5:
#         return 'Go + MF'
#     else:
#         return 'Unknown'
#
#
# m['SmallClusterName'] = m.apply(f1, axis=1)

########################################

# x = pd.read_csv('Neuropixels_FeaturesWithNonQualResponsiveUnits.csv')
# m = pd.read_csv('WardClustering_NonQualityUnits_5Classes_AllFeatures.csv')
#
# g = pd.merge(m, x[['MeanAmpBlock', 'Sample', 'Unit',
#                    'ACGViolationRatio', 'tf_MIFRBlockHz',
#                    'tf_ModeIsi', 'tf_Perc5Isi',
#                    'tf_CV2Median', 'tf_CV',
#                    'tf_Ir', 'tf_Lv',
#                    'tf_Si', 'tf_skw',
#                    'wf_EndSlopeTau']], how='outer', on=['Sample', 'Unit'])
#
#
# g.to_csv('WardClustering_NonQualityUnits_5Classes_AllFeatures.csv', index=False)
#
# print(g.head())

#######################################

# x = pd.read_csv('WardClustering_NonQualityUnits_5Classes_AllFeatures.csv')
# y = pd.read_csv('WardClustering_ACGs_5Classes.csv')
#
# z = pd.merge(x, y[['Sample', 'Unit', 'ClusterACG']], on=['Sample', 'Unit'])
# z.to_csv('WardClustering_ACGs_and_OtherFeatures_5Classes.csv', index=False)

#######################################

# x = pd.read_csv('../Images/Pipeline/Sample_20-15-06_DK186_probe1/Features/Files/Sample-20-15-06_DK186-unit-425.csv')
# x1 = pd.read_csv('../Images/Pipeline/Sample_20-15-06_DK186_probe1/Features/Files/Sample-20-15-06_DK186-unit-425.csv')
# x2 = pd.read_csv('../Images/Pipeline/Sample_20-15-06_DK186_probe1/Features/Files/Sample-20-15-06_DK186-unit-425.csv')
# x3 = pd.read_csv('../Images/Pipeline/Sample_20-15-06_DK186_probe1/Features/Files/Sample-20-15-06_DK186-unit-425.csv')
# x4 = pd.read_csv('../Images/Pipeline/Sample_20-15-06_DK186_probe1/Features/Files/Sample-20-15-06_DK186-unit-425.csv')

#frames = [x, x1, x2, x3, x4]

#y = pd.concat(frames)

#y.to_csv('Merged.csv', index=False)

x = pd.read_csv('Merged.csv')
y = pd.read_csv('UltimateData.csv')

z = pd.merge(x, y, on=['Sample', 'Unit'])

print(z)

# z.rename(columns={"Cluster": "Purkinje", "B": "c"})

# z.to_csv('WardClustering_ACGs_and_OtherFeatures_5Classes_Comparison.csv', index=False)
