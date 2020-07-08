import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def patch_violinplot(palette, n):
    from matplotlib.collections import PolyCollection
    ax = plt.gca()
    violins = [art for art in ax.get_children() if isinstance(art, PolyCollection)]
    colors = sns.color_palette(palette, n_colors=n) * (len(violins)//n)
    for i in range(len(violins)):
        violins[i].set_edgecolor(colors[i])


df = pd.read_csv('WardClustering_3Clusters.csv')

# Violin plot
# fig, ax = plt.subplots(figsize=(8,5))
# ax.set_ylim(-10, 300)

# Option 1
colors = ["salmon", "light blue", "windows blue"]
metrics = ['MFRBlockHz', 'tf_Entropy', 'tf_CV2Mean', 'tf_MedIsi', 'tf_LvR']
legends = ['MFR', 'Entropy', 'Mean CV2', 'Median ISI', 'LvR']
num_cols = df['Cluster'].nunique()

# for metric in metrics:
#     ax = sns.violinplot(data=df,
#                         x='Cluster',
#                         y=metric,
#                         scale='width',
#                         bw=2.5,
#                         palette=sns.xkcd_palette(colors))
#
#
#     ax.spines["right"].set_visible(False)
#     ax.spines["top"].set_visible(False)
#     ax.spines['bottom'].set_color('#939799')
#     ax.spines['left'].set_color('#939799')
#     ax.tick_params(axis='x', colors='#939799')
#     ax.tick_params(axis='y', colors='#939799')
#     ax.yaxis.label.set_color('#939799')
#     ax.xaxis.label.set_color('#939799')
#     ax.set(xlabel="Cluster", ylabel=legends[metrics.index(metric)])
#     patch_violinplot(sns.xkcd_palette(colors), num_cols)
#     plt.show()

# Option 2
# for i, key in enumerate([1, 2, 3]):
#     ax.violinplot(df[df.Cluster == key]["MFRBlockHz"].values,
#                   positions=[i],
#                   showmeans=True)

# ANOVA one-way
# PENDING! Check assumptions: normality, homogeneity of variance, and independence
# https://www.pythonfordatascience.org/anova-python/#assumption_check

# This assumption is tested when the study is designed.
# What this means is that all groups are mutually exclusive,
# i.e. an individual can only belong in one group.
# Also, this means that the data is not repeated measures
# (not collected through time). In this example, this condition is met.

import scipy.stats as stats

# ANOVA test
# for metric in metrics:
#     s = stats.f_oneway(df[metric][df['Cluster'] == 1],
#                        df[metric][df['Cluster'] == 2],
#                        df[metric][df['Cluster'] == 3])
#     print(s)

# Variance homogeneity assumption >> Failed! https://stats.stackexchange.com/questions/56971/alternative-to-one-way-anova-unequal-variance

# for metric in metrics:
#     s = stats.levene(df[metric][df['Cluster'] == 1],
#                      df[metric][df['Cluster'] == 2],
#                      df[metric][df['Cluster'] == 3])
#     print(s)

import statistics
for metric in metrics:

    s1 = statistics.variance(df[metric][df['Cluster'] == 1], statistics.mean(df[metric][df['Cluster'] == 1]))
    s2 = statistics.variance(df[metric][df['Cluster'] == 2], statistics.mean(df[metric][df['Cluster'] == 2]))
    s3 = statistics.variance(df[metric][df['Cluster'] == 3], statistics.mean(df[metric][df['Cluster'] == 3]))

    minimum_var = min(s1, s2, s3)
    maximum_var = max(s1, s2, s3)

    print(minimum_var)
    print(maximum_var)

    if maximum_var <= 4*minimum_var:
        print(f'Metric {metric} complies!')
