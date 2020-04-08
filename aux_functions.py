# import numpy as np
# import pandas as pd
#
# import os.path as op;opj=op.join
# from pathlib import Path
#
# from rtn import npa
# from rtn.npix.gl import get_units, load_units_qualities
# from rtn.utils import peakdetect
# from rtn.npix.io import ConcatenatedArrays, _pad, _range_from_slice, read_spikeglx_meta, chan_map
# from rtn.npix.spk_t import trn, isi, mfr
# from rtn.npix.corr import crosscorrelate_cyrille, ccg, acg
# from rtn.npix.spk_wvf import wvf, templates, get_peak_chan, get_depthSort_peakChans
# from rtn.npix.plot import plot_wvf, hist_MB, plot_raw, plot_raw_units, plot_acg, plot_ccg
# from rtn.npix.circuitProphyler import Prophyler, Dataset, Unit
#
# import matplotlib.pyplot as plt
#
# import seaborn as sns
# import os
# from dataclasses import dataclass
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib
# from typing import *
# from scipy import stats
#
# # Auxiliary function to draw
# Patch = matplotlib.patches.Patch
# PosVal = Tuple[float, Tuple[float, float]]
# Axis = matplotlib.axes.Axes
# PosValFunc = Callable[[Patch], PosVal]
#
# @dataclass
# class AnnotateBars:
#     font_size: int = 10
#     color: str = "black"
#     n_dec: int = 2
#
#     def horizontal(self, ax: Axis, centered=False):
#         def get_vals(p: Patch) -> PosVal:
#             value = p.get_width()
#             div = 2 if centered else 1
#             pos = (
#                 p.get_x() + p.get_width() / div,
#                 p.get_y() + p.get_height() / 2,
#             )
#             return value, pos
#         ha = "center" if centered else  "left"
#         self._annotate(ax, get_vals, ha=ha, va="center")
#
#     def vertical(self, ax: Axis, centered:bool=False):
#         def get_vals(p: Patch) -> PosVal:
#             value = p.get_height()
#             div = 2 if centered else 1
#             pos = (p.get_x() + p.get_width() / 2,
#                    p.get_y() + p.get_height() / div
#             )
#             return value, pos
#         va = "center" if centered else "bottom"
#         self._annotate(ax, get_vals, ha="center", va=va)
#
#     def _annotate(self, ax, func: PosValFunc, **kwargs):
#         cfg = {"color": self.color,
#                "fontsize": self.font_size, **kwargs}
#         for p in ax.patches:
#             if p.get_height() != 0:
#                 value, pos = func(p)
#                 ax.annotate(f"{value:.{self.n_dec}f}", pos, **cfg)
#
#
# def nice_plot(plot_obj, x_label, y_label, title):
#     plot_obj.spines["top"].set_visible(False)
#     plot_obj.spines["right"].set_visible(False)
#     plot_obj.set(xlabel=x_label, ylabel=y_label)
#     plot_obj.spines['bottom'].set_color('#939799')
#     plot_obj.spines['left'].set_color('#939799')
#     plot_obj.yaxis.label.set_color('#939799')
#     plot_obj.xaxis.label.set_color('#939799')
#     plot_obj.tick_params(axis='x', colors='#939799')
#     plot_obj.tick_params(axis='y', colors='#939799')
#     plot_obj.tick_params(axis='both', which='minor', labelsize=9)
#     plot_obj.tick_params(axis='both', which='major', labelsize=9)
#     plot_obj.set_title(title, fontsize=9, fontname="DejaVu Sans", loc='center', color='#939799')
#     return plot_obj
#
#
#
# def plot_units_composition(cell_list):
#     n = 1
#
#     sample = {}
#     results = {}
#
#     for dp in cell_list:
#
#         year = dp[11:13]
#         date = dp[12:20]
#         cell = dp[8:11]
#
#         unit_qualities = load_units_qualities(dp)
#         total_units = len(unit_qualities)
#
#         good_units = get_units(dp, quality='good')
#         good_units = len(good_units)
#         prop_good_units = round(good_units / total_units * 100, 2)
#
#         mua_units = get_units(dp, quality='mua')
#         mua_units = len(mua_units)
#         prop_MUA_units = round(mua_units / total_units * 100, 2)
#
#         noise_units = get_units(dp, quality='noise')
#         noise_units = len(noise_units)
#         prop_noise_units = round(noise_units / total_units * 100, 2)
#
#         total_classified = good_units + mua_units + noise_units
#         prop_classified_units = round(total_classified / total_units * 100, 2)
#         total_unassigned = total_units - total_classified
#
#         if total_unassigned != 0:
#             prop_unassigned_units = round(total_unassigned / total_units * 100, 2)
#         else:
#             prop_unassigned_units = 0
#
#         sample["Sample_{}".format(n)] = [cell, year, date, total_units, good_units, prop_good_units, mua_units,
#                                          prop_MUA_units,
#                                          noise_units, prop_noise_units, total_classified, prop_classified_units,
#                                          total_unassigned, prop_unassigned_units]
#
#         results.update(sample)
#         n += 1
#
#     import matplotlib.pyplot as plt
#     from itertools import cycle, islice
#
#     # Transform dict to pandas
#     df = pd.DataFrame(results).T
#     df = df.reset_index()
#     df.columns = ['Sample', 'Cell-Type', 'Year', 'Date', 'Total', 'Good', '% Good', 'MUA', '% MUA',
#                   'Noise', '% Noise', 'Classified', '% Classified', 'Unassigned', '% Unassigned']
#
#     df['Date'] = pd.to_datetime(df['Date'], format='%y-%m-%d').dt.date
#     df = df.sort_values(by='Date')
#
#     # Define colors
#     # my_colors = list(islice(cycle(['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#CFECF9', '#7F7F7F', '#BCBD22', '#17BECF']), None, len(df)))
#     my_colors = list(islice(cycle(
#         ['#4E79A7', '#F28E2C', '#E15759', '#76B7B2', '#59A14F', '#EDC949', '#AF7AA1', '#FF9DA7', '#9C755F', '#BAB0AB']),
#                             None, len(df)))
#
#     # Define one plane
#     fig, axes = plt.subplots(1, 1)
#
#     # Plot values
#     gx = df.plot.bar(y=['% Good', '% MUA', '% Noise', '% Unassigned'], figsize=(15, 10), stacked=True, color=my_colors,
#                      ax=axes)
#
#     # Annotate proportions
#     AnnotateBars(font_size=8, n_dec=1, color="white").vertical(axes, True)
#
#     # Title
#     gx.set_title('Proportion of units by their quality in {} samples'.format(cell), fontsize=20, fontname="DejaVu Sans",
#                  pad=20, loc='left', color='#939799')
#
#     # No frame
#     gx.spines["top"].set_visible(False)
#     gx.spines["right"].set_visible(False)
#
#     # Axis and labels colors
#     gx.legend(loc='best')
#     gx.set(xlabel="Samples", ylabel="% of units per quality bucket")
#     gx.set_xticklabels(df['Date'], rotation=25)
#     gx.spines['bottom'].set_color('#939799')
#     gx.spines['left'].set_color('#939799')
#     gx.yaxis.label.set_color('#939799')
#     gx.xaxis.label.set_color('#939799')
#     gx.tick_params(axis='x', colors='#939799')
#     gx.tick_params(axis='y', colors='#939799')
#     gx.tick_params(axis='both', which='minor', labelsize=8)
#     gx.tick_params(axis='both', which='major', labelsize=10)
#
#     plt.show()
#
#     return gx
#
#
# # def goodness_fit(distributions, data):
# #
# #     for dist in distributions.keys():
# #         global MA_chunk_is_gaussian
# #
# #         KS_test = stats.kstest(data, dist, distributions[dist].fit(data))
# #         p_value = KS_test[1]
# #
# #         if p_value < 0.005:
# #             MA_chunk_is_gaussian = MA_chunk_is_gaussian
# #             print(f"H0 REJECTED (95% conf) >> AMP data not coming from {dist} distribution")
# #             if p_value < 0.001:
# #                 print(f"H0 REJECTED (99% conf) >> AMP data not coming from {dist} distribution")
# #         else:
# #             MA_chunk_is_gaussian = 1
# #             print(f"H0 ACCEPTED (95% conf) >> AMP data coming from {dist} distribution >> ")
# #
# #         return MA_chunk_is_gaussian
#
#
# def peak_detector_norm_waveform(waveform):
#
#     template = waveform.tolist()
#
#     q75, q25 = np.percentile(template, [75, 25])
#     # iqr = q75 - q25
#
#     median = np.median(template)
#
#     max_negative_outlier_index = template.index(-1.)
#     # max_negative_outlier = template[max_negative_outlier_index]
#
#     positives_before = [i for i in template if template.index(i) < max_negative_outlier_index]
#     max_positive_before = np.max(positives_before)
#     # max_positive_before_index = template.index(max_positive_before)
#
#     positives_after = [i for i in template if template.index(i) > max_negative_outlier_index]
#     max_positive_after = np.max(positives_after)  # 1
#     # max_positive_after_index = template.index(max_positive_after)
#
#     amp_max_positive_after = max_positive_after - median
#     diff = np.abs(max_positive_after - max_positive_before)
#
#     thr = amp_max_positive_after * 0.6
#
#     if (max_positive_before < max_positive_after) and (diff > thr):
#         # print('First peak NEGATIVE')
#         first_peak_negative = 1
#     else:
#         # print('First peak NOT NEGATIVE')
#         first_peak_negative = 0
#
#     return first_peak_negative
#
