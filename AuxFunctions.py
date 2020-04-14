"""
Project: Neuropixels classification pipeline
Author: Gabriela Martinez
Script: AuxFunctions.py

"""

import matplotlib.pyplot as plt
from rtn import npa
from rtn.npix.gl import get_units, load_units_qualities
from rtn.npix.io import ConcatenatedArrays, _pad, _range_from_slice, read_spikeglx_meta, chan_map
from rtn.npix.spk_t import trn, isi, mfr
from rtn.npix.corr import crosscorrelate_cyrille, ccg, acg
from rtn.npix.spk_wvf import wvf, templates, get_peak_chan, get_depthSort_peakChans
from rtn.npix.plot import plot_wvf, hist_MB, plot_raw, plot_raw_units, plot_acg, plot_ccg
from rtn.npix.circuitProphyler import Prophyler, Dataset, Unit

from rtn.npix.corr import crosscorrelate_cyrille, ccg, acg
from rtn.utils import peakdetect
import numpy as np
import pandas as pd
import os.path as op;opj=op.join
from dataclasses import dataclass
import matplotlib
from typing import *
import scipy.optimize as opt
import scipy.stats as stats
from scipy.stats import iqr


def estimate_bins(x, rule):

    n = len(x)
    maxi = max(x)
    mini = min(x)

    # Freedman-Diaconis rule
    if rule == 'Fd':
        iqr_ = iqr(x)
        h = 2 * iqr_ * (n ** (-1/3))
        b = int(round((maxi-mini)/h, 0))
        return b

    # Square-root choice
    elif rule == 'Sqrt':
        b = int(np.sqrt(n))
        return b


def not_gaussian_amp_est(x, nBins):
    num, bins = np.histogram(x, bins=nBins)
    maxNum = max(num)
    maxNum_index = np.min(np.where(num == maxNum)[0])
    cutoff = np.where(num == 0)[0][-1]
    halfSym = num[int(maxNum_index):len(num)]
    steps = np.flipud(halfSym[1:])
    fullSym = np.concatenate([steps, halfSym])
    percent_missing = 100 * sum(fullSym[0:cutoff]) / sum(fullSym)
    percent_missing = int(round(percent_missing, 2))
    return percent_missing


def gaussian_amp_est(x, Nbins):
    x1, p0 = ampli_fit_gaussian_cut(x, Nbins)
    n_fit = gaussian_cut(x1, a=p0[0], mu=p0[1], sigma=p0[2], xcut=p0[3])
    min_amp = p0[3]
    n_fit_no_cut = gaussian_cut(x1, a=p0[0], mu=p0[1], sigma=p0[2], xcut=0)
    percent_missing = int(round(100 * stats.norm.cdf((min_amp - p0[1]) / p0[2]), 0))
    return x1, p0, min_amp, n_fit, n_fit_no_cut, percent_missing


def gaussian_cut(x, a, mu, sigma, xcut):
    g = a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    g[x < xcut] = 0
    return g


def curve_fit_(x, num, p1):
    popt = opt.curve_fit(gaussian_cut, x, num, p1, maxfev=10000)
    return popt


def ampli_fit_gaussian_cut(x, Nbins):
    a = np.asarray(x, dtype='float64')
    num, bins = np.histogram(a, bins=Nbins)
    mode_seed = bins[np.where(num == max(num))]
    bin_steps = np.diff(bins[0:2])[0]
    x = bins[0:len(bins) - 1] + bin_steps / 2
    next_low_bin = x[0] - bin_steps
    add_points = np.arange(start=next_low_bin, stop=0, step=-bin_steps)
    add_points = np.flipud(add_points)
    x = np.concatenate([add_points, x])
    zeros = np.zeros((len(add_points), 1))
    zeros = zeros.reshape(len(zeros), )
    num = np.concatenate([zeros, num])

    if len(mode_seed) > 1:
        mode_seed = np.mean(mode_seed)

    p0 = [np.max(num), mode_seed, np.nanstd(a), np.percentile(a, 1)]
    p0 = np.asarray(p0, dtype='float64')

    # Curve fit
    popt = curve_fit_(x, num, p0)
    p0 = popt[0]

    return x, p0


# def remove_duplicate_spikes(trn):
#     diffs = [y - x for x, y in zip(trn, trn[1:])]
#     unique_mask = []
#     z = [False if i <= 0.5 else True for i in diffs]
#     unique_mask.append(z)
#     unique_spikes_mask = unique_mask[0]
#     diffs = list(compress(diffs, unique_spikes_mask))
#     return diffs


def compute_acg(dp, unit, cbin=0.2, cwin=80):
    ACG = acg(dp, unit, bin_size=cbin, win_size=cwin, subset_selection='all', normalize='Hertz')
    x = np.linspace(-cwin * 1. / 2, cwin * 1. / 2, ACG.shape[0])
    y = ACG.copy()
    # acg25, acg35 = ACG[:int(len(ACG) * 2. / 5)], ACG[int(len(ACG) * 3. / 5):]
    # acg_std = np.std(np.append(acg25, acg35))
    # acg_mn = np.mean(np.append(acg25, acg35))
    ylim1 = 0
    yl = max(ACG)
    ylim2 = int(yl) + 5 - (yl % 5)

    return x, y, ylim1, ylim2


def compute_isi(trn, quantile):
    diffs = np.diff(trn)
    isi_ = np.asarray(diffs, dtype='float64')
    isi_ = isi_[(isi_ >= np.quantile(isi_, quantile)) & (isi_ <= np.quantile(isi_, 1 - quantile))]
    return isi_


def range_normalization(x):
    max_x = np.max(x)
    min_x = np.min(x)
    range_x = max_x - min_x
    norm_x = 2 * (
                (x - min_x) / range_x) - 1
    return norm_x


def waveform_mean_subtraction(wvf):
    ms_wvf = (wvf - np.mean(wvf[0:10])) / np.max(wvf)
    return ms_wvf


def waveform_sigma(wvf):
    sigma = np.std(wvf[0:10])
    return sigma


def detect_peaks(wvf):
    detected_peaks = []
    xs = []
    ys = []

    for lk in list(range(5, 60, 5)):
        detected_peaks.append(peakdetect(wvf, lookahead=lk))

    detected_peaks = [x for x in detected_peaks[0] if x != []]
    detected_peaks = [item for items in detected_peaks for item in items]

    for peaks in detected_peaks:
        if (peaks[1] < -0.3) or (peaks[1] > 0.3):
            xs.append(peaks[0])
            ys.append(peaks[1])

    count_wvf_peaks = len(xs)
    return xs, ys, count_wvf_peaks


def detect_biggest_peak(wvf):

    max_pk = np.max(wvf)
    min_pk = np.abs(np.min(wvf))

    if min_pk > max_pk:
        biggest_peak_negative = 1
    else:
        biggest_peak_negative = 0

    return biggest_peak_negative


def cosine_similarity(wvf1, wvf2):
    dot_product = np.dot(wvf1, wvf2)
    norm_wavf1 = np.linalg.norm(wvf1)
    norm_wvf2 = np.linalg.norm(wvf2)
    cos_sim = dot_product / (norm_wavf1 * norm_wvf2)
    cos_sim = round(float(cos_sim), 2)
    cos_sim_t = 1 if cos_sim >= 0.6 else 0
    return cos_sim, cos_sim_t


def rvp_and_fp(isi, N, T, tauR=0.002, tauC=0.0005, fs=30000):

    # based on Hill et al., J Neuro, 2011
    # N = spikes_unit
    # T = 20 * 60 * 1000  # T: total experiment duration in milliseconds
    # tauR = 2  # tauR: refractory period >> 2 milliseconds
    # tauC = 0.5  # tauC: censored period >> 0.5 milliseconds

    # rpv = sum(isi <= tauR*fs/1000)  # Refractory period violations: in spikes
    rpv = sum(isi <= tauR)
    a = 2 * (tauR - tauC) * (N ** 2) / T  # In spikes >> r = 2*(tauR - tauC) * N^2 * (1-Fp) * Fp / T >> solve for Fp

    if rpv == 0:
        Fp = 0  # Fraction of contamination
        overestimate = 0
    else:
        rts = np.roots([-1, 1, -rpv/a])
        Fp = min(rts)  # r >> solve for Fp
        overestimate = 0
        if isinstance(Fp, complex):
            overestimate = 1
            if rpv < N:
                Fp = int(100 * round(rpv / (2 * (tauR - tauC) * (N - rpv)), 2)) # Integer
            else:
                Fp = 1

    return rpv, Fp


def mean_firing_rate(isi):
    MFR = 1000. / np.mean(isi)
    MFR = round(MFR, 2)
    return MFR


def mean_amplitude(samples):
    mean_amplitude = np.mean(samples)
    mean_amplitude = round(float(mean_amplitude), 2)
    return mean_amplitude


def remove_outliers(x, q):
    x = x[(x >= np.quantile(x, q)) & (x <= np.quantile(x, 1 - q))]
    x = x.reshape(len(x), )
    return x


# Auxiliary function to draw
Patch = matplotlib.patches.Patch
PosVal = Tuple[float, Tuple[float, float]]
Axis = matplotlib.axes.Axes
PosValFunc = Callable[[Patch], PosVal]

@dataclass
class AnnotateBars:
    font_size: int = 10
    color: str = "black"
    n_dec: int = 2

    def horizontal(self, ax: Axis, centered=False):
        def get_vals(p: Patch) -> PosVal:
            value = p.get_width()
            div = 2 if centered else 1
            pos = (
                p.get_x() + p.get_width() / div,
                p.get_y() + p.get_height() / 2,
            )
            return value, pos
        ha = "center" if centered else  "left"
        self._annotate(ax, get_vals, ha=ha, va="center")

    def vertical(self, ax: Axis, centered:bool=False):
        def get_vals(p: Patch) -> PosVal:
            value = p.get_height()
            div = 2 if centered else 1
            pos = (p.get_x() + p.get_width() / 2,
                   p.get_y() + p.get_height() / div
            )
            return value, pos
        va = "center" if centered else "bottom"
        self._annotate(ax, get_vals, ha="center", va=va)

    def _annotate(self, ax, func: PosValFunc, **kwargs):
        cfg = {"color": self.color,
               "fontsize": self.font_size, **kwargs}
        for p in ax.patches:
            if p.get_height() != 0:
                value, pos = func(p)
                ax.annotate(f"{value:.{self.n_dec}f}", pos, **cfg)


def nice_plot(plot_obj, x_label, y_label, title):
    plot_obj.spines["top"].set_visible(False)
    plot_obj.spines["right"].set_visible(False)
    plot_obj.set(xlabel=x_label, ylabel=y_label)
    plot_obj.spines['bottom'].set_color('#939799')
    plot_obj.spines['left'].set_color('#939799')
    plot_obj.yaxis.label.set_color('#939799')
    plot_obj.xaxis.label.set_color('#939799')
    plot_obj.tick_params(axis='x', colors='#939799')
    plot_obj.tick_params(axis='y', colors='#939799')
    plot_obj.tick_params(axis='both', which='minor', labelsize=9)
    plot_obj.tick_params(axis='both', which='major', labelsize=9)
    plot_obj.set_title(title, fontsize=9, fontname="DejaVu Sans", loc='center', color='#939799')
    return plot_obj


def format_plot(axes):
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.tick_params(axis='x', colors='#939799')
    axes.tick_params(axis='y', colors='#939799')
    axes.tick_params(axis='both', which='minor', labelsize=8)
    axes.tick_params(axis='both', which='major', labelsize=10)
    axes.yaxis.label.set_color('#939799')
    axes.xaxis.label.set_color('#939799')
    axes.spines['bottom'].set_color('#939799')
    axes.spines['left'].set_color('#939799')
    # axs[1, 2].tick_params(labelleft=False)
    # axs[1, 2].tick_params(labelbottom=False)
    return axes


def plot_units_composition(cell_list):
    n = 1

    sample = {}
    results = {}

    for dp in cell_list:

        year = dp[11:13]
        date = dp[12:20]
        cell = dp[8:11]

        unit_qualities = load_units_qualities(dp)
        total_units = len(unit_qualities)

        good_units = get_units(dp, quality='good')
        good_units = len(good_units)
        prop_good_units = round(good_units / total_units * 100, 2)

        mua_units = get_units(dp, quality='mua')
        mua_units = len(mua_units)
        prop_MUA_units = round(mua_units / total_units * 100, 2)

        noise_units = get_units(dp, quality='noise')
        noise_units = len(noise_units)
        prop_noise_units = round(noise_units / total_units * 100, 2)

        total_classified = good_units + mua_units + noise_units
        prop_classified_units = round(total_classified / total_units * 100, 2)
        total_unassigned = total_units - total_classified

        if total_unassigned != 0:
            prop_unassigned_units = round(total_unassigned / total_units * 100, 2)
        else:
            prop_unassigned_units = 0

        sample["Sample_{}".format(n)] = [cell, year, date, total_units, good_units, prop_good_units, mua_units,
                                         prop_MUA_units,
                                         noise_units, prop_noise_units, total_classified, prop_classified_units,
                                         total_unassigned, prop_unassigned_units]

        results.update(sample)
        n += 1

    import matplotlib.pyplot as plt
    from itertools import cycle, islice

    # Transform dict to pandas
    df = pd.DataFrame(results).T
    df = df.reset_index()
    df.columns = ['Sample', 'Cell-Type', 'Year', 'Date', 'Total', 'Good', '% Good', 'MUA', '% MUA',
                  'Noise', '% Noise', 'Classified', '% Classified', 'Unassigned', '% Unassigned']

    df['Date'] = pd.to_datetime(df['Date'], format='%y-%m-%d').dt.date
    df = df.sort_values(by='Date')

    # Define colors
    # my_colors = list(islice(cycle(['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#CFECF9', '#7F7F7F', '#BCBD22', '#17BECF']), None, len(df)))
    my_colors = list(islice(cycle(
        ['#4E79A7', '#F28E2C', '#E15759', '#76B7B2', '#59A14F', '#EDC949', '#AF7AA1', '#FF9DA7', '#9C755F', '#BAB0AB']),
                            None, len(df)))

    # Define one plane
    fig, axes = plt.subplots(1, 1)

    # Plot values
    gx = df.plot.bar(y=['% Good', '% MUA', '% Noise', '% Unassigned'], figsize=(15, 10), stacked=True, color=my_colors,
                     ax=axes)

    # Annotate proportions
    AnnotateBars(font_size=8, n_dec=1, color="white").vertical(axes, True)

    # Title
    gx.set_title('Proportion of units by their quality in {} samples'.format(cell), fontsize=20, fontname="DejaVu Sans",
                 pad=20, loc='left', color='#939799')

    # No frame
    gx.spines["top"].set_visible(False)
    gx.spines["right"].set_visible(False)

    # Axis and labels colors
    gx.legend(loc='best')
    gx.set(xlabel="Samples", ylabel="% of units per quality bucket")
    gx.set_xticklabels(df['Date'], rotation=25)
    gx.spines['bottom'].set_color('#939799')
    gx.spines['left'].set_color('#939799')
    gx.yaxis.label.set_color('#939799')
    gx.xaxis.label.set_color('#939799')
    gx.tick_params(axis='x', colors='#939799')
    gx.tick_params(axis='y', colors='#939799')
    gx.tick_params(axis='both', which='minor', labelsize=8)
    gx.tick_params(axis='both', which='major', labelsize=10)

    plt.show()

    return gx