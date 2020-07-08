"""
Project: Neuropixels classification pipeline
Author: Gabriela Martinez
Script: AuxFunctions.py

"""

import matplotlib.pyplot as plt
from rtn import npa
from itertools import compress
from scipy import interpolate
from rtn.npix.gl import get_units, load_units_qualities
from rtn.npix.io import ConcatenatedArrays, _pad, _range_from_slice, read_spikeglx_meta, chan_map
from rtn.npix.spk_t import trn, isi, mfr
from rtn.npix.corr import crosscorrelate_cyrille, ccg, acg
from rtn.npix.spk_wvf import wvf, templates, get_peak_chan, get_depthSort_peakChans
from rtn.npix.plot import plot_wvf, hist_MB, plot_raw, plot_raw_units, plot_acg, plot_ccg
from rtn.npix.circuitProphyler import Prophyler, Dataset, Unit
import os
import scipy as sp
from scipy import ndimage


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


def find_neighbors(x):
    if (x - 1 >= 0) & (x + 1 <= 19):
        before_x = x - 1
        after_x = x + 1
        return before_x, after_x
    elif x - 1 < 0:
        before_x = x + 1
        after_x = x + 2
        return before_x, after_x
    elif x + 1 > 19:
        before_x = x - 2
        after_x = x - 1
        return before_x, after_x


def closest_waveforms(x1, current, x2, all, waveforms):

    if (x1 in all) & (x2 in all):

        before_neighbor_index = all.index(x1)
        deepest_chunk_index = all.index(current)
        after_neighbor_index = all.index(x2)

        before_neighbor_wvf = waveforms[before_neighbor_index]
        deepest_chunk_wvf = waveforms[deepest_chunk_index]
        after_neighbor_wvf = waveforms[after_neighbor_index]

        waves = []
        waves.append(before_neighbor_wvf)
        waves.append(deepest_chunk_wvf)
        waves.append(after_neighbor_wvf)

        chunks_for_wvf = [x1, current, x2]
        norm_wvf_block = np.mean(waves, axis=0)

        return norm_wvf_block, chunks_for_wvf

    elif (x1 in all) & (x2 not in all):

        if current == 19:
            deepest_chunk_index = all.index(current)
            deepest_chunk_wvf = waveforms[deepest_chunk_index]
            chunks_for_wvf = [current]
            return deepest_chunk_wvf, chunks_for_wvf

        else:
            before_neighbor_index = all.index(x1)
            deepest_chunk_index = all.index(current)
            before_neighbor_wvf = waveforms[before_neighbor_index]
            deepest_chunk_wvf = waveforms[deepest_chunk_index]

            waves = []
            waves.append(before_neighbor_wvf)
            waves.append(deepest_chunk_wvf)

            chunks_for_wvf = [x1, current]
            norm_wvf_block = np.mean(waves, axis=0)

            return norm_wvf_block, chunks_for_wvf

    elif (x1 not in all) & (x2 in all):

        if current == 0:

            deepest_chunk_index = all.index(current)
            deepest_chunk_wvf = waveforms[deepest_chunk_index]
            chunks_for_wvf = [current]

            return deepest_chunk_wvf, chunks_for_wvf

        else:
            deepest_chunk_index = all.index(current)
            after_neighbor_index = all.index(x2)
            deepest_chunk_wvf = waveforms[deepest_chunk_index]
            after_neighbor_wvf = waveforms[after_neighbor_index]

            waves = []
            waves.append(deepest_chunk_wvf)
            waves.append(after_neighbor_wvf)

            chunks_for_wvf = [current, x2]
            norm_wvf_block = np.mean(waves, axis=0)

            return norm_wvf_block, chunks_for_wvf

    elif (x1 not in all) & (x2 not in all):

        deepest_chunk_index = all.index(current)
        deepest_chunk_wvf = waveforms[deepest_chunk_index]
        chunks_for_wvf = [current]

        return deepest_chunk_wvf, chunks_for_wvf


def estimate_bins(x, rule):

    n = len(x)
    maxi = max(x)
    mini = min(x)

    # Freedman-Diaconis rule
    if rule == 'Fd':

        data = np.asarray(x, dtype=np.float_)
        iqr_ = iqr(data, scale="raw", nan_policy="omit")
        n = data.size
        bw = (2 * iqr_) / np.power(n, 1 / 3)
        datmin= min(data)
        datmax = max(data)
        datrng = datmax - datmin
        bins = int(datrng/bw + 1)

        # q75, q25 = np.percentile(x, [75, 25])
        # iqr_ = q75 - q25
        # print('iqr', iqr_)
        # h = 2 * iqr_ * (n ** (-1/3))
        # print('h', h)
        # b = int(round((maxi-mini)/h, 0))

        return bins

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


def gaussian_amp_est(x, n_bins):

    try:
        x1, p0 = ampli_fit_gaussian_cut(x, n_bins)
        n_fit = gaussian_cut(x1, a=p0[0], mu=p0[1], sigma=p0[2], x_cut=p0[3])
        min_amp = p0[3]
        n_fit_no_cut = gaussian_cut(x1, a=p0[0], mu=p0[1], sigma=p0[2], x_cut=0)
        percent_missing = int(round(100 * stats.norm.cdf((min_amp - p0[1]) / p0[2]), 0))

    except RuntimeError:
        x1, p0, min_amp, n_fit, n_fit_no_cut, percent_missing = None, None, None, None, None, np.nan

    return x1, p0, min_amp, n_fit, n_fit_no_cut, percent_missing


def gaussian_cut(x, a, mu, sigma, x_cut):
    g = a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    g[x < x_cut] = 0
    return g


# def log_fit(x, a, mu, sigma):
#     return a / x * 1. / (sigma * np.sqrt(2. * np.pi)) * np.exp(-(np.log(x) - mu)**2 / (2. * sigma**2))
#
#
# def isi_fit_log(x, n_bins):
#     a = np.asarray(x, dtype='float64')
#     num, bins = np.histogram(a, bins=n_bins)
#     mode = np.argmax(num)
#     yM = num[mode]
#     xM = bins[mode]
#     xR = bins / xM
#     yR = num / yM
#     sol, err = opt.curve_fit(log_fit, xR, yR, maxfev=10000)
#     scaledSol = [yM * sol[0] * xM, sol[1] + np.log(xM), sol[2]]
#     yF = np.fromiter((log_fit(xx, *sol) for xx in xR), np.float)
#     yFIR = np.fromiter((log_fit(xx, *scaledSol) for xx in x), np.float)


def curve_fit_(x, num, p1):
    pop_t = opt.curve_fit(gaussian_cut, x, num, p1, maxfev=10000)
    return pop_t


def exponential_fit(x, a, r):
    return a*np.exp(r*x)


def curve_fit_exp(x, y, p):
    pop_t = opt.curve_fit(exponential_fit, x, y, p, maxfev=10000)
    return pop_t


def ampli_fit_gaussian_cut(x, n_bins):
    a = np.asarray(x, dtype='float64')
    num, bins = np.histogram(a, bins=n_bins)
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
    ylim1 = 0
    yl = max(ACG)
    ylim2 = int(yl) + 5 - (yl % 5)

    return x, y, ylim1, ylim2


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


def delete_routines(command, dir):

    if command is True:
        for filename in os.listdir(dir):
            file_path = os.path.join(dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    else:
        pass


def detect_peaks(wvf, outliers_dev):
    detected_peaks = []
    xs = []
    ys = []
    count_wvf_peaks = 0

    for lk in list(range(3, 60, 3)):
        detected_peaks.append(peakdetect(wvf, lookahead=lk))

    detected_peaks = [x for x in detected_peaks[0] if x != []]
    detected_peaks = [item for items in detected_peaks for item in items]

    iqr_ = iqr(wvf) * outliers_dev
    _med = np.median(wvf)
    lower = _med - iqr_
    upper = _med + iqr_

    for peaks in detected_peaks:
        if (peaks[1] >= upper) | (peaks[1] <= lower):
            xs.append(peaks[0])
            ys.append(peaks[1])

    # Add max positive peak
    if np.amax(wvf) not in ys:
        ys.append(np.amax(wvf))
        xs.append(np.where(wvf == np.amax(wvf))[0][0])

    bool_x = [(i >= 31) and (i <= 50) for i in xs]
    ys_center = list(compress(ys, bool_x))
    count_wvf_center_peaks = len(ys_center)
    count_wvf_peaks = len(ys)

    return xs, ys, count_wvf_peaks, count_wvf_center_peaks


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
    norm_wvf1 = np.linalg.norm(wvf1)
    norm_wvf2 = np.linalg.norm(wvf2)
    cos_sim = dot_product / (norm_wvf1 * norm_wvf2)
    cos_sim = round(float(cos_sim), 2)
    return cos_sim


def rvp_and_fp(isi, N, T, taur, tauc):

    # based on Hill et al., J Neuro, 2011
    # N = spikes_unit
    # T = 20 * 60 * 1000  # T: total experiment duration in milliseconds
    # taur = 1.5  # taur: refractory period >> 1.5 milliseconds
    # tauc = 0.5  # tauc: censored period >> 0.5 milliseconds

    # rpv = sum(isi <= taur)
    # rpv = np.count_nonzero(np.diff(trn(dp, unit, subset_selection=[(0, 20*60)]))/30 < 2)
    rpv = np.count_nonzero(isi <= taur)
    a = T / (2 * (taur - tauc) * (N ** 2))
    b = rpv*a

    if rpv == 0:
        fp = 0
    else:
        rts = np.roots([1, -1, b])
        rts = rts[~np.iscomplex(rts)]
        if rts.size != 0:
            fp = round(min(rts), 2)

        else:
            if rpv < N:
                # This happens when constant b is greater than 0.25 >> lots of rpv wrt the total spikes!!!
                # This means the rpv is huge wrt to N
                # If we had more spikes in the unit/block, fp would be lower
                # (b**2) - (4*a*c) < 0 -- range in which the equation does not have real roots!
                # b = 0.25
                # rts = np.roots([1, -1, b])
                # rts = rts[~np.iscomplex(rts)]
                # fp = round(min(rts), 2)
                fp = np.nan
            else:
                fp = 1
    return rpv, fp


def compute_entropy_dorval(isint):

    """
    Dorval2007:
    Using logqrithmic ISIs i.e. ISIs binned in bISI(k) =ISI0 * 10**k/κ with k=1:Klog.
    Classical entropy estimation from Shannon & Weaver, 1949.
    Spike entropy characterizes the regularity of firing (the higher the less regular)
    """

    ## Compute entropy as in Dorval et al., 2009
    # 1) Pisi is the logscaled discrete density of the ISIs for a given unit
    # (density = normalized histogram)
    # right hand side of k_th bin is bISI(k) =ISI0 * 10**k/κ with k=1:Klog
    # where ISI0 is smaller than the smallest ISI -> 0.01
    # and Klog is picked such that bISI(Klog) is larger than the largest ISI -> 300 so 350
    # K is the number of bins per ISI decade.

    # Entropy can be thought as a measurement of the sharpness of the histogram peaks,
    # which is directly related with a better defined structural information

    ISI0 = 0.1
    Klog = 350
    K = 200

    try:
        # binsLog = ISI0 * 10 ** (np.arange(1, Klog + 1, 1) * 1. / K)
        binsLog = 200
        num, bins = np.histogram(isint, binsLog)
        histy, histx = num * 1. / np.sum(num), bins[1:]
        sigma = (1. / 6) * np.std(histy)
        Pisi = ndimage.gaussian_filter1d(histy, sigma)

    except ValueError:
        binsLog = 200
        num, bins = np.histogram(isint, binsLog)
        histy, histx = num * 1. / np.sum(num), bins[1:]
        sigma = (1. / 6) * np.std(histy)
        Pisi = ndimage.gaussian_filter1d(histy, sigma)

    # Remove 0 values
    non0vals = (Pisi > 0)
    Pisi = Pisi[non0vals]

    entropy = 0

    for i in range(len(Pisi)):
        entropy += -Pisi[i] * np.log2(Pisi[i])

    return entropy


def mean_firing_rate(isi):
    mfr = 1. / np.mean(isi)  # rate='Hz'
    mfr = round(mfr, 2)
    return mfr


def compute_isi(train, *args, **kwargs):

    """This function returns the isi in ms! """

    quantile = kwargs.get('quantile', None)
    # train = trn * 1. / (30000 * 1. / 1000)  # From samples to ms
    # diffs = np.diff(trn)/30000
    diffs = np.diff(train)
    isi_ = np.asarray(diffs, dtype='float64')
    if quantile:
        isi_ = isi_[(isi_ >= np.quantile(isi_, quantile)) & (isi_ <= np.quantile(isi_, 1 - quantile))]
        return isi_/(30000)
    else:
        return isi_


def compute_isi_features(isint):

    # This assumes that ISI is already in ms

    #isint = isint[isint != 0]
    isint_s = isint * 1. / 1000  # Transform isi from ms to seconds

    # Transform the isi into a histogram
    # Pi = np.histogram(isint, bins=np.arange(0, 100, 0.1))

    # Firing pattern features

    # Mean Instantaneous Firing Rate. Why in seconds?
    # Instantaneous frequencies were calculated for each interspike interval as the reciprocal of the isi;
    # mean instantaneous frequency as the arithmetic mean of all values.
    # MIFR = np.mean(1./isint_s)
    mifr = round(float(np.mean(1./isint)), 3)

    # Median inter-spike interval distribution
    # medISI = np.median(isint)
    med_isi = round(float(np.median(isint)), 3)

    # Mode of inter-spike interval distribution
    # modeISI = Pi[1][:-1][Pi[0] == np.max(Pi[0])][0]
    num, bins = np.histogram(isint, bins=np.arange(0, 100, 0.1))
    mode_isi = np.argmax(num)

    # Burstiness of firing: 5th percentile of inter-spike interval distribution
    prct5ISI = round(np.percentile(isint, 5), 3)

    # Entropy of inter-spike interval distribution
    entropyD = round(compute_entropy_dorval(isint), 3)

    # Average coefficient of variation for a sequence of 2 ISIs
    # Relative difference of adjacent ISIs
    CV2_mean = round(float(np.mean(2 * np.abs(isint[1:] - isint[:-1]) / (isint[1:] + isint[:-1]))), 3)
    CV2_median = round(float(np.median(2 * np.abs(isint[1:] - isint[:-1]) / (isint[1:] + isint[:-1]))), 3)  # (Holt et al., 1996)

    # Coefficient of variation
    # Checked!
    CV = round(np.std(isint) / np.mean(isint), 3)

    # Instantaneous irregularity >> equivalent to the difference of the log ISIs
    # Checked!
    IR = round(float(np.mean(np.abs(np.log(isint[1:] / isint[:-1])))), 3)

    # # Local Variation
    # Checked!
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2701610/pdf/pcbi.1000433.pdf
    Lv = round(3 * np.mean(np.ones((len(isint) - 1)) - (4 * isint[:-1] * isint[1:]) / ((isint[:-1] + isint[1:]) ** 2)), 3)

    # Revised Local Variation, with R the refractory period in the same unit as isint
    # Checked!
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2701610/pdf/pcbi.1000433.pdf
    R = 0.8  # ms
    LvR = 3 * np.mean((np.ones((len(isint) - 1)) - (4 * isint[:-1] * isint[1:]) / ((isint[:-1] + isint[1:]) ** 2)) *
                      (np.ones((len(isint) - 1)) + (4 * R / (isint[:-1] + isint[1:]))))
    LvR = round(LvR, 3)

    # Coefficient of variation of the log ISIs
    # Checked!
    LcV = round(np.std(np.log10(isint)) * 1. / np.mean(np.log10(isint)), 3)

    # Geometric average of the rescaled cross correlation of ISIs
    # Checked!
    SI = round(-np.mean(0.5 * np.log10((4 * isint[:-1] * isint[1:]) / ((isint[:-1] + isint[1:]) ** 2))), 3)

    # Skewness of the inter-spikes intervals distribution
    # Checked!
    SKW = round(float(sp.stats.skew(isint)), 3)

    # Entropy not included
    return mifr, med_isi, mode_isi, prct5ISI, entropyD, CV2_mean, CV2_median, CV, IR, Lv, LvR, LcV, SI, SKW


def spline_interpolation(wvf, x_axis, y):

    y_reduced = np.array(wvf) - y
    first_root = interpolate.UnivariateSpline(x_axis, y_reduced).roots()[0]
    second_root = interpolate.UnivariateSpline(x_axis, y_reduced).roots()[1]
    x_diff = np.abs(second_root - first_root)

    return first_root, second_root, x_diff


def compute_waveforms_features(wave, peaks_x_center, peaks_y_center, fs=30000, ampliFactor=500):

    # # Add min negative peak
    # if np.amin(wave) not in peaks_y_center:
    #     peaks_y_center.append(np.amin(wave))
    #     peaks_x_center.append(np.where(wvf == np.amin(wave))[0][0])

    bool_x = [(i >= 31) and (i <= 50) for i in peaks_x_center]
    peaks_x_center = list(compress(peaks_x_center, bool_x))
    peaks_y_center = list(compress(peaks_y_center, bool_x))

    pk_neg_amplitude = np.min(peaks_y_center)
    pk_pos_amplitude = np.max(peaks_y_center)


def mean_amplitude(samples):
    mean_amplitude = np.mean(samples)
    mean_amplitude = round(float(mean_amplitude), 2)
    return mean_amplitude


def remove_outliers(x, q):
    x = x[(x >= np.quantile(x, q)) & (x <= np.quantile(x, 1 - q))]
    x = x.reshape(len(x), )
    return x


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
    # return array[min(range(len(array)), key=lambda i: abs(array[i] - value))]

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