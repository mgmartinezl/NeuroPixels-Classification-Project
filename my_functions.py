"""
Neuropixels probe recordings analysis

Auxiliary functions
Author: Gabiela Martinez

"""

# from rtn import npa
# from rtn.npix.gl import get_units, load_units_qualities
# from rtn.npix.io import ConcatenatedArrays, _pad, _range_from_slice, read_spikeglx_meta, chan_map
# from rtn.npix.spk_t import trn, isi, mfr
# from rtn.npix.corr import crosscorrelate_cyrille, ccg, acg
# from rtn.npix.spk_wvf import wvf, templates, get_peak_chan, get_depthSort_peakChans
# from rtn.npix.plot import plot_wvf, hist_MB, plot_raw, plot_raw_units, plot_acg, plot_ccg
# from rtn.npix.circuitProphyler import Prophyler, Dataset, Unit

from rtn.npix.corr import crosscorrelate_cyrille, ccg, acg
from rtn.utils import peakdetect
import numpy as np
import os.path as op;opj=op.join
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib
from typing import *
import scipy.optimize as opt
import scipy.stats as stats


def gaussian_cut(x, a, mu, sigma, xcut):
    g = a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    g[x < xcut] = 0
    return g


def curve_fit_(x, num, p1):
    popt = opt.curve_fit(gaussian_cut, x, num, p1, maxfev=10000)
    return popt


def ampli_fit_gaussian_cut(a):
    a = np.asarray(a, dtype='float64')
    num, bins = np.histogram(a, bins=80)
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


def waveform_mean_subtraction(norm_waveform_pk_ch):
    ms_norm_waveform_pk_ch = (norm_waveform_pk_ch - np.mean(norm_waveform_pk_ch[0:10])
                                     ) / np.max(norm_waveform_pk_ch)
    return ms_norm_waveform_pk_ch


def waveform_sigma(wvf):
    sigma = np.std(wvf[0:10])
    return sigma


def detect_peaks(waveform_pk_ch):
    detected_peaks = []
    xs = []
    ys = []

    s = waveform_sigma(waveform_pk_ch)

    for lk in list(range(5, 60, 5)):
        detected_peaks.append(peakdetect(waveform_pk_ch, lookahead=lk))

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


def cosine_similarity(wavf1, wvf2):
    dot_product = np.dot(wavf1, wvf2)
    norm_wavf1 = np.linalg.norm(wavf1)
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
        if isinstance(Fp, complex):  # function returns imaginary number if r is too high.
            overestimate = 1
            if rpv < N:
                Fp = round(rpv / (2 * (tauR - tauC) * (N - rpv)), 2)
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

