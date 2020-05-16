"""
Project: Neuropixels classification pipeline
Author: Gabriela Martinez
Script: QualityCheckPipeline.py
Description: implementation of quality metrics on units and chunks to choose the right ones for the next stage in the
classification pipeline.

"""

from AuxFunctions import *
import pandas as pd
from operator import itemgetter
from rtn.npix.spk_t import trn
from scipy import interpolate
import os
import re
import ast


np.seterr(all='ignore')

data_sets = [
    'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1'
    #'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1'
]

# Parameters
fs = 30000
exclusion_quantile = 0.02
unit_size_s = 20 * 60
unit_size_ms = unit_size_s * 1000
chunk_size_s = 60
chunk_size_ms = 60 * 1000
samples_fr = unit_size_s * fs
n_chunks = int(unit_size_s / chunk_size_s)
n_waveforms = 400
ignore_nwvf = True
t_waveforms = 120
again = False

for dp in data_sets:

    sample_probe = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}_\w{5}\d{1}", dp)[0]
    sample = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}", dp)[0]
    print('***************************')
    print('Sample:', sample_probe)
    cell_type = dp[8:11]

    # Load kilosort aux files
    amplitudes_sample = np.load(f'{dp}//amplitudes.npy')  # shape N_tot_spikes x 1
    spike_times = np.load(f'{dp}//spike_times.npy')  # in samples
    spike_clusters = np.load(f'{dp}//spike_clusters.npy')

    # Extract good units of current sample
    all_good_units = pd.read_csv('Images/Pipeline/GoodUnitsAllRecordings.csv')
    good_units = ast.literal_eval(all_good_units[all_good_units['Sample'].str.match(sample_probe)]['Units'].values[0])

    # Features path
    features_path = f'Images/Pipeline/Sample_{sample_probe}/Features'

    print(f"Quality units found in current sample: {len(good_units)} --> {good_units}")

    # Extract chunks for good units
    units_chunks = pd.read_csv(f'Images/Pipeline/Sample_{sample_probe}/Sample-{sample_probe}-ChunksUnits.csv')

    for unit in good_units:

        # Create alternative dir for routines
        if not os.path.exists(features_path):
            os.makedirs(features_path)

        print(f'Unit >>>> {unit}')

        # Extract chunks
        chunks = list(map(int, eval(units_chunks[units_chunks['Unit'] == unit]['GoodChunks'][
            units_chunks[units_chunks['Unit'] == unit].index.values.astype(int)[0]])))

        chunks_wvf = list(map(int, eval(units_chunks[units_chunks['Unit'] == unit]['WvfChunks'][
            units_chunks[units_chunks['Unit'] == unit].index.values.astype(int)[0]])))

        # Store quality metrics
        unit_quality_metrics = pd.read_csv(f'Images/Pipeline/Sample_{sample_probe}/GoodUnits/Files/Sample-{sample}-unit-{unit}.csv')

        # Retrieve peak channel for this unit
        peak_channel = unit_quality_metrics['PeakChUnit'][0]

        # Collect for later...
        l_chunk_trns = []
        l_chunk_wvfs = []

        # Read chunks...
        for i in range(n_chunks):
            if i in chunks and i not in chunks_wvf:

                # These chunks will be used to compute temporal features
                # Chunk length and times in seconds
                chunk_start_time = i * chunk_size_s
                chunk_end_time = (i + 1) * chunk_size_s

                # Chunk spikes
                trn_samples_chunk = trn(dp, unit=unit,
                                        subset_selection=[(chunk_start_time, chunk_end_time)],
                                        enforced_rp=0.5)

                l_chunk_trns.append(trn_samples_chunk)

                # Compute Inter-Spike Interval for this chunk
                isi_chunk_clipped = compute_isi(trn_samples_chunk, quantile=exclusion_quantile)
                isi_chunk_no_clipped = compute_isi(trn_samples_chunk)

                # Estimate number of optimal bins for ISI
                isi_bins = estimate_bins(isi_chunk_no_clipped, rule='Sqrt')

                isi_log_bins = None

                try:
                    isi_log_bins = np.geomspace(isi_chunk_no_clipped.min(), isi_chunk_no_clipped.max(), isi_bins)
                except ValueError:
                    pass

                # Compute temporal features for every chunk
                # Check if maybe isi_log_bins is a better option
                mfr = mean_firing_rate(isi_chunk_clipped)
                mifr, med_isi, mode_isi, prct5ISI, entropy, CV2_mean, CV2_median, CV, IR, Lv, LvR, LcV, SI, skw \
                    = compute_isi_features(isi_chunk_clipped, isi_bins)

            else:
                continue

        for i in range(n_chunks):
            if i in chunks_wvf:

                # These chunks will be used to compute waveform-based features. They will be maximum 3
                # Chunk length and times in seconds
                chunk_start_time = i * chunk_size_s
                chunk_end_time = (i + 1) * chunk_size_s

                # Chunk spikes
                trn_samples_chunk = trn(dp, unit=unit,
                                        subset_selection=[(chunk_start_time, chunk_end_time)],
                                        enforced_rp=0.5)

                l_chunk_trns.append(trn_samples_chunk)

                # Compute Inter-Spike Interval for this chunk
                isi_chunk_clipped = compute_isi(trn_samples_chunk, quantile=exclusion_quantile)
                isi_chunk_no_clipped = compute_isi(trn_samples_chunk)

                # Estimate number of optimal bins for ISI
                isi_bins = estimate_bins(isi_chunk_no_clipped, rule='Sqrt')

                isi_log_bins = None

                try:
                    isi_log_bins = np.geomspace(isi_chunk_no_clipped.min(), isi_chunk_no_clipped.max(), isi_bins)
                except ValueError:
                    pass

                # Compute temporal features for every chunk
                mfr = mean_firing_rate(isi_chunk_clipped)
                mifr, med_isi, mode_isi, prct5ISI, entropy, CV2_mean, CV2_median, CV, IR, Lv, LvR, LcV, SI, skw \
                    = compute_isi_features(isi_chunk_clipped, isi_bins)

                # Chunk mean waveform around peak channel
                chunk_wvf = np.mean(
                    wvf(dp, unit,
                        n_waveforms=n_waveforms,
                        t_waveforms=t_waveforms,
                        subset_selection=[(chunk_start_time, chunk_end_time)],
                        again=again,
                        ignore_nwvf=ignore_nwvf)
                    [:, :, peak_channel], axis=0)
                l_chunk_wvfs.append(chunk_wvf)

                # Normalize mean waveform
                chunk_wvf_n = waveform_mean_subtraction(chunk_wvf)

                # Compute the baseline
                baseline = np.mean(chunk_wvf[0:40])

                # Make pairwise list of tuples with samples and wave position
                # x_samples = np.arange(0, chunk_wvf.shape[0])  # From 0 to 119, in samples. Maybe it is better to put in ms?
                x_samples = list(np.arange(len(chunk_wvf), step=1))
                y_samples = chunk_wvf.copy()
                x = [round(i / 30, 2) for i in x_samples]  # in ms
                points = list(zip(x, chunk_wvf))

                # Find peak amplitudes
                # max_amp = np.max(chunk_wvf) - baseline  # Ask about this
                # min_amp = np.min(chunk_wvf) - baseline
                pos_amp = max(points, key=itemgetter(1))[1]
                pos_amp_time = max(points, key=itemgetter(1))[0]
                neg_amp = min(points, key=itemgetter(1))[1]
                neg_amp_time = min(points, key=itemgetter(1))[0]

                # # Find cut points >> 10%, 50%, 90%
                pos_amp_5 = pos_amp * 0.05
                # pos_amp_5_time = interpolate.UnivariateSpline(x[x.index(neg_amp_time):x.index(pos_amp_time)], (
                #         np.array(chunk_wvf[x.index(neg_amp_time):x.index(pos_amp_time)]) - pos_amp_5)).roots()[0]
                #
                pos_amp_10 = pos_amp * 0.1
                # pos_rise_time = interpolate.UnivariateSpline(x[x.index(pos_amp_5_time):x.index(pos_amp_time)], (
                #                   np.array(chunk_wvf[x.index(pos_amp_5_time):x.index(pos_amp_time)]) - pos_amp_10)).roots()[0]
                # pos_fall_time = interpolate.UnivariateSpline(x[x.index(pos_amp_time):], (
                #                 np.array(chunk_wvf[x.index(pos_amp_time):]) - pos_amp_10)).roots()[0]


                pos_amp_50 = pos_amp * 0.5
                pos_amp_90 = pos_amp * 0.9
                neg_amp_10 = neg_amp * 0.1
                neg_amp_5 = neg_amp * 0.05
                neg_amp_50 = neg_amp * 0.5
                neg_amp_90 = neg_amp * 0.9

                # Interpolation of cut points >> THIS DOES NOT WORK
                # pos_rise_time, pos_fall_time, posPk_duration = spline_interpolation(chunk_wvf, x, pos_amp_10)
                # pos_rise_hw, pos_fall_hw, posHw_duration = spline_interpolation(chunk_wvf, x, pos_amp_50)
                # pos_risePk_time, pos_fallPk_time, posPkClimax_duration = spline_interpolation(chunk_wvf, x, pos_amp_90)

                # For negative peak, compute cutoff times and durations >> THIS DOES NOT WORK
                # neg_fall_time, neg_rise_time, negPk_duration = spline_interpolation(chunk_wvf, x, neg_amp_10)
                # neg_fall_hw, neg_rise_hw, negHw_duration = spline_interpolation(chunk_wvf, x, neg_amp_50)
                # neg_fallPk_time, neg_risePk_time, negPkClimax_duration = spline_interpolation(chunk_wvf, x, neg_amp_90)

                # A couple of features depend on the order of the peaks in the waveform
                # Check if the first peak is positive
                if pos_amp_time < neg_amp_time:
                    neg_peak_first = 0
                else:
                    neg_peak_first = 1

                # Compute onset time
                if neg_peak_first == 1:
                    onset = neg_amp * 0.05
                    end = pos_amp * 0.05
                    f1 = interpolate.interp1d(x[x.index(pos_amp_time):], chunk_wvf[x.index(pos_amp_time):])
                    onset_time = interpolate.UnivariateSpline(
                        x[x.index(neg_amp_time)-15:x.index(neg_amp_time)],
                        (np.array(chunk_wvf[x.index(neg_amp_time)-15:x.index(neg_amp_time)]) - onset)).roots()[-1]

                    end_time = interpolate.UnivariateSpline(x[x.index(pos_amp_time):], (
                            np.array(chunk_wvf[x.index(pos_amp_time):]) - end)).roots()[0]
                    duration = round(pos_amp_time - neg_amp_time, 3)

                    # For the exponential fit
                    x_exp = x[x.index(pos_amp_time):]
                    y_exp = chunk_wvf[x.index(pos_amp_time):]

                else:
                    onset = pos_amp * 0.05
                    end = neg_amp * 0.05
                    f1 = interpolate.interp1d(x, chunk_wvf)
                    onset_time = interpolate.UnivariateSpline(
                        x[x.index(neg_amp_time) - 15:x.index(pos_amp_time)],
                        (np.array(chunk_wvf[x.index(pos_amp_time) - 15:x.index(pos_amp_time)]) - onset)).roots()[0]
                    end_time = interpolate.UnivariateSpline(x[x.index(neg_amp_time):], (
                                np.array(chunk_wvf[x.index(neg_amp_time):]) - end)).roots()[0]
                    duration = round(neg_amp_time - pos_amp_time, 3)

                    # For the exponential fit
                    x_exp = x[x.index(pos_amp_time):]
                    y_exp = chunk_wvf[x.index(pos_amp_time):]

                # Compute first peak end time (when the negative or the positive peak ends)
                x1 = x[x.index(neg_amp_time):x.index(pos_amp_time)]
                y = chunk_wvf[x.index(neg_amp_time):x.index(pos_amp_time)]
                f = interpolate.interp1d(x1, y)
                zero_crossing_time = interpolate.UnivariateSpline(x1, (np.array(y) - 0)).roots()[0]
                zero_crossing = f(zero_crossing_time)

                # RECOMPUTE HALF-WIDTH
                # pos_rise_time, pos_fall_time, posPk_duration = spline_interpolation(chunk_wvf, x, pos_amp_10)
                # neg_fall_time, neg_rise_time, negPk_duration = spline_interpolation(chunk_wvf, x, neg_amp_10)
                if neg_peak_first == 1:

                    # Positive 10%
                    pos_rise_time = interpolate.UnivariateSpline(
                                  x[x.index(neg_amp_time):x.index(pos_amp_time)], (
                                  np.array(chunk_wvf[x.index(neg_amp_time):x.index(pos_amp_time)]) - pos_amp_10)).roots()[0]

                    pos_fall_time = interpolate.UnivariateSpline(
                        x[x.index(pos_amp_time):], (np.array(chunk_wvf[x.index(pos_amp_time):]) - pos_amp_10)).roots()[0]

                    posPk_duration = pos_fall_time - pos_rise_time

                    # Negative 10%
                    neg_fall_time = interpolate.UnivariateSpline(
                                  x[:x.index(neg_amp_time)], (
                                  np.array(chunk_wvf)[:x.index(neg_amp_time)] - neg_amp_10)).roots()[-1]

                    neg_rise_time = interpolate.UnivariateSpline(
                                  x[x.index(neg_amp_time):x.index(pos_amp_time)], (
                                  np.array(chunk_wvf[x.index(neg_amp_time):x.index(pos_amp_time)]) - neg_amp_10)).roots()[0]

                    negPk_duration = neg_rise_time - neg_fall_time

                    # Positive 50%
                    pos_rise_hw = interpolate.UnivariateSpline(
                                  x[x.index(neg_amp_time):x.index(pos_amp_time)], (
                                  np.array(chunk_wvf[x.index(neg_amp_time):x.index(pos_amp_time)]) - pos_amp_50)).roots()[0]

                    pos_fall_hw = interpolate.UnivariateSpline(
                        x[x.index(pos_amp_time):], (np.array(chunk_wvf[x.index(pos_amp_time):]) - pos_amp_50)).roots()[0]

                    posHw_duration = pos_fall_hw - pos_rise_hw

                    # Negative 50%
                    neg_fall_hw = interpolate.UnivariateSpline(
                                  x[:x.index(neg_amp_time)],
                                  (np.array(chunk_wvf[:x.index(neg_amp_time)]) - neg_amp_50)).roots()[-1]

                    neg_rise_hw = interpolate.UnivariateSpline(
                                  x[x.index(neg_amp_time):x.index(pos_amp_time)],
                                  (np.array(chunk_wvf[x.index(neg_amp_time):x.index(pos_amp_time)]) - neg_amp_50)).roots()[0]

                    negHw_duration = neg_rise_hw - neg_fall_hw

                    # Positive 90%
                    pos_risePk_time = interpolate.UnivariateSpline(
                        x[x.index(neg_amp_time):], (np.array(chunk_wvf[x.index(neg_amp_time):]) - pos_amp_90)).roots()[0]

                    pos_fallPk_time = interpolate.UnivariateSpline(
                        x[x.index(neg_amp_time):], (np.array(chunk_wvf[x.index(neg_amp_time):]) - pos_amp_90)).roots()[1]

                    posPkClimax_duration = pos_fallPk_time - pos_risePk_time

                    # Negative 90%
                    neg_fallPk_time = interpolate.UnivariateSpline(
                        x, (np.array(chunk_wvf) - neg_amp_90)).roots()[-1]

                    neg_risePk_time = interpolate.UnivariateSpline(
                        x, (np.array(chunk_wvf) - neg_amp_90)).roots()[0]

                    negPkClimax_duration = neg_risePk_time - neg_fallPk_time

                    # Compute fall slope for positive peak
                    neg_fall_slope = np.abs(neg_amp_90 - neg_amp_10)
                    neg_fall_slope_time = np.abs(neg_fallPk_time - neg_fall_time)

                    # Negative decay time: spike duration from negative peak to 10% of negative peak.
                    neg_decay_time = neg_rise_time - neg_amp_time

                else:
                    # TODO
                    # Verify the 90's

                    # Positive 10%
                    pos_rise_time = interpolate.UnivariateSpline(
                        x[x.index(pos_amp_time):], (np.array(chunk_wvf[:x.index(pos_amp_time)]) - pos_amp_10)).roots()[0]

                    pos_fall_time = interpolate.UnivariateSpline(
                        x[x.index(pos_amp_time):x.index(neg_amp_time)],
                        (np.array(chunk_wvf[x.index(pos_amp_time):x.index(neg_amp_time)]) - pos_amp_10)).roots()[0]

                    posPk_duration = pos_fall_time - pos_rise_time

                    # Negative 10%
                    neg_fall_time = interpolate.UnivariateSpline(
                        x[x.index(pos_amp_time):x.index(neg_amp_time)], (
                        np.array(chunk_wvf)[x.index(pos_amp_time):x.index(neg_amp_time)] - neg_amp_10)).roots()[0]

                    neg_rise_time = interpolate.UnivariateSpline(
                        x[x.index(neg_amp_time):], (
                        np.array(chunk_wvf[x.index(neg_amp_time):]) - neg_amp_10)).roots()[0]

                    negPk_duration = neg_rise_time - neg_fall_time

                    # Positive 50%
                    pos_rise_hw = interpolate.UnivariateSpline(
                        x[x.index(pos_amp_time):],
                        (np.array(chunk_wvf[:x.index(pos_amp_time)]) - pos_amp_50)).roots()[0]

                    pos_fall_hw = interpolate.UnivariateSpline(
                        x[x.index(pos_amp_time):x.index(neg_amp_time)],
                        (np.array(chunk_wvf[x.index(pos_amp_time):x.index(neg_amp_time)]) - pos_amp_50)).roots()[0]

                    posHw_duration = pos_fall_hw - pos_rise_hw

                    # Negative 50%
                    neg_fall_hw = interpolate.UnivariateSpline(
                        x[x.index(pos_amp_time):x.index(neg_amp_time)], (
                        np.array(chunk_wvf)[x.index(pos_amp_time):x.index(neg_amp_time)] - neg_amp_50)).roots()[0]

                    neg_rise_hw = interpolate.UnivariateSpline(
                        x[x.index(neg_amp_time):], (
                        np.array(chunk_wvf[x.index(neg_amp_time):]) - neg_amp_50)).roots()[0]

                    negHw_duration = neg_rise_hw - neg_fall_hw

                    # Positive 90%
                    pos_risePk_time = interpolate.UnivariateSpline(
                        x[x.index(pos_amp_time):],
                        (np.array(chunk_wvf[:x.index(pos_amp_time)]) - pos_amp_90)).roots()[-1]

                    pos_fallPk_time = interpolate.UnivariateSpline(
                        x[x.index(pos_amp_time):x.index(neg_amp_time)],
                        (np.array(chunk_wvf[x.index(pos_amp_time):x.index(neg_amp_time)]) - pos_amp_90)).roots()[0]

                    posPkClimax_duration = pos_fallPk_time - pos_risePk_time

                    # Negative 90%
                    neg_fallPk_time = interpolate.UnivariateSpline(
                        x, (np.array(chunk_wvf) - neg_amp_90)).roots()[-1]

                    neg_risePk_time = interpolate.UnivariateSpline(
                        x, (np.array(chunk_wvf) - neg_amp_90)).roots()[0]

                    negPkClimax_duration = neg_risePk_time - neg_fallPk_time

                    # Compute fall slope for positive peak
                    neg_fall_slope = np.abs(neg_amp_90 - neg_amp_10)
                    neg_fall_slope_time = np.abs(neg_fallPk_time - neg_fall_time)

                    # Negative decay time: spike duration from negative peak to 10% of negative peak.
                    neg_decay_time = neg_rise_time - neg_amp_time

                # Compute rise slope for positive peak
                pos_rise_slope = np.abs(pos_amp_90 - pos_amp_10)
                pos_rise_slope_time = pos_risePk_time - pos_rise_time

                # Positive decay time: spike duration from negative peak to 10% of negative peak.
                pos_decay_time = pos_fall_time - pos_amp_time

                # Compute peak-trough ratio
                # Distance from peak to baseline
                # peak_baseline = np.abs(pos_amp - baseline)
                # trough_baseline = np.abs(neg_amp - baseline)
                # peak_trough_ratio = round(peak_baseline/trough_baseline, 3)  # Relative to baseline
                peak_trough_ratio = round(np.abs(pos_amp / neg_amp), 3)  # Relative to zero (Jia et al)

                # Re-polarization slope
                # Fit a linear regression to the points between the trough and the peak
                xs = x[x.index(neg_amp_time):x.index(pos_amp_time)]
                ys = chunk_wvf[x.index(neg_amp_time):x.index(pos_amp_time)]
                rep_slope, rep_intercept, r_value, p_value, std_error = stats.linregress(xs, ys)
                # l = np.array([list(a) for a in zip(xs, ys)])
                abline_values = [rep_slope * i + rep_intercept for i in xs]
                last = abline_values[-2]
                first = abline_values[1]

                # Recovery slope
                if neg_peak_first == 1:
                    x_end_sl = x[x.index(pos_amp_time):x.index(pos_amp_time)+10]
                    y_end_sl = chunk_wvf[x.index(pos_amp_time):x.index(pos_amp_time)+10]
                    rec_slope, rec_intercept, r_value1, p_value1, std_error1 = stats.linregress(x_end_sl, y_end_sl)
                    abline_values1 = [rec_slope * i + rec_intercept for i in x_end_sl]
                    last1 = abline_values1[-2]
                    first1 = abline_values1[1]
                else:
                    x_end_sl = x[x.index(neg_amp_time):x.index(pos_amp_time)+10]
                    y_end_sl = chunk_wvf[x.index(neg_amp_time):x.index(pos_amp_time)+10]
                    rec_slope, rec_intercept, r_value1, p_value1, std_error1 = stats.linregress(x_end_sl, y_end_sl)
                    abline_values1 = [rec_slope * i + rec_intercept for i in x_end_sl]
                    last1 = abline_values1[-2]
                    first1 = abline_values1[1]

                # Fit exponential to ending slope
                popt, pcov = opt.curve_fit(exponential_fit, x_exp, y_exp, p0=(0, 0))
                x_fit = np.linspace(min(x_exp), max(x_exp), 25)
                y_fit = exponential_fit(x_fit, *popt)

                # Find tau (time constant of the exponential decay)
                # The time constant τ is the amount of time that an exponentially decaying quantity takes
                # to decay by a factor of 1/e. Because 1/e is approximately 0.368,
                # τ is the amount of time that the quantity takes to decay to approximately 36.8%
                # of its original amount.
                tau = 1/popt[1]*-1

                # *****************************************************************************************************

                fig, axs = plt.subplots(1, 2, figsize=(11, 5))

                # Plot some results
                axs[1].plot(x, y_samples, color='lightgrey')
                axs[1].set_ylabel("mV (norm)", size=9, color='#939799')
                axs[1].set_xlabel("ms", size=9, color='#939799')
                axs[1].set_title(f' \n \n \n Chunk {i} - Waveforms: {len(trn_samples_chunk)}  \n', fontsize=9,
                                 loc='center', color='#939799')

                # Plot key points
                axs[1].plot([pos_amp_time], [pos_amp], marker='D', color='lightgrey')
                axs[1].plot([neg_amp_time], [neg_amp], marker='D', color='lightgrey')
                axs[1].plot([zero_crossing_time], [zero_crossing], marker='D', color='lightgrey')
                axs[1].plot([onset_time], [onset], marker='D', color='salmon')
                axs[1].plot([end_time], [end], marker='D', color='salmon')

                # Plot fall and rise "slopes"
                axs[1].plot([zero_crossing_time, zero_crossing_time], [neg_amp_10, neg_amp_90], color='lightpink')
                axs[1].plot([zero_crossing_time], [neg_amp_90], color='lightpink', marker='v')
                axs[1].plot([zero_crossing_time, zero_crossing_time], [pos_amp_10, pos_amp_90], color='lightpink')
                axs[1].plot([zero_crossing_time], [pos_amp_90], color='lightpink', marker='^')

                # Plot re-polarization slope
                axs[1].plot(xs[1:-1], abline_values[1:-1], color='darkmagenta')

                # Plot exponential fit
                axs[1].plot(x_fit[1:], y_fit[1:], color='lightpink')

                # Plot recovery slope
                axs[1].plot(x_end_sl[2:-1], abline_values1[2:-1], color='darkmagenta')

                # Plot horizontal widths
                axs[1].plot([pos_rise_hw, pos_fall_hw], [pos_amp_50, pos_amp_50], color='gray', ls='dashed')
                axs[1].plot([pos_rise_time, pos_fall_time], [pos_amp_10, pos_amp_10], color='gray', ls='dashed')
                axs[1].plot([pos_risePk_time, pos_fallPk_time], [pos_amp_90, pos_amp_90], color='gray', ls='dashed')
                axs[1].plot([neg_fall_hw, neg_rise_hw], [neg_amp_50, neg_amp_50], color='gray', ls='dashed')
                axs[1].plot([neg_fall_time, neg_rise_time], [neg_amp_10, neg_amp_10], color='gray', ls='dashed')
                axs[1].plot([neg_fallPk_time, neg_risePk_time], [neg_amp_90, neg_amp_90], color='gray', ls='dashed')

                # Plot duration
                axs[1].plot([neg_amp_time, pos_amp_time], [neg_amp-30, neg_amp-30], color='dimgray')
                axs[1].plot([neg_amp_time], [neg_amp-30], color='dimgray', marker='>')
                axs[1].plot([pos_amp_time], [neg_amp-30], color='dimgray', marker='<')

                leg_wvf_features = [f'Max Amp = {pos_amp:.2f}  \n'
                                    f'Min Amp = {neg_amp:.2f}  \n'
                                    f'Duration = {duration:.2f}  \n'
                                    f'Onset time = {onset_time:.2f}  \n'
                                    f'End time = {end_time:.2f}  \n'
                                    f'Pk-tr ratio = {peak_trough_ratio:.2f}  \n'
                                    f'Repol. slope = {rep_slope:.2f}  \n'
                                    f'Recov. slope = {rec_slope:.2f}  \n'
                                    f'Rise slope = {pos_rise_slope:.2f}  \n'
                                    f'Fall slope = {neg_fall_slope:.2f}  \n'
                                    f'Endslope Tau = {tau:.2f}  \n'
                                    f'Pos. half dur = {posHw_duration:.2f}  \n'
                                    f'Neg. half dur = {negHw_duration:.2f}  \n'
                                    ]

                leg_1 = axs[1].legend(leg_wvf_features, loc='best', frameon=False, fontsize=7)

                for text in leg_1.get_texts():
                    text.set_color("gray")

                for lh in leg_1.legendHandles:
                    lh.set_alpha(0)

                # Plot temporal features?
                axs[0].hist(isi_chunk_no_clipped, bins=isi_log_bins, color='lightgray')
                axs[0].set_xscale('log')
                # axs[1, 0].set_xlim([0.1, 1000])
                axs[0].set_xlim(left=0.1)
                axs[0].set_xlabel('ms (log scale)', size=9)
                axs[0].set_title(f' \n \n \n Chunk {i} - Inter Spike Interval  \n',
                                 fontsize=9, loc='center', color='#939799')

                leg_temp_features = [f'mfr = {mfr} Hz \n'
                                     f'mifr = {mifr} Hz \n'
                                     f'Entropy = {entropy} \n'
                                     f'Median (isi) = {med_isi}  \n'
                                     f'Mode (isi) = {mode_isi}  \n'
                                     f'Perc 5th (isi) = {prct5ISI}  \n'
                                     f'Mean CV2 = {CV2_mean}  \n'
                                     f'Median CV2 = {CV2_median}  \n'
                                     f'Mean CV = {CV}  \n'
                                     f'IR = {IR}  \n'
                                     f'Lv = {Lv}  \n'
                                     f'LvR = {LvR}  \n'
                                     f'LcV = {LcV}  \n'
                                     f'SI = {SI}  \n'
                                     f'skw = {skw}  \n']

                leg_0 = axs[0].legend(leg_temp_features, loc='best', frameon=False, fontsize=7)

                for text in leg_0.get_texts():
                    text.set_color("gray")

                for lh in leg_0.legendHandles:
                    lh.set_alpha(0)

                plt.suptitle(f'Recording {sample_probe} - Unit {unit}', fontsize=10, color='#939799')
                format_plot(axs[0])
                format_plot(axs[1])
                plt.tight_layout()
                plt.show()

        # Join trn samples of good chunks for temporal features computation
        # trn_samples_block = np.concatenate(l_chunk_trns).ravel()

        # Compute Inter-Spike Interval for the block of good chunks
        # We need to exclude the times between chunks with the exclusion quartile
        # isi_block_clipped = compute_isi(trn_samples_block, quantile=exclusion_quantile)
        # isi_block_no_clipped = compute_isi(trn_samples_block)

        # plt.hist(isi_block_no_clipped, bins=isi_log_bins, color='lightgray')
        # plt.gca().set_xscale('log')
        # plt.show()

        # Compute temporal features
        # mifr, med_isi, mode_isi, prct5ISI, entropy, CV2_mean, CV2_median, CV, IR, Lv, LvR, LcV, SI, skw\
        #     = compute_isi_features(isi_block_clipped)

        # print('MFR: ', unit_quality_metrics['MFRBlockHz'][0])
        # print('MIFR: ', mifr)
        # print('med_isi: ', med_isi)
        # print('mode_isi: ', mode_isi)
        # print('prct5ISI: ', prct5ISI)
        # print('entropy: ', entropy)
        # print('CV2_mean: ', CV2_mean)
        # print('CV2_median: ', CV2_median)
        # print('CV: ', CV)
        # print('IR: ', IR)
        # print('Lv: ', Lv)
        # print('LvR: ', LvR)
        # print('LcV: ', LcV)
        # print('SI: ', SI)
        # print('skw: ', skw)

        # unit_temporal_features = pd.DataFrame({
        #                             'Unit': unit,
        #                             'tf_MIFRBlockHz': mifr,
        #                             'tf_MedIsi': med_isi,
        #                             'tf_ModeIsi': mode_isi,
        #                             'tf_Perc5Isi': prct5ISI,
        #                             'tf_entropy': entropy,
        #                             'tf_CV2Mean': CV2_mean,
        #                             'tf_CV2Median': CV2_median,
        #                             'tf_CV': CV,
        #                             'tf_Ir': IR,
        #                             'tf_Lv': Lv,
        #                             'tf_LvR': LvR,
        #                             'tf_LcV': LcV,
        #                             'tf_Si': SI,
        #                             'tf_skw': skw
        #                             }, index=[0])

        # df_qm_temp = pd.merge(unit_quality_metrics, unit_temporal_features, left_on='Unit', right_on='Unit')
        # df_qm_temp.to_csv(f'Images/Pipeline/Sample_{sample_probe}/Features/Sample-{sample}-unit-{unit}.csv', index=False)

