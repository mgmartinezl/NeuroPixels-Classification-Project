"""
Project: Neuropixels classification pipeline
Author: Gabriela Martinez
Script: QualityCheckPipeline.py
Description: implementation of quality metrics on units and chunks to choose the right ones for the next stage in the
classification pipeline.

"""

from DataPreprocessing.AuxFunctions import *
import pandas as pd
from operator import itemgetter
from rtn.npix.spk_t import trn
from scipy import interpolate
from scipy.interpolate import interp1d
import os
import re
import ast


np.seterr(all='ignore')

data_sets = [

    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1'  # DONE
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1'  # DONE
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1'  # DONE
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1'  # DONE
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1'  # DONE
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1'  # DONE
    # 'F:/data/MLI/19-12-13_YC007/19-12-13_YC007_probe1'  # DONE
    # 'F:/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1'  # DONE
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1'  # DONE
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1'  # DONE
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1'  # DONE
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1'  # DONE
    # 'F:/data/MFB/19-01-23_YC012/19-01-23_YC012_probe1'  # DONE
    # 'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1'  # DONE
    # 'F:/data/MFB/19-01-23_YC013/19-01-23_YC013_probe1'  # DONE
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1'  # DONE
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1'  # DONE
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2'  # DONE
    # 'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1'  # DONE
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1'  # DONE
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1'  # DONE
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1'  # DONE
    # 'F:/data/MFB/19-10-02_YC019/19-10-02_YC019_probe1'  # DONE
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1'  # DONE
    # 'F:/data/MFB/19-10-02_YC020/19-10-02_YC020_probe1'  # DONE
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1'  # DONE
    # 'C:/Recordings/19-10-22_YC022/19-10-22_YC022_probe1'  # DONE
    # 'C:/Recordings/19-10-23_YC022/19-10-23_YC022_probe1'  # DONE
    # 'F:/data/MLI/19-11-14_YC029/19-11-14_YC029_probe1'  # DONE
    # 'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1'  # DONE
    # 'F:/data/PkC/19-10-28_YC031/19-10-28_YC031_probe1'  # DONE
    # 'C:/Recordings/19-11-04_YC036/19-11-04_YC036_probe1'  # DONE
    # 'C:/Recordings/19-11-05_YC036/19-11-05_YC036_probe1'  # DONE
    # 'C:/Recordings/19-11-04_YC037/19-11-04_YC037_probe1'  # DONE
    # 'C:/Recordings/19-11-05_YC037/19-11-05_YC037_probe1'  # DONE 359, 407 ERRORS
    # 'C:/Recordings/19-11-08_YC038/19-11-08_YC038_probe2'  # DONE
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe2'  # NO GOOD UNITS
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1'  # DONE
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1'   # DONE
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1'  # DONE
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2'  # 344, 373, 591, 613, 62 DONE
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe1'  # DONE
    # DK 186 and 187 missing for Golgi
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2'  # 344, 373, 591, 613, 62
    # 'D:/Recordings/20-15-06_DK186/20-15-06_DK186_probe1'
    'D:/Recordings/20-27-06_DK187/20-27-06_DK187_probe1'

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
    all_good_units = pd.read_csv('../Images/Pipeline/GoodUnitsAllRecordings.csv')
    good_units = ast.literal_eval(all_good_units[all_good_units['Sample'].str.match(sample_probe)]['Units'].values[0])

    paths = [f'../Images/Pipeline/Sample_{sample_probe}/Features/Images',
             f'../Images/Pipeline/Sample_{sample_probe}/Features/Files']

    print(f"Quality units found in current sample: {len(good_units)} --> {good_units}")

    # Extract chunks for good units
    units_chunks = pd.read_csv(f'../Images/Pipeline/Sample_{sample_probe}/Sample-{sample_probe}-ChunksUnits.csv')

    good_units = [493, 479, 385, 404]

    d = 3
    h = 2
    r = 10

    for unit in good_units:

        for features_path in paths:
            if not os.path.exists(features_path):
                os.makedirs(features_path)

        print(f'Unit >>>> {unit}')

        # Extract chunks
        chunks = list(map(int, eval(units_chunks[units_chunks['Unit'] == unit]['GoodChunks'][
            units_chunks[units_chunks['Unit'] == unit].index.values.astype(int)[0]])))

        chunks_wvf = list(map(int, eval(units_chunks[units_chunks['Unit'] == unit]['WvfChunks'][
            units_chunks[units_chunks['Unit'] == unit].index.values.astype(int)[0]])))

        chunks_wvf = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

        # Store quality metrics
        unit_quality_metrics = pd.read_csv(f'../Images/Pipeline/Sample_{sample_probe}/GoodUnits/Files/Sample-{sample}-unit-{unit}.csv')

        # Retrieve peak channel for this unit
        peak_channel = unit_quality_metrics['PeakChUnit'][0]

        # Collect for later...
        l_chunk_trns = []
        l_chunk_wvfs = []

        # Read chunks...
        for i in range(n_chunks):
            if i in chunks:

                # These chunks will be used to compute temporal features
                # Chunk length and times in seconds
                chunk_start_time = i * chunk_size_s
                chunk_end_time = (i + 1) * chunk_size_s

                # Chunk spikes
                trn_samples_chunk = trn(dp, unit=unit,
                                        subset_selection=[(chunk_start_time, chunk_end_time)],
                                        enforced_rp=0.5, again=again)

                l_chunk_trns.append(trn_samples_chunk)

            else:
                continue

        for i in range(n_chunks):
            if i in chunks_wvf:

                # These chunks will be used to compute waveform-based features. They will be maximum 3
                # Chunk length and times in seconds
                chunk_start_time = i * chunk_size_s
                chunk_end_time = (i + 1) * chunk_size_s

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

        # *********************************************************************************************************

        # Put chunks together
        # Join trn samples of good chunks for temporal features computation
        trn_samples_block = np.concatenate(l_chunk_trns).ravel()

        # Compute Inter-Spike Interval for the block of good chunks
        # We need to exclude the times between chunks with the exclusion quartile
        isi_block_clipped = compute_isi(trn_samples_block, quantile=exclusion_quantile)
        isi_block_no_clipped = compute_isi(trn_samples_block)

        # Estimate number of optimal bins for ISI
        isi_bins = estimate_bins(isi_block_no_clipped, rule='Sqrt')

        isi_log_bins = None

        try:
            isi_log_bins = np.geomspace(isi_block_no_clipped.min(), isi_block_no_clipped.max(), isi_bins)
        except ValueError:
            pass

        # Compute temporal features for the unit
        mfr = mean_firing_rate(isi_block_clipped)
        mifr, med_isi, mode_isi, prct5ISI, entropy, CV2_mean, CV2_median, CV, IR, Lv, LvR, LcV, SI, skw \
            = compute_isi_features(isi_block_clipped)

        # Put waveforms together
        waveform_block = np.mean(l_chunk_wvfs, axis=0)

        # Peaks (not normalized)
        maxAmp = round(max(waveform_block), 2)
        minAmp = round(min(waveform_block), 2)

        # Normalize mean waveform
        # waveform_block_n = waveform_mean_subtraction(waveform_block)
        waveform_block = waveform_block / -min(waveform_block)

        # Compute the baseline
        baseline = np.mean(waveform_block[0:40])

        # Make pairwise list of tuples with samples and wave position
        # x_samples = np.arange(0, chunk_wvf.shape[0])  # From 0 to 119, in samples. Maybe it is better to put in ms?
        x_samples = list(np.arange(len(waveform_block), step=1))
        y_samples = waveform_block.copy()
        x = [round(i / 30, 2) for i in x_samples]  # in ms
        points = list(zip(x, waveform_block))

        # Find peak amplitudes
        # max_amp = np.max(chunk_wvf) - baseline  # Ask about this
        # min_amp = np.min(chunk_wvf) - baseline
        pos_amp = max(points, key=itemgetter(1))[1]  # Point 1 of 9
        pos_amp_time = max(points, key=itemgetter(1))[0]  # Point 1 of 9
        neg_amp = min(points, key=itemgetter(1))[1]  # Point 2 of 9
        neg_amp_time = min(points, key=itemgetter(1))[0]  # Point 2 of 9

        # # Find cut points >> 10%, 50%, 90%
        pos_amp_5 = pos_amp * 0.05
        pos_amp_10 = pos_amp * 0.1
        pos_amp_50 = pos_amp * 0.5
        pos_amp_90 = pos_amp * 0.9
        neg_amp_10 = neg_amp * 0.1
        neg_amp_50 = neg_amp * 0.5
        neg_amp_90 = neg_amp * 0.9

        # A couple of features depend on the order of the peaks in the waveform
        # Check if the first peak is positive
        if pos_amp_time < neg_amp_time:
            neg_peak_first = 0
        else:
            neg_peak_first = 1

        # Compute onset time
        if neg_peak_first == 1:

            waveform_block_onset = waveform_block * -1
            onset = min(waveform_block_onset) * 0.05

            try:
                onset_time = interpolate.UnivariateSpline(
                    x[:x.index(neg_amp_time)],
                    (np.array(waveform_block[:x.index(neg_amp_time)]) - onset)).roots()[-1]

            except Exception:
                try:
                    onset_time = interpolate.UnivariateSpline(
                        x[:x.index(neg_amp_time)],
                        (np.array(waveform_block[:x.index(neg_amp_time)]) - np.mean(waveform_block[:10]))).roots()[-1]

                except Exception:
                    try:
                        f_ = interp1d(waveform_block[:x.index(neg_amp_time)],
                                      x[:x.index(neg_amp_time)],
                                      fill_value="extrapolate")
                        onset_time = f_(onset)

                    except Exception:
                        x1 = x[:20]
                        y = waveform_block[:20]
                        slope, intercept, r_value, p_value, std_error = stats.linregress(x1, y)
                        onset_time = -intercept / slope

            end = pos_amp * 0.05
            try:
                # end_time = interpolate.UnivariateSpline(x[x.index(pos_amp_time):], (
                #           np.array(waveform_block[x.index(pos_amp_time):]) - end)).roots()[0]  # Point 4 of 9

                # end, idx = find_nearest(waveform_block[x.index(neg_amp_time)+35:x.index(neg_amp_time) + 45], end)
                # end_time = x[x.index(neg_amp_time)+35:x.index(neg_amp_time) + 45][idx]
                end, idx = find_nearest(waveform_block[x.index(neg_amp_time):], end)
                end_time = x[x.index(neg_amp_time):][idx]

            except Exception:
                end_time = interpolate.UnivariateSpline(x[x.index(pos_amp_time):], (
                        np.array(waveform_block[x.index(pos_amp_time):]) - np.mean(waveform_block[-10:]))).roots()[0]

            duration = round(np.abs(pos_amp_time - neg_amp_time), 3)

            # For the exponential fit

            popt = None

            try:
                x_exp = x[x.index(pos_amp_time):]
                y_exp = waveform_block[x.index(pos_amp_time):]
                popt, pcov = opt.curve_fit(exponential_fit, x_exp, y_exp, p0=(0, 0))
                x_fit = np.linspace(min(x_exp), max(x_exp), 25)
                y_fit = exponential_fit(x_fit, *popt)

            except Exception:
                try:
                    x_exp = x[x.index(neg_amp_time):]
                    y_exp = waveform_block[x.index(neg_amp_time):]
                    popt, pcov = opt.curve_fit(exponential_fit, x_exp, y_exp, p0=(0, 0))
                    x_fit = np.linspace(min(x_exp), max(x_exp), 25)
                    y_fit = exponential_fit(x_fit, *popt)
                except Exception:
                    pass

            # Find tau (time constant of the exponential decay)
            # The time constant τ is the amount of time that an exponentially decaying quantity takes
            # to decay by a factor of 1/e. Because 1/e is approximately 0.368,
            # τ is the amount of time that the quantity takes to decay to approximately 36.8%
            # of its original amount.
            try:
                tau = 1 / popt[1] * -1
            except Exception:
                tau = 0
                print('ALERT! > Tau could not be computed')

            # Compute first peak end time or zero-crossing (when the negative or the positive peak ends)
            try:
                x1 = x[x.index(neg_amp_time):x.index(pos_amp_time)]
                y = waveform_block[x.index(neg_amp_time):x.index(pos_amp_time)]
                f = interpolate.interp1d(x1, y)
                zero_crossing_time = interpolate.UnivariateSpline(x1, np.array(y) - 0).roots()[0]
                zero_crossing = f(zero_crossing_time)

            except Exception:
                x1 = x[x.index(neg_amp_time):x.index(pos_amp_time)]
                y = waveform_block[x.index(neg_amp_time):x.index(pos_amp_time)]
                slope, intercept, r_value, p_value, std_error = stats.linregress(x1, y)
                zero_crossing_time = -intercept / slope
                zero_crossing = (slope * zero_crossing_time) + intercept

        else:
            # If peak positive is first
            onset = pos_amp * 0.05
            end = neg_amp * 0.05

            try:
                onset_time = interpolate.UnivariateSpline(
                    x[x.index(pos_amp_time)-10:x.index(pos_amp_time)-1],
                    (np.array(waveform_block[x.index(pos_amp_time)-10:x.index(pos_amp_time)-1]) - onset)).roots()[-1]

            except Exception:
                try:
                    onset_time = interpolate.UnivariateSpline(
                        x[:x.index(pos_amp_time)],
                        (np.array(waveform_block[:x.index(pos_amp_time)]) - np.mean(waveform_block[:10]))).roots()[-1]

                except Exception:
                    f_ = interp1d(waveform_block[:x.index(pos_amp_time)], x[:x.index(pos_amp_time)])
                    onset_time = f_(onset)

            try:
                end_time = interpolate.UnivariateSpline(x[x.index(neg_amp_time)-1:], (
                        np.array(waveform_block[x.index(neg_amp_time)-1:]) - end)).roots()[0]

            except Exception:
                end_time = interpolate.UnivariateSpline(x[x.index(neg_amp_time) - 1:], (
                        np.array(waveform_block[x.index(neg_amp_time) - 1:]) - np.mean(waveform_block[-10:]))).roots()[0]

                # end, idx = find_nearest(waveform_block[x.index(neg_amp_time)+15:x.index(neg_amp_time) + 45], end)
                # end_time = x[x.index(neg_amp_time)+15:x.index(neg_amp_time) + 45][idx]

            duration = round(np.abs(neg_amp_time - pos_amp_time), 3)

            # Fit exponential to ending slope
            try:
                # For the exponential fit
                x_exp = x[x.index(neg_amp_time):]
                y_exp = waveform_block[x.index(neg_amp_time):]
                popt, pcov = opt.curve_fit(exponential_fit, x_exp, y_exp, p0=(0, 0))
                x_fit = np.linspace(min(x_exp), max(x_exp), 25)
                y_fit = exponential_fit(x_fit, *popt)

            except Exception:
                # For the exponential fit
                x_exp = x[x.index(pos_amp_time):]
                y_exp = waveform_block[x.index(pos_amp_time):]
                popt, pcov = opt.curve_fit(exponential_fit, x_exp, y_exp, p0=(0, 0))
                x_fit = np.linspace(min(x_exp), max(x_exp), 25)
                y_fit = exponential_fit(x_fit, *popt)

            # Find tau (time constant of the exponential decay)
            # The time constant τ is the amount of time that an exponentially decaying quantity takes
            # to decay by a factor of 1/e. Because 1/e is approximately 0.368,
            # τ is the amount of time that the quantity takes to decay to approximately 36.8%
            # of its original amount.
            try:
                tau = 1 / popt[1] * -1
            except Exception:
                tau = None
                print('ALERT! > Tau could not be computed')

            # Compute first peak end time (when the negative or the positive peak ends) >> 2 OPTIONS
            try:
                x1 = x[x.index(pos_amp_time):x.index(neg_amp_time)]
                y = waveform_block[x.index(pos_amp_time):x.index(neg_amp_time)]
                f = interpolate.interp1d(x1, y)
                zero_crossing_time = interpolate.UnivariateSpline(x1, np.array(y) - 0).roots()[0]
                zero_crossing = f(zero_crossing_time)

            except Exception:
                x1 = x[x.index(pos_amp_time):x.index(neg_amp_time)]
                y = waveform_block[x.index(pos_amp_time):x.index(neg_amp_time)]
                slope, intercept, r_value, p_value, std_error = stats.linregress(x1, y)
                zero_crossing_time = -intercept / slope
                zero_crossing = (slope * zero_crossing_time) + intercept

        # COMPUTE HALF-WIDTH POINTS >> 10%, 50%, 90%
        if neg_peak_first == 1:

            print('Negative peak first')

            # Positive 10%, 90%, and Positive Fall Slope >> 3 options to compute
            pos_fall_slope = np.abs(pos_amp_90 - pos_amp_10)  # This really doesn't matter

            pos_amp_10_time = None
            pos_amp_90_time = None

            try:

                pos_amp_10_time = interpolate.UnivariateSpline(
                    x[x.index(neg_amp_time):x.index(pos_amp_time) + 1],
                    (np.array(waveform_block[x.index(neg_amp_time):x.index(pos_amp_time) + 1]) - pos_amp_10)).roots()[0]

                pos_amp_90_time = interpolate.UnivariateSpline(
                    x[x.index(neg_amp_time):x.index(pos_amp_time) + 1],
                    (np.array(waveform_block[x.index(neg_amp_time):x.index(pos_amp_time) + 1]) - pos_amp_90)).roots()[0]

                # pos_amp_10, pos_amp_10_idx = find_nearest(waveform_block[x.index(neg_amp_time):], pos_amp_10)
                # pos_amp_10_time = x[x.index(neg_amp_time):][pos_amp_10_idx]
                #
                # pos_amp_90, pos_amp_90_idx = find_nearest(waveform_block[x.index(neg_amp_time):], pos_amp_90)
                # pos_amp_90_time = x[x.index(neg_amp_time):][pos_amp_90_idx]

                # Check a few things
                if pos_amp_90_time < pos_amp_10_time:
                    a = pos_amp_90_time
                    pos_amp_90_time = pos_amp_10_time
                    pos_amp_10_time = a

                if pos_amp_90_time < zero_crossing_time or pos_amp_10_time < zero_crossing_time:
                    pos_amp_90_time = zero_crossing_time + duration / 2
                    pos_amp_10_time = zero_crossing_time

                if pos_amp_90_time > pos_amp_time or pos_amp_10_time > pos_amp_time:
                    pos_amp_90_time = zero_crossing_time + duration / 2
                    pos_amp_10_time = zero_crossing_time

                pos_fall_slope_time = np.abs(pos_amp_90_time - pos_amp_10_time)

                # Positive decay time: spike duration from positive peak to 10% of positive peak.
                pos_decay_time = np.abs(pos_amp_time - pos_amp_10_time)

                print('+ 10%, 90% Case 1')

            except Exception:

                pos_amp_10_time = interpolate.UnivariateSpline(
                    x[:x.index(pos_amp_time)+1],
                    (np.array(waveform_block[:x.index(pos_amp_time)+1]) - pos_amp_10)).roots()[-1]

                pos_amp_90_time = interpolate.UnivariateSpline(
                    x[:x.index(pos_amp_time)+1],
                    (np.array(waveform_block[:x.index(pos_amp_time) + 1]) - pos_amp_90)).roots()[-1]

                # pos_amp_10, pos_amp_10_idx = find_nearest(waveform_block[x.index(pos_amp_time):], pos_amp_10)
                # pos_amp_10_time = x[x.index(pos_amp_time):][pos_amp_10_idx]
                #
                # pos_amp_90, pos_amp_90_idx = find_nearest(waveform_block[x.index(pos_amp_time):], pos_amp_90)
                # pos_amp_90_time = x[x.index(pos_amp_time):][pos_amp_90_idx]

                # Check a few things
                if pos_amp_90_time < pos_amp_10_time:
                    a = pos_amp_90_time
                    pos_amp_90_time = pos_amp_10_time
                    pos_amp_10_time = a

                if pos_amp_90_time < zero_crossing_time or pos_amp_10_time < zero_crossing_time:
                    pos_amp_90_time = zero_crossing_time + duration / 2
                    pos_amp_10_time = zero_crossing_time

                if pos_amp_90_time > pos_amp_time or pos_amp_10_time > pos_amp_time:
                    pos_amp_90_time = zero_crossing_time + duration / 2
                    pos_amp_10_time = zero_crossing_time

                pos_fall_slope_time = np.abs(pos_amp_90_time - pos_amp_10_time)

                # Positive decay time: spike duration from positive peak to 10% of positive peak.
                pos_decay_time = np.abs(pos_amp_time - pos_amp_10_time)

                print('+ 10%, 90% Case 2')

            # Negative 10%, 90%, and Negative Fall Slope >> 3 options to compute
            neg_fall_slope = np.abs(neg_amp_90 - neg_amp_10)

            try:
                neg_amp_10_time = interpolate.UnivariateSpline(
                    x[x.index(neg_amp_time):x.index(pos_amp_time)],
                    (np.array(waveform_block)[x.index(neg_amp_time):x.index(pos_amp_time)] - neg_amp_10),
                    ext=1).roots()[-1]

                neg_amp_90_time = interpolate.UnivariateSpline(x, (np.array(waveform_block) - neg_amp_90)).roots()[-1]

                # Check a few things
                if neg_amp_90_time > neg_amp_10_time:
                    a = pos_amp_90_time
                    neg_amp_90_time = neg_amp_10_time
                    neg_amp_10_time = a

                if neg_amp_90_time > zero_crossing_time or neg_amp_10_time > zero_crossing_time:
                    neg_amp_90_time = zero_crossing_time - duration / 2
                    neg_amp_10_time = zero_crossing_time

                if neg_amp_90_time < neg_amp_time or neg_amp_10_time < neg_amp_time:
                    neg_amp_90_time = zero_crossing_time - duration / 2
                    neg_amp_10_time = zero_crossing_time

                # neg_amp_10_time = neg_amp_10_time - duration

                print('- 10%, 90% Case 1')

            except Exception:
                f_ = interp1d(waveform_block[x.index(neg_amp_time):x.index(pos_amp_time)],
                              x[x.index(neg_amp_time):x.index(pos_amp_time)],
                              fill_value="extrapolate")

                neg_amp_10_time = f_(neg_amp_10)

                f_ = interp1d(waveform_block[x.index(neg_amp_time):x.index(pos_amp_time)],
                              x[x.index(neg_amp_time):x.index(pos_amp_time)])

                neg_amp_90_time = f_(neg_amp_90)

                # Check a few things
                if neg_amp_90_time > neg_amp_10_time:
                    a = pos_amp_90_time
                    neg_amp_90_time = neg_amp_10_time
                    neg_amp_10_time = a

                if neg_amp_90_time > zero_crossing_time or neg_amp_10_time > zero_crossing_time:
                    neg_amp_90_time = zero_crossing_time - duration / 2
                    neg_amp_10_time = zero_crossing_time

                if neg_amp_90_time < neg_amp_time or neg_amp_10_time < neg_amp_time:
                    neg_amp_90_time = zero_crossing_time - duration / 2
                    neg_amp_10_time = zero_crossing_time

                print('- 10%, 90% Case 2')

            # pos_amp_10_time = pos_amp_10_time - 0.22 * duration

            neg_fall_slope_time = np.abs(neg_amp_90_time - neg_amp_10_time)

            # Negative decay time: spike duration from negative peak to 10% of negative peak.
            neg_decay_time = np.abs(neg_amp_time - neg_amp_10_time)

            # Positive 50% >> 2 sides NEEDED
            try:

                x_rise_hw, x_rise_hw_idx = find_nearest(waveform_block[x.index(neg_amp_time) - 1:x.index(pos_amp_time) + 1], pos_amp_50)
                pos_rise_hw = x[x.index(neg_amp_time) - 1:x.index(pos_amp_time) + 1][x_rise_hw_idx]

                print('Pos rise half Case 1')

            except Exception:
                # IF NEW IDEA DOSNT WORK, UNCOMMENT THIS
                # f_ = interp1d(waveform_block[:x.index(pos_amp_time)],
                #               x[:x.index(pos_amp_time)],
                #               fill_value="extrapolate")
                # pos_rise_hw = f_(pos_amp_50)

                pos_rise_hw = interpolate.UnivariateSpline(
                    x[x.index(neg_amp_time) - 1:x.index(pos_amp_time) + 1],
                    (np.array(waveform_block[x.index(neg_amp_time) - 1:x.index(pos_amp_time) + 1]) - pos_amp_50)).roots()[0]

                print('MODIFICATION: Pos rise half Case 2')

            try:

                x_fall_hw, x_fall_hw_idx = find_nearest(waveform_block[x.index(pos_amp_time):x.index(pos_amp_time)], pos_amp_50)
                pos_fall_hw = x[x.index(pos_amp_time):x.index(pos_amp_time)][x_fall_hw_idx]

                # x_fall_hw, x_fall_hw_idx = find_nearest(waveform_block[:x.index(pos_amp_time)+10], pos_amp_50)
                # pos_fall_hw = x[:x.index(pos_amp_time)+10][x_fall_hw_idx]
                #
                # if pos_fall_hw > end_time:
                #     pos_fall_hw = neg_amp_time + duration*1.5

            except Exception:

                # f_ = interp1d(waveform_block[x.index(pos_amp_time):], x[x.index(pos_amp_time):], fill_value="extrapolate")
                # pos_fall_hw = f_(pos_amp_50)

                pos_fall_hw = interpolate.UnivariateSpline(
                    x[x.index(pos_amp_time):],
                    (np.array(waveform_block[x.index(pos_amp_time):]) - pos_amp_50)).roots()[0]  # Point 8 of 13

                print('MODIFICATION: Pos fall half Case 2')

            # Check a few things
            if pos_fall_hw < pos_rise_hw:
                a = pos_fall_hw
                pos_rise_hw = pos_fall_hw
                pos_fall_hw = a

            if pos_rise_hw > pos_amp_time:
                pos_rise_hw = pos_amp_time - duration/2

            if pos_fall_hw < pos_amp_time:
                pos_fall_hw = pos_amp_time + duration/2

            if pos_rise_hw < zero_crossing_time or pos_fall_hw < zero_crossing_time:
                pos_rise_hw = zero_crossing_time
                pos_fall_hw = zero_crossing_time + duration

            # pos_fall_hw = pos_fall_hw - 0.5

            posHw_duration = np.abs(pos_fall_hw - pos_rise_hw)

            # Negative 50% >> 2 sides NEEDED
            try:

                x_fall_hw, x_fall_hw_idx = find_nearest(waveform_block[:x.index(neg_amp_time)], neg_amp_50)
                neg_fall_hw = x[:x.index(neg_amp_time)][x_fall_hw_idx]

                print('Neg fall half Case 1')

            except Exception:
                try:
                    # Look for second positive maximum
                    points2 = [p for p in points if p[0] < neg_amp_time]
                    pos_amp2 = max(points2, key=itemgetter(1))[1]
                    pos_amp_time2 = max(points2, key=itemgetter(1))[0]
                    f_ = interp1d(waveform_block[x.index(pos_amp_time2):x.index(neg_amp_time)],
                                  x[x.index(pos_amp_time2):x.index(neg_amp_time)])
                    neg_fall_hw = f_(neg_amp_50)

                    print('Neg fall half Case 2')

                except Exception:

                    # f_ = interp1d(waveform_block[:x.index(neg_amp_time)],
                    #               x[:x.index(neg_amp_time)],
                    #               fill_value="extrapolate")
                    # neg_fall_hw = f_(neg_amp_50)

                    neg_fall_hw = interpolate.UnivariateSpline(
                        x[:x.index(neg_amp_time) + 1],
                        (np.array(waveform_block[:x.index(neg_amp_time) + 1]) - neg_amp_50)).roots()[-1]  # Point 9 of 13

                    if neg_fall_hw < onset_time:
                        neg_fall_hw = onset_time

                    print('MODIFICATION: Neg fall half Case 3')

            try:

                x_rise_hw, x_rise_hw_idx = find_nearest(waveform_block[x.index(neg_amp_time) - 1:x.index(pos_amp_time) + 1], neg_amp_50)
                neg_rise_hw = x[x.index(neg_amp_time) - 1:x.index(pos_amp_time) + 1][x_rise_hw_idx]

                print('Neg rise half Case 1')

            except Exception:

                # f_ = interp1d(waveform_block[x.index(neg_amp_time)-1:x.index(pos_amp_time)+1],
                #               x[x.index(neg_amp_time)-1:x.index(pos_amp_time)+1],
                #               fill_value="extrapolate")
                # neg_rise_hw = f_(neg_amp_50)

                neg_rise_hw = interpolate.UnivariateSpline(
                    x[x.index(neg_amp_time) - 1:x.index(pos_amp_time) + 1],
                    (np.array(waveform_block[x.index(neg_amp_time) - 1:x.index(pos_amp_time) + 1]) - neg_amp_50)).roots()[0]  # Point 10 of 13

                print('MODIFICATION: Neg rise half Case 2')

            # Check a few things
            if neg_rise_hw < neg_fall_hw:
                a = neg_rise_hw
                neg_rise_hw = neg_fall_hw
                neg_fall_hw = a

            # if neg_rise_hw > neg_amp_10_time:
            #     neg_rise_hw = neg_rise_hw + neg_fall_slope_time
            #
            # if neg_fall_hw > neg_amp_10_time:
            #     neg_fall_hw = neg_amp_10_time - duration

            if neg_rise_hw < neg_amp_time:
                # neg_rise_hw = neg_rise_hw + neg_fall_slope_time
                neg_rise_hw = neg_amp_time + duration/2

            if neg_fall_hw > neg_amp_time:
                neg_fall_hw = neg_amp_time - duration/2

            if neg_rise_hw > zero_crossing_time or neg_fall_hw > zero_crossing_time:
                neg_rise_hw = zero_crossing_time
                neg_fall_hw = zero_crossing_time - duration

            # Restrictions on onset time after all computations
            if np.abs(zero_crossing_time - onset_time) > 1. and onset < pos_amp_10:
                onset_time = zero_crossing_time - duration

            # Restrictions on the negative half-width
            if neg_fall_hw < onset_time:
                neg_fall_hw = onset_time + duration

            # neg_fall_hw = neg_fall_hw + duration*2.5
            # neg_rise_hw = neg_rise_hw - 0.05
            # neg_fall_hw = neg_fall_hw - 0.1
            # neg_fall_hw = neg_fall_hw - duration*5
            # neg_rise_hw = neg_rise_hw - duration*2

            negHw_duration = np.abs(neg_rise_hw - neg_fall_hw)

        else:

            # If first peak is positive
            print('Positive peak first')

            # Positive 10%, 90%, and Positive Fall Slope >> 3 options to compute
            pos_fall_slope = np.abs(pos_amp_90 - pos_amp_10)

            try:
                pos_amp_10_time = interpolate.UnivariateSpline(
                    x[:x.index(pos_amp_time)+1],
                    (np.array(waveform_block[:x.index(pos_amp_time)+1]) - pos_amp_10)).roots()[-1]
                pos_amp_90_time = interpolate.UnivariateSpline(
                    x[:x.index(pos_amp_time)+1],
                    (np.array(waveform_block)[:x.index(pos_amp_time)+1] - pos_amp_90)).roots()[0]

                # Check a few things
                if pos_amp_90_time < pos_amp_10_time:
                    a = pos_amp_90_time
                    pos_amp_90_time = pos_amp_10_time
                    pos_amp_10_time = a

                pos_fall_slope_time = np.abs(pos_amp_90_time - pos_amp_10_time)

                # Positive decay time: spike duration from positive peak to 10% of positive peak.
                pos_decay_time = pos_amp_time - pos_amp_10_time

                print('+ 10% 90% Case 1')

            except Exception:
                try:
                    pos_amp_10_time = interpolate.UnivariateSpline(
                        x[x.index(pos_amp_time):x.index(neg_amp_time)], (
                        np.array(waveform_block[x.index(pos_amp_time):x.index(neg_amp_time)]) - pos_amp_10)).roots()[0]

                    pos_amp_90_time = interpolate.UnivariateSpline(
                        x[x.index(pos_amp_time):x.index(neg_amp_time)], (
                        np.array(waveform_block[x.index(pos_amp_time):x.index(neg_amp_time)]) - pos_amp_90)).roots()[0]

                    # Check a few things
                    if pos_amp_90_time > pos_amp_10_time:
                        a = pos_amp_90_time
                        pos_amp_90_time = pos_amp_10_time
                        pos_amp_10_time = a

                    if pos_amp_90_time > zero_crossing_time or pos_amp_10_time > zero_crossing_time:
                        pos_amp_90_time = zero_crossing_time - duration / 2
                        pos_amp_10_time = zero_crossing_time

                    if pos_amp_90_time > pos_amp_time or pos_amp_10_time > pos_amp_time:
                        pos_amp_90_time = zero_crossing_time - duration / 2
                        pos_amp_10_time = zero_crossing_time

                    #pos_amp_90_time = pos_amp_90_time + 0.1

                    pos_fall_slope_time = np.abs(pos_amp_10_time - pos_amp_90_time)

                    # Positive decay time: spike duration from positive peak to 10% of positive peak.
                    pos_decay_time = pos_amp_10_time - pos_amp_time

                    print('+ 10% 90% Case 2')

                except Exception:
                    f_ = interp1d(waveform_block[x.index(pos_amp_time):], x[x.index(pos_amp_time):])
                    pos_amp_10_time = f_(pos_amp_10)

                    f_ = interp1d(waveform_block[x.index(pos_amp_time):], x[x.index(pos_amp_time):])
                    pos_amp_90_time = f_(pos_amp_90)

                    # Check a few things
                    if pos_amp_90_time > pos_amp_10_time:
                        a = pos_amp_90_time
                        pos_amp_90_time = pos_amp_10_time
                        pos_amp_10_time = a

                    if pos_amp_90_time > zero_crossing_time or pos_amp_10_time > zero_crossing_time:
                        pos_amp_90_time = zero_crossing_time - duration / 2
                        pos_amp_10_time = zero_crossing_time

                    if pos_amp_90_time > pos_amp_time or pos_amp_10_time > pos_amp_time:
                        pos_amp_90_time = zero_crossing_time - duration / 2
                        pos_amp_10_time = zero_crossing_time

                    pos_fall_slope_time = np.abs(pos_amp_10_time - pos_amp_90_time)

                    # Positive decay time: spike duration from positive peak to 10% of positive peak.
                    pos_decay_time = pos_amp_10_time - pos_amp_time

                    print('+ 10% 90% Case 3')

            # Negative 10%, 90%, and Negative Fall Slope >> 3 options to compute
            neg_fall_slope = np.abs(neg_amp_90 - neg_amp_10)

            try:
                neg_amp_10_time = interpolate.UnivariateSpline(
                    x[x.index(neg_amp_time):],
                    (np.array(waveform_block[x.index(neg_amp_time):]) - neg_amp_10)).roots()[-1]

                neg_amp_90_time = interpolate.UnivariateSpline(x, (np.array(waveform_block) - neg_amp_90)).roots()[-1]

                # Check a few things
                if neg_amp_10_time < neg_amp_90_time:
                    a = neg_amp_10_time
                    neg_amp_90_time = neg_amp_10_time
                    neg_amp_10_time = a

                neg_fall_slope_time = np.abs(neg_amp_10_time - neg_amp_90_time)

                # Negative decay time: spike duration from negative peak to 10% of negative peak.
                neg_decay_time = neg_amp_10_time - neg_amp_time

                print('- 10% 90% Case 1')

            except Exception:
                try:
                    neg_amp_10_time = interpolate.UnivariateSpline(
                        x[x.index(pos_amp_time):x.index(neg_amp_time)],
                        (np.array(waveform_block[x.index(pos_amp_time):x.index(neg_amp_time)]) - neg_amp_10)).roots()[0]

                    neg_amp_90_time = interpolate.UnivariateSpline(
                        x[x.index(pos_amp_time):x.index(neg_amp_time)],
                        (np.array(waveform_block[x.index(pos_amp_time):x.index(neg_amp_time)]) - neg_amp_90)).roots()[0]

                    # Check a few things
                    if neg_amp_10_time > neg_amp_90_time:
                        a = neg_amp_10_time
                        neg_amp_90_time = neg_amp_10_time
                        neg_amp_10_time = a

                    if neg_amp_10_time < zero_crossing_time or neg_amp_90_time < zero_crossing_time:
                        neg_amp_10_time = zero_crossing_time
                        neg_amp_90_time = zero_crossing_time + duration / 2

                    if neg_amp_10_time > neg_amp_time or neg_amp_90_time > neg_amp_time:
                        neg_amp_10_time = zero_crossing_time
                        neg_amp_90_time = zero_crossing_time + duration / 2

                    # neg_amp_10_time = neg_amp_10_time + duration

                    neg_fall_slope_time = np.abs(neg_amp_90_time - neg_amp_10_time)

                    # Negative decay time: spike duration from negative peak to 10% of negative peak.
                    neg_decay_time = np.abs(neg_amp_time - neg_amp_10_time)

                    print('- 10% 90% Case 2')

                except Exception:
                    slope, intercept, r_value, p_value, std_error = \
                        stats.linregress(x[x.index(neg_amp_time):],
                                         waveform_block[x.index(neg_amp_time):])
                    neg_amp_10_time = (slope * neg_amp_10) + intercept

                    f_ = interp1d(waveform_block[x.index(neg_amp_time):], x[x.index(neg_amp_time):])
                    neg_amp_90_time = f_(neg_amp_90)

                    # try:
                    #     neg_amp_90_time = interpolate.UnivariateSpline(
                    #         x[x.index(pos_amp_time):x.index(neg_amp_time)],
                    #         (np.array(waveform_block[x.index(pos_amp_time):x.index(neg_amp_time)]) - neg_amp_90)).roots()[0]
                    #
                    # except Exception:

                    # REVIEW THIS CAREFULLY!
                    # if neg_amp_10_time < neg_amp_time and neg_amp_10_time < zero_crossing_time:
                    #     neg_amp_10_time = zero_crossing_time + (duration/2)

                    # Check a few things
                    if neg_amp_10_time < neg_amp_90_time:
                        a = neg_amp_10_time
                        neg_amp_90_time = neg_amp_10_time
                        neg_amp_10_time = a

                    if neg_amp_10_time < neg_amp_time or neg_amp_90_time < neg_amp_time:
                        neg_amp_10_time = neg_amp_time + duration/2
                        neg_amp_90_time = neg_amp_time

                    # neg_amp_10, neg_amp_10_time_idx = find_nearest(waveform_block[x.index(pos_amp_time):], neg_amp_10)
                    # neg_amp_10_time = x[x.index(pos_amp_time):][neg_amp_10_time_idx]

                    # neg_amp_10_time = neg_amp_10_time + duration
                    neg_amp_10_time = neg_amp_10_time - duration

                    neg_fall_slope_time = np.abs(neg_amp_90_time - neg_amp_10_time)

                    # Negative decay time: spike duration from negative peak to 10% of negative peak.
                    neg_decay_time = np.abs(neg_amp_time - neg_amp_10_time)

                    print('- 10% 90% Case 3')

            # Positive 50% >> 2 sides NEEDED
            try:

                x_rise_hw, x_rise_hw_idx = find_nearest(waveform_block[:x.index(pos_amp_time) + 1], pos_amp_50)
                pos_rise_hw = x[:x.index(pos_amp_time) + 1][x_rise_hw_idx]

                x_fall_hw, x_fall_hw_idx = find_nearest(waveform_block[x.index(pos_amp_time):x.index(neg_amp_time)],
                                                        pos_amp_50)
                pos_fall_hw = x[x.index(pos_amp_time):x.index(neg_amp_time)][x_fall_hw_idx]

                print('Pos half Case 1')

            except Exception:

                # f_ = interp1d(waveform_block[:x.index(pos_amp_time)+1], x[:x.index(pos_amp_time)+1])
                # pos_rise_hw = f_(pos_amp_50)
                #
                # f_ = interp1d(waveform_block[x.index(pos_amp_time):], x[x.index(pos_amp_time):])
                # pos_fall_hw = f_(pos_amp_50)

                pos_rise_hw = interpolate.UnivariateSpline(
                    x[:x.index(pos_amp_time)+1],
                    (np.array(waveform_block[:x.index(pos_amp_time)+1]) - pos_amp_50)).roots()[-1]

                pos_fall_hw = interpolate.UnivariateSpline(
                    x[x.index(pos_amp_time):],
                    (np.array(waveform_block[x.index(pos_amp_time):]) - pos_amp_50)).roots()[0]

                print('MODIFICATION: Pos half Case 2')

            # Check a few things
            if pos_fall_hw < pos_rise_hw:
                a = pos_fall_hw
                pos_rise_hw = pos_fall_hw
                pos_fall_hw = a

            if pos_rise_hw > pos_amp_time:
                pos_rise_hw = pos_amp_time - duration / 2

            if pos_fall_hw < pos_amp_time:
                pos_fall_hw = pos_amp_time + duration / 2

            if pos_rise_hw > zero_crossing_time or pos_fall_hw > zero_crossing_time:
                pos_rise_hw = pos_amp_10_time - duration/2
                pos_fall_hw = pos_amp_10_time

            if pos_rise_hw > zero_crossing_time or pos_fall_hw > zero_crossing_time:
                pos_rise_hw = zero_crossing_time - duration / 2
                pos_fall_hw = zero_crossing_time

            posHw_duration = np.abs(pos_fall_hw - pos_rise_hw)

            # Negative 50% >> 2 sides NEEDED
            try:

                x_fall_hw, x_fall_hw_idx = find_nearest(waveform_block[x.index(pos_amp_time):x.index(neg_amp_time)],
                                                        neg_amp_50)
                neg_fall_hw = x[x.index(pos_amp_time):x.index(neg_amp_time)][x_fall_hw_idx]

                x_rise_hw, x_rise_hw_idx = find_nearest(waveform_block[x.index(neg_amp_time):], neg_amp_50)
                neg_rise_hw = x[x.index(neg_amp_time):][x_rise_hw_idx]

                # neg_fall_hw = interpolate.UnivariateSpline(
                #     x[x.index(pos_amp_time):x.index(neg_amp_time)],
                #     (np.array(waveform_block)[x.index(pos_amp_time):x.index(neg_amp_time)] - neg_amp_50)).roots()[0]
                #
                # neg_rise_hw = interpolate.UnivariateSpline(
                #     x[x.index(neg_amp_time):], (
                #             np.array(waveform_block[x.index(neg_amp_time):]) - neg_amp_50)).roots()[0]

                print('Neg half Case 1')

            except Exception:

                neg_fall_hw = interpolate.UnivariateSpline(
                    x[x.index(pos_amp_time):x.index(neg_amp_time)],
                    (np.array(waveform_block)[x.index(pos_amp_time):x.index(neg_amp_time)] - neg_amp_50)).roots()[0]

                neg_rise_hw = interpolate.UnivariateSpline(
                    x[x.index(neg_amp_time):], (
                    np.array(waveform_block[x.index(neg_amp_time):]) - neg_amp_50)).roots()[0]

                # f_ = interp1d(waveform_block[
                #               x.index(pos_amp_time):x.index(neg_amp_time)],
                #               x[x.index(pos_amp_time):x.index(neg_amp_time)],
                #               fill_value="extrapolate")
                # neg_fall_hw = f_(neg_amp_50)
                #
                # f_ = interp1d(waveform_block[x.index(neg_amp_time):],
                #               x[x.index(neg_amp_time):],
                #               fill_value="extrapolate")
                # neg_rise_hw = f_(neg_amp_50)

                print('MODIFICATION: Neg half Case 2')

            # Check a few things
            if neg_rise_hw < neg_fall_hw:
                a = neg_rise_hw
                neg_rise_hw = neg_fall_hw
                neg_fall_hw = a

            if neg_rise_hw < neg_amp_time:
                neg_rise_hw = neg_amp_time + duration / 2

            if neg_fall_hw > neg_amp_time:
                neg_fall_hw = neg_amp_time - duration / 2

            if neg_rise_hw < zero_crossing_time and neg_fall_hw < zero_crossing_time:
                neg_rise_hw = neg_amp_10_time + duration / 2
                neg_fall_hw = neg_amp_10_time

            # neg_rise_hw = neg_rise_hw - 3*duration

            negHw_duration = neg_rise_hw - neg_fall_hw

            # Restrictions on onset time after all computations
            if np.abs(zero_crossing_time - onset_time) > 2.5 and onset < pos_amp_10:
                onset_time = zero_crossing_time - duration

            # Restrictions on the positive half-width
            if pos_rise_hw < onset_time:
                pos_rise_hw = onset_time + duration / 2

            # Restrictions on the negative half-width
            # if neg_rise_hw > end_time:
            #     neg_rise_hw = end_time - duration / 2

        # Compute peak-trough ratio >> Does not depend on the peaks order
        peak_trough_ratio = round(np.abs(pos_amp / neg_amp), 3)  # Relative to zero (Jia et al)

        xs_dep, dep_values, xs_rep, rep_values, xs_rec, rec_values = None, None, None, None, None, None

        # Slopes computations
        if neg_peak_first == 1:

            # Depolarization slope
            xs_dep = x[x.index(neg_amp_time) - d:x.index(neg_amp_time)+1]
            ys_dep = waveform_block[x.index(neg_amp_time) - d:x.index(neg_amp_time)+1]
            dep_slope, dep_intercept, dep_r_value, dep_p_value, dep_std_error = stats.linregress(xs_dep, ys_dep)
            dep_values = [dep_slope * i + dep_intercept for i in xs_dep]
            dep_last = dep_values[-2]
            dep_first = dep_values[1]

            # Re-polarization slope
            xs_rep = x[x.index(neg_amp_time):x.index(pos_amp_time)-h]
            ys_rep = waveform_block[x.index(neg_amp_time):x.index(pos_amp_time)-h]
            rep_slope, rep_intercept, rep_r_value, rep_p_value, rep_std_error = stats.linregress(xs_rep, ys_rep)
            rep_values = [rep_slope * i + rep_intercept for i in xs_rep]
            rep_last = rep_values[-2]
            rep_first = rep_values[1]

            # Recovery slope
            xs_rec = x[x.index(pos_amp_time):x.index(pos_amp_time) + r]
            ys_rec = waveform_block[x.index(pos_amp_time):x.index(pos_amp_time) + r]
            rec_slope, rec_intercept, rec_r_value1, rec_p_value1, rec_std_error = stats.linregress(xs_rec, ys_rec)
            rec_values = [rec_slope * i + rec_intercept for i in xs_rec]
            rec_last = rec_values[-2]
            rec_first = rec_values[1]

        else:

            # Depolarization slope
            xs_dep = x[x.index(pos_amp_time):x.index(neg_amp_time)]
            ys_dep = waveform_block[x.index(pos_amp_time):x.index(neg_amp_time)]
            dep_slope, dep_intercept, dep_r_value, dep_p_value, dep_std_error = stats.linregress(xs_dep, ys_dep)
            dep_values = [dep_slope * i + dep_intercept for i in xs_dep]
            dep_last = dep_values[-2]
            dep_first = dep_values[1]

            # Re-polarization slope
            xs_rep = x[x.index(neg_amp_time):x.index(neg_amp_time) + 5]
            ys_rep = waveform_block[x.index(neg_amp_time):x.index(neg_amp_time) + 5]
            rep_slope, rep_intercept, rep_r_value, rep_p_value, rep_std_error = stats.linregress(xs_rep, ys_rep)
            rep_values = [rep_slope * i + rep_intercept for i in xs_rep]
            rep_last = rep_values[-2]
            rep_first = rep_values[1]

            # Recovery slope (if second maximum exists, otherwise set to 0)
            try:
                p = [p for p in points if p[0] > neg_amp_time]
                sec_pos_amp = max(p, key=itemgetter(1))[1]
                sec_pos_amp_time = max(p, key=itemgetter(1))[0]
                f_ = interp1d(waveform_block[x.index(sec_pos_amp_time):], x[x.index(sec_pos_amp_time):])

                xs_rec = x[x.index(sec_pos_amp_time):x.index(sec_pos_amp_time) + r]
                ys_rec = waveform_block[x.index(sec_pos_amp_time):x.index(sec_pos_amp_time) + r]
                rec_slope, rec_intercept, rec_r_value, rec_p_value, rec_std_error = stats.linregress(xs_rec, ys_rec)
                rec_values = [rec_slope * i + rec_intercept for i in xs_rec]
                rec_last = rec_values[-2]
                rec_first = rec_values[1]

            except ValueError:
                rec_slope = 0
                xs_rec = np.zeros(len(x[x.index(neg_amp_time)+5:]))
                rec_values = [0 for i in xs_rec]

        # Fix onset time
        if neg_peak_first == 1:

            if onset not in waveform_block:
                # onset = waveform_block[x.index(neg_amp_time)-15]
                # onset_time = x[x.index(neg_amp_time)-15]
                onset, idx = find_nearest(waveform_block[x.index(neg_amp_time)-15:x.index(neg_amp_time)], onset)
                onset_time = x[x.index(neg_amp_time)-15:x.index(neg_amp_time)][idx]

            if end not in waveform_block:
                # end = waveform_block[x.index(pos_amp_time)+15]
                # end_time = x[x.index(pos_amp_time)+15]
                end, idx = find_nearest(waveform_block[x.index(pos_amp_time):x.index(pos_amp_time)+15], onset)
                end_time = x[x.index(pos_amp_time):x.index(pos_amp_time)+15][idx]

                if end_time < xs_rec[-1]:
                    # end = waveform_block[x.index(pos_amp_time) + 25]
                    # end_time = x[x.index(pos_amp_time) + 25]
                    # end = y_fit[-3]
                    # end_time = x_fit[-3]

                    x_fit_mask = [i > xs_rec[-1] for i in x_fit]
                    new_end_time = x_fit[x_fit_mask][2]

                    end_time, end_time_idx = find_nearest(x[x.index(pos_amp_time):], new_end_time)
                    # end, idx = find_nearest(waveform_block[x.index(pos_amp_time):], end)
                    end = waveform_block[x.index(pos_amp_time):][end_time_idx]

        else:

            if onset not in waveform_block:
                # onset = waveform_block[x.index(pos_amp_time)-10]
                # onset_time = x[x.index(pos_amp_time)-10]
                onset, idx = find_nearest(waveform_block[x.index(pos_amp_time) - 15:x.index(pos_amp_time)], onset)
                onset_time = x[x.index(pos_amp_time) - 15:x.index(pos_amp_time)][idx]

            if end not in waveform_block:
                # end = waveform_block[x.index(neg_amp_time)+15]
                # end_time = x[x.index(neg_amp_time)+15]
                end, idx = find_nearest(waveform_block[x.index(neg_amp_time):x.index(neg_amp_time) + 15], end)
                end_time = x[x.index(neg_amp_time):x.index(neg_amp_time) + 15][idx]

                if end_time < xs_rec[-1]:
                    # end = waveform_block[x.index(neg_amp_time) + 30]
                    # end_time = x[x.index(neg_amp_time) + 30]
                    # end = y_fit[-3]
                    # end_time = x_fit[-3]

                    try:
                        x_fit_mask = [i > xs_rec[-1] for i in x_fit]
                        new_end_time = x_fit[x_fit_mask][2]
                        end_time, end_time_idx = find_nearest(x[x.index(neg_amp_time):], new_end_time)
                        end = waveform_block[x.index(neg_amp_time):][end_time_idx]
                    except IndexError:
                        rec_slope = 0
                        xs_rec = np.zeros(len(x[x.index(neg_amp_time) + 5:]))
                        rec_values = [0 for i in xs_rec]

        # end, idx = find_nearest(waveform_block[x.index(neg_amp_time)+40:x.index(neg_amp_time) + 75], end)
        # end_time = x[x.index(neg_amp_time)+40:x.index(neg_amp_time) + 75][idx]

        # onset, idx = find_nearest(waveform_block[x.index(neg_amp_time)-20:x.index(neg_amp_time)-10], onset)
        # onset_time = x[x.index(neg_amp_time)-20:x.index(neg_amp_time)-10][idx]

        # Plot
        # ***********************************************************************************************************
        fig, axs = plt.subplots(1, 2, figsize=(11, 5))

        # Plot some results
        axs[1].plot(x, y_samples, color='lightgrey')
        axs[1].set_ylabel("mV (norm)", size=9, color='#939799')
        axs[1].set_xlabel("ms", size=9, color='#939799')
        axs[1].set_title(f' \n \n \n Chunks {chunks_wvf} - Waveforms: {len(trn_samples_block)}  \n', fontsize=9,
                         loc='center', color='#939799')

        # Plot key points
        axs[1].plot([pos_amp_time], [pos_amp], marker='o', color='lightgrey')
        axs[1].plot([neg_amp_time], [neg_amp], marker='o', color='lightgrey')
        axs[1].plot([zero_crossing_time], [zero_crossing], marker='o', color='lightgrey')
        axs[1].plot([onset_time], [onset], marker='o', color='navy')
        axs[1].plot([pos_amp_90_time], [pos_amp_90], marker='o', color='salmon')
        axs[1].plot([pos_amp_10_time], [pos_amp_10], marker='o', color='salmon')
        axs[1].plot([neg_amp_90_time], [neg_amp_90], marker='o', color='gold')
        axs[1].plot([neg_amp_10_time], [neg_amp_10], marker='o', color='gold')
        # axs[1].plot([pos_rise_hw], [pos_amp_50], marker='x', color='navy')
        # axs[1].plot([pos_fall_hw], [pos_amp_50], marker='x', color='navy')
        # axs[1].plot([neg_fall_hw], [neg_amp_50], marker='x', color='navy')
        # axs[1].plot([neg_rise_hw], [neg_amp_50], marker='x', color='navy')

        # axs[1].plot([pos_amp_10_time, pos_amp_90_time], [0, 0], color='gray', ls='dashed')

        # Plot fall and rise "slopes"
        #axs[1].plot([zero_crossing_time, zero_crossing_time], [neg_amp_10, neg_amp_90], color='lightpink')
        #axs[1].plot([zero_crossing_time], [neg_amp_90], color='lightpink', marker='v')
        #axs[1].plot([zero_crossing_time, zero_crossing_time], [pos_amp_10, pos_amp_90], color='lightpink')
        #axs[1].plot([zero_crossing_time], [pos_amp_90], color='lightpink', marker='^')

        # Plot depolarization slope
        axs[1].plot(xs_dep, dep_values, color='darkmagenta')

        # Plot re-polarization slope
        axs[1].plot(xs_rep, rep_values, color='darkmagenta')

        # Plot recovery slope
        axs[1].plot(xs_rec, rec_values, color='darkmagenta')

        # Plot exponential fit
        axs[1].plot(x_fit[1:], y_fit[1:], color='lightpink')

        # End time
        axs[1].plot([end_time], [end], marker='o', color='navy')

        # Plot horizontal widths
        axs[1].plot([pos_rise_hw, pos_fall_hw], [pos_amp_50, pos_amp_50], color='gray', ls='dashed')
        #axs[1].plot([pos_rise_time, pos_fall_time], [pos_amp_10, pos_amp_10], color='gray', ls='dashed')
        #axs[1].plot([pos_risePk_time, pos_fallPk_time], [pos_amp_90, pos_amp_90], color='gray', ls='dashed')
        axs[1].plot([neg_fall_hw, neg_rise_hw], [neg_amp_50, neg_amp_50], color='gray', ls='dashed')
        #axs[1].plot([neg_fall_time, neg_rise_time], [neg_amp_10, neg_amp_10], color='gray', ls='dashed')
        #axs[1].plot([neg_fallPk_time, neg_risePk_time], [neg_amp_90, neg_amp_90], color='gray', ls='dashed')

        # Plot duration
        axs[1].plot([neg_amp_time, pos_amp_time], [neg_amp-0.25, neg_amp-0.25], color='dimgray')
        axs[1].plot([neg_amp_time], [neg_amp-0.25], color='dimgray', marker='>')
        axs[1].plot([pos_amp_time], [neg_amp-0.25], color='dimgray', marker='<')

        leg_wvf_features = [f'Max Amp = {pos_amp:.2f} ({maxAmp:.2f} mV) \n'
                            f'Min Amp = {neg_amp:.2f} ({minAmp:.2f} mV) \n'
                            f'Duration (Pk-tr) = {duration:.2f} ms \n'
                            f'Pos. half duration = {posHw_duration:.2f} ms \n'
                            f'Neg. half duration = {negHw_duration:.2f} ms \n'
                            f'Onset time = {onset_time:.2f} ms  \n'
                            f'End time = {end_time:.2f} ms \n'
                            f'Crossing time = {zero_crossing_time:.2f} ms \n'
                            f'Pk-tr. Ratio = {peak_trough_ratio:.2f}  \n'
                            f'Depol. slope = {dep_slope:.2f}  \n'
                            f'Repol. slope = {rep_slope:.2f}  \n'
                            f'Recov. slope = {rec_slope:.2f}  \n'
                            f'Rise time = {pos_fall_slope_time:.2f}  \n'
                            # f'Rise slope = {pos_fall_slope:.2f}  \n'
                            f'Fall time = {neg_fall_slope_time:.2f}  \n'
                            # f'Fall slope = {neg_fall_slope:.2f}  \n'
                            f'Endslope Tau = {tau:.2f}  \n'
                            ]

        leg_1 = axs[1].legend(leg_wvf_features, loc='best', frameon=False, fontsize=7)

        for text in leg_1.get_texts():
            text.set_color("gray")

        for lh in leg_1.legendHandles:
            lh.set_alpha(0)

        # Plot temporal features
        axs[0].hist(isi_block_no_clipped, bins=isi_log_bins, color='lightgray')
        axs[0].set_xscale('log')
        # axs[1, 0].set_xlim([0.1, 1000])
        axs[0].set_xlim(left=0.1)
        axs[0].set_xlabel('ms (log scale)', size=9)
        # axs[0].set_xlabel('ms', size=9)
        axs[0].set_title(f' \n \n \n Inter Spike Interval  \n Chunks {chunks}',
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

        fig.savefig(f'{paths[0]}/Sample-{sample}-unit-{unit}.png')
        print(f'{paths[0]}/Sample-{sample}-unit-{unit}.png file successfully saved!')

        unit_temporal_features = pd.DataFrame({
                                    'Unit': unit,
                                    'tf_MIFRBlockHz': mifr,
                                    'tf_MedIsi': med_isi,
                                    'tf_ModeIsi': mode_isi,
                                    'tf_Perc5Isi': prct5ISI,
                                    'tf_Entropy': entropy,
                                    'tf_CV2Mean': CV2_mean,
                                    'tf_CV2Median': CV2_median,
                                    'tf_CV': CV,
                                    'tf_Ir': IR,
                                    'tf_Lv': Lv,
                                    'tf_LvR': LvR,
                                    'tf_LcV': LcV,
                                    'tf_Si': SI,
                                    'tf_skw': skw,
                                    'wf_MaxAmpNorm': pos_amp,
                                    'wf_MaxAmp': maxAmp,
                                    'wf_MaxAmpTime': pos_amp_time,
                                    'wf_MinAmpNorm': neg_amp,
                                    'wf_MinAmp': minAmp,
                                    'wf_MinAmpTime': neg_amp_time,
                                    'wf_Duration': duration,
                                    'wf_PosHwDuration': posHw_duration,
                                    'wf_NegHwDuration': negHw_duration,
                                    'wf_Onset': onset,
                                    'wf_OnsetTime': onset_time,
                                    'wf_End': end,
                                    'wf_EndTime': end_time,
                                    'wf_Crossing': zero_crossing,
                                    'wf_CrossingTime': zero_crossing_time,
                                    'wf_Pk10': pos_amp_10,
                                    'wf_Pk10Time': pos_amp_10_time,
                                    'wf_Pk90': pos_amp_90,
                                    'wf_Pk90Time': pos_amp_90_time,
                                    'wf_Pk50': pos_amp_50,
                                    'wf_Pk50Time1': pos_rise_hw,
                                    'wf_Pk50Time2': pos_fall_hw,
                                    'wf_Tr10': neg_amp_10,
                                    'wf_Tr10Time': neg_amp_10_time,
                                    'wf_Tr90': neg_amp_90,
                                    'wf_Tr90Time': neg_amp_90_time,
                                    'wf_Tr50': neg_amp_50,
                                    'wf_Tr50Time1': neg_fall_hw,
                                    'wf_Tr50Time2': neg_rise_hw,
                                    'wf_PkTrRatio': peak_trough_ratio,
                                    'wf_DepolarizationSlope': dep_slope,
                                    'wf_RepolarizationSlope': rep_slope,
                                    'wf_RecoverySlope': rec_slope,
                                    'wf_RiseTime': pos_fall_slope_time,
                                    'wf_PosDecayTime': pos_decay_time,
                                    'wf_FallTime': neg_fall_slope_time,
                                    'wf_NegDecayTime': neg_decay_time,
                                    'wf_EndSlopeTau': tau}, index=[0])

        df_qm_temp = pd.merge(unit_quality_metrics, unit_temporal_features, left_on='Unit', right_on='Unit')
        df_qm_temp.to_csv(f'{paths[1]}/Sample-{sample}-unit-{unit}.csv', index=False)
        print(f'{paths[1]}/Sample-{sample}-unit-{unit}.csv file successfully saved!')

