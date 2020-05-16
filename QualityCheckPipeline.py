"""
Project: Neuropixels classification pipeline
Author: Gabriela Martinez
Script: QualityCheckPipeline.py
Description: implementation of quality metrics on units and chunks to choose the right ones for the next stage in the
classification pipeline.

"""

from AuxFunctions import *
import pandas as pd
from rtn.npix.gl import get_units
from rtn.npix.spk_t import trn, isi, mfr
from rtn.npix.corr import acg
from rtn.npix.spk_wvf import wvf, templates, get_peak_chan
import matplotlib.pyplot as plt
import os
import re

np.seterr(all='ignore')

data_sets = [
    # 'F:/data/PkC/18-08-30_YC001/18-08-30_YC001_probe1',  # 176
    # 'F:/data/PkC/18-08-30_YC002/18-08-30_YC002_probe1',  # 103
    # 'F:/data/PkC/18-12-18_YC010/18-12-18_YC010_probe1',  # 53
    # 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1',  # 61
    # 'F:/data/GrC/18-08-31_YC003/18-08-31_YC003_probe1',  # 122
    # 'F:/data/GrC/18-08-31_YC004/18-08-31_YC004_probe1',  # 48
    # 'F:/data/GrC/18-12-17_YC009/18-12-17_YC009_probe1',  # 155
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe1',  # 147
    # 'F:/data/GrC/19-08-14_YC015/19-08-14_YC015_probe2',  # 128
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe2',  # 148
    # 'F:/data/GrC/19-08-16_YC014/19-08-16_YC014_probe1',  # 80
    # 'F:/data/GoC/19-11-11_YC040/19-11-11_YC040_probe2',  # 0
    # 'F:/data/MFB/18-09-03_YC005/18-09-03_YC005_probe1',  # 68
    # 'F:/data/MFB/19-01-23_YC012/19-01-23_YC012_probe1',  # 185
    # 'F:/data/MFB/19-01-23_YC013/19-01-23_YC013_probe1',  #
    # 'F:/data/MFB/19-01-24_YC012/19-01-24_YC012_probe1',  # 96
    # 'F:/data/MFB/19-10-02_YC019/19-10-02_YC019_probe1',  # 43
    # 'F:/data/MFB/19-10-02_YC020/19-10-02_YC020_probe1',  # GOOD
    # 'F:/data/MFB/19-10-03_YC019/19-10-03_YC019_probe1',  # GOOD
    # 'F:/data/MFB/19-10-03_YC020/19-10-03_YC020_probe1',  # GOOD
    # 'F:/data/MLI/19-09-30_YC017/19-09-30_YC017_probe1',  # GOOD
    # 'F:/data/MLI/19-09-30_YC018/19-09-30_YC018_probe1',  # GOOD
    # 'F:/data/MLI/19-10-01_YC017/19-10-01_YC017_probe1',  # GOOD
    # 'F:/data/MLI/19-11-14_YC029/19-11-14_YC029_probe1',  # GOOD
    # 'F:/data/MLI/19-11-14_YC030/19-11-14_YC030_probe1'  # GOOD
    # 'F:/data/PkC/18-12-17_YC010/18-12-17_YC010_probe1',
    # 'F:/data/GrC/19-08-14_YC016/19-08-14_YC016_probe1',
    # 'F:/data/GrC/19-08-15_YC016/19-08-15_YC016_probe1',
    # 'F:/data/GrC/19-08-16_YC015/19-08-16_YC015_probe1',
    # 'F:/data/MFB/18-09-03_YC006/18-09-03_YC006_probe1',
    # 'F:/data/MFB/19-01-24_YC013/19-01-24_YC013_probe1',
    # 'F:/data/MLI/19-12-13_YC007/19-12-13_YC007_probe1',
    # 'Z:/npix_data/optotagging/Back-Up-YYC_2019_optotagging/data/GrC/18-12-13_YC008/18-12-13_YC008_probe1'
    # 'F:/data/GrC/19-11-04_YC036/19-11-04_YC036_probe1',
    #'F:/data/GrC/19-11-04_YC037/19-11-04_YC037_probe1',
    # 'F:/data/GrC/19-11-05_YC036/19-11-05_YC036_probe1',
    'F:/data/GrC/19-11-05_YC037/19-11-05_YC037_probe1'

]

deleteBool = True

for dp in data_sets:

    sample_probe = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}_\w{5}\d{1}", dp)[0]
    sample = re.findall(r"\d{2}-\d{2}-\d{2}_\w{2}\d{3}", dp)[0]
    print('***************************')
    print('Sample:', sample_probe)
    cell_type = dp[8:11]
    # routines_path = f'{dp}/routinesMemory'
    routines_mem = f'D:/routinesMem/{sample_probe}/routinesMemory'
    # routines_mem = f'{dp}/routinesMemory'
    print('cell type:', cell_type)
    print('routines_mem:', routines_mem)

    # Create alternative dir for routines
    if not os.path.exists(routines_mem):
        os.makedirs(routines_mem)

    # Load kilosort aux files
    amplitudes_sample = np.load(f'{dp}//amplitudes.npy')  # shape N_tot_spikes x 1
    spike_times = np.load(f'{dp}//spike_times.npy')  # in samples
    spike_clusters = np.load(f'{dp}//spike_clusters.npy')

    # Parameters
    fs = 30000
    exclusion_quantile = 0.02
    unit_size_s = 20 * 60
    unit_size_ms = unit_size_s * 1000
    chunk_size_s = 60
    chunk_size_ms = 60 * 1000
    samples_fr = unit_size_s * fs
    n_chunks = int(unit_size_s / chunk_size_s)
    c_bin = 0.2
    c_win = 400
    violations_ms = 0.8
    rpv_threshold = 0.05
    taur = 0.0015
    tauc = 0.0005
    spikes_threshold = 300
    peaks_threshold = 3
    missing_threshold = 30
    waveforms = 400
    again = False
    outliers_dev = 1.5

    # Extract good units of current sample
    good_units = get_units(dp, quality='good')
    all_units = get_units(dp)
    print("All units in sample:", len(all_units))
    print(f"Good units found in current sample: {len(good_units)} --> {good_units}")
    n = 0
    x = 0

    fp_block_fancy = []
    fp_block_regular = []
    l_units = []
    total = 0

    for unit in good_units:

        l_chunk_list = []
        l_chunk_len = []
        l_chunk_waveform_peak_channel = []
        l_norm_chunk_waveform_peak_channel = []
        l_ms_norm_chunk_waveform_peak_channel = []
        l_trn_samples_chunk = []
        l_trn_ms_chunk = []
        l_trn_s_chunk = []
        l_spikes_chunk = []
        l_chunk_amplitudes = []
        l_chunk_spikes_missing = []
        l_good_chunks = []

        print(f'Unit >>>> {unit}')

        # Unit spikes during first 20 minutes
        trn_samples_unit_20 = trn(dp, unit=unit, subset_selection=[(0, unit_size_s)], enforced_rp=0.5, again=again)
        trn_ms_unit_20 = trn_samples_unit_20 * 1. / (fs * 1. / 1000)
        spikes_unit_20 = len(trn_ms_unit_20)

        # Extract peak channel of current unit (where the deflection is maximum)
        peak_channel = get_peak_chan(dp, unit)

        if spikes_unit_20 != 0 and spikes_unit_20 > spikes_threshold:

            l_units.append(unit)

            # Create a local path to store this unit info
            paths = [f"Images/Pipeline/Sample_{sample_probe}/GoodUnits/Files",
                     f"Images/Pipeline/Sample_{sample_probe}/GoodUnits/Images",
                     f"Images/Pipeline/Sample_{sample_probe}/BadUnits/Files",
                     f"Images/Pipeline/Sample_{sample_probe}/BadUnits/Images"
                     ]

            for path in paths:
                if not os.path.exists(path):
                    os.makedirs(path)

            # Unit amplitudes
            amplitudes_unit = amplitudes_sample[spike_clusters == unit]
            spike_times_unit = spike_times[spike_clusters == unit]
            unit_mask_20 = (spike_times_unit <= samples_fr)
            spike_times_unit_20 = spike_times_unit[unit_mask_20]
            amplitudes_unit_20 = amplitudes_unit[unit_mask_20]

            # Unit ACG
            block_ACG, x_block, y_block, y_lim1_unit, yl_unit, y_lim2_unit, rpv_ratio_acg = None, None, None, None, None, None, None

            try:
                block_ACG = acg(dp, unit, c_bin, c_win, subset_selection=[(0, unit_size_s)])
                x_block = np.linspace(-c_win * 1. / 2, c_win * 1. / 2, block_ACG.shape[0])
                y_block = block_ACG.copy()
                y_lim1_unit = 0
                yl_unit = max(block_ACG)
                y_lim2_unit = int(yl_unit) + 5 - (yl_unit % 5)

                # Find refractory period violations
                booleanCond = np.zeros(len(y_block), dtype=np.bool)
                booleanCond[(x_block >= -violations_ms) & (x_block <= violations_ms)] = True
                violations = y_block[booleanCond]
                violations_mean = np.mean(violations)

                # Select normal refractory points in the ACG
                booleanCond2 = np.zeros(len(y_block), dtype=np.bool)
                booleanCond2[:80] = True
                booleanCond2[-80:] = True
                normal_obs = y_block[booleanCond2]
                normal_obs_mean = np.mean(normal_obs)

                # Compute ACG ratio to account for refractory violations
                rpv_ratio_acg = round(violations_mean/normal_obs_mean, 2)

            except ValueError:
                print("ACG could not be computed!")

            # Unit template
            # unit_template = np.mean(templates(dp, unit)[:, :, peak_channel], axis=0)
            # unit_template_norm = range_normalization(unit_template)

            # >> CHUNKS PROCESSING <<
            num_block_spikes = 0
            block_time = 0
            deepest_deflections = []

            for i in range(n_chunks):

                try:

                    # Chunk mask for times
                    chunk_mask = (i * chunk_size_s * fs <= spike_times_unit_20) & \
                                 (spike_times_unit_20 < (i + 1) * chunk_size_s * fs)
                    chunk_mask = chunk_mask.reshape(len(spike_times_unit_20), )

                    # Chunk amplitudes
                    amplitudes_chunk = amplitudes_unit_20[chunk_mask]
                    amplitudes_chunk = np.asarray(amplitudes_chunk, dtype='float64')

                    # Estimate optimal number of bins for Gaussian fit to amplitudes
                    chunk_bins = estimate_bins(amplitudes_chunk, rule='Fd')

                    # % of missing spikes per chunk
                    x_c, p0_c, min_amp_c, n_fit_c, n_fit_no_cut_c, chunk_spikes_missing = gaussian_amp_est(amplitudes_chunk, chunk_bins)

                    # If chunk % of missing spikes is less than 30%, we can proceed
                    if (~np.isnan(chunk_spikes_missing)) & (chunk_spikes_missing <= 30):

                        l_good_chunks.append(i)
                        l_chunk_spikes_missing.append(chunk_spikes_missing)
                        l_chunk_amplitudes.append(amplitudes_chunk)

                        # Chunk length and times in seconds
                        chunk_start_time = i * chunk_size_s
                        chunk_end_time = (i + 1) * chunk_size_s
                        chunk_len = (chunk_start_time, chunk_end_time)
                        l_chunk_len.append(chunk_len)

                        # Chunk mean waveform around peak channel
                        chunk_waveform_peak_channel = np.mean(
                            wvf(dp, unit,
                                n_waveforms=waveforms,
                                t_waveforms=82,
                                subset_selection=[(chunk_start_time, chunk_end_time)],
                                again=again,
                                ignore_nwvf=False)
                            [:, :, peak_channel], axis=0)
                        l_chunk_waveform_peak_channel.append(chunk_waveform_peak_channel)

                        # Find deepest deflection in the center of the waveform
                        chunk_deepest_deflection = min(chunk_waveform_peak_channel[30:49])
                        deepest_deflections.append(chunk_deepest_deflection)

                        # Chunk spikes
                        trn_samples_chunk = trn(dp,
                                                unit=unit,
                                                subset_selection=[(chunk_start_time, chunk_end_time)],
                                                enforced_rp=0.5)

                        l_trn_samples_chunk.append(trn_samples_chunk)
                        trn_ms_chunk = trn_samples_chunk * 1. / (fs * 1. / 1000)
                        l_trn_ms_chunk.append(trn_ms_chunk)

                        trn_s_chunk = trn_samples_chunk * 1. / fs
                        l_trn_s_chunk.append(trn_s_chunk)

                        spikes_chunk = len(trn_s_chunk)
                        l_spikes_chunk.append(spikes_chunk)

                        num_block_spikes += spikes_chunk
                        block_time += chunk_size_s

                except Exception:
                    continue

            try:
                # Put together all the good chunks!

                # Find deepest deflection chunk
                block_deepest_deflection = min(deepest_deflections)
                block_deepest_deflection_index = deepest_deflections.index(block_deepest_deflection)
                print('Deepest block', block_deepest_deflection_index)
                deepest_chunk = l_good_chunks[block_deepest_deflection_index]

                # Find if neighbors are good chunks
                chunk_before, chunk_after = find_neighbors(deepest_chunk)
                print('chunk_before', chunk_before)
                print('chunk_after', chunk_after)

                # Compute mean waveform from neighbors
                wvf_block, chunks_for_wvf = closest_waveforms(chunk_before, deepest_chunk, chunk_after, l_good_chunks,
                                                              l_chunk_waveform_peak_channel)

                wvf_block_norm = range_normalization(wvf_block)

                # Join trn samples
                trn_samples_block = np.concatenate(l_trn_samples_chunk).ravel()
                # trn_seconds_block = trn_samples_block * 1. / fs
                trn_milliseconds_block = trn_samples_block * 1. / (fs * 1. / 1000)
                spikes_block = len(trn_milliseconds_block)

                # Join amplitudes
                amplitudes_block_no_clipped = np.concatenate(l_chunk_amplitudes).ravel()
                amplitudes_block_clipped = remove_outliers(amplitudes_block_no_clipped, exclusion_quantile)

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

                # >> Compute fraction of contamination for the block << Everything in seconds!!
                rpv_block, fp_block = rvp_and_fp(np.diff(trn_samples_block)/30000,
                                                 N=num_block_spikes,
                                                 T=block_time,
                                                 taur=taur,
                                                 tauc=tauc)
                fp_block_fancy.append(fp_block)

                # >> Compute version 2 of fraction of contamination for the block <<
                fp_block_2 = round(rpv_block/num_block_spikes, 2)
                fp_block_regular.append(fp_block_2)

                # >> Waveform analysis <<
                # Biggest peak detection
                big_peak_neg_block = detect_biggest_peak(wvf_block)

                # Detect peaks
                xs_block, ys_block, count_wvf_peaks, count_wvf_center_peaks = detect_peaks(wvf_block, outliers_dev=outliers_dev)
                wvf_peaks_block_dummy = 1 if 1 <= count_wvf_center_peaks <= peaks_threshold else 0

                # Coefficient of variation
                # iqr_ = iqr(wvf_block) * outliers_dev
                # med_ = np.median(wvf_block)
                # coef_of_var = round(iqr_ / med_, 2)
                # q1 = np.percentile(wvf_block, 25)
                # q3 = np.percentile(wvf_block, 75)
                # dispersion = round((q3-q1)/(q3+q1), 2)

                # Cosine similarity
                # cosine_template_block = cosine_similarity(unit_template_norm, wvf_block_norm)

                # Block amplitudes
                amplitudes_block_bins = estimate_bins(amplitudes_block_no_clipped, rule='Fd')

                # Block Gaussian fit for plotting purposes
                x_b, p0_b, min_amp_b, n_fit_b, n_fit_no_cut_b, block_percent_missing = \
                    gaussian_amp_est(amplitudes_block_no_clipped, amplitudes_block_bins)

                # The real % of missing spikes of the block will be the average of the chunks
                block_spikes_missing = int(np.nanmean(l_chunk_spikes_missing))

                # Mean Firing Rate >> With clipped ISI
                mfr_block = mean_firing_rate(isi_block_clipped)

                # Mean Amplitude
                ma_block = mean_amplitude(amplitudes_block_clipped)

                # Chunks
                chunks_in_block = len(l_good_chunks)
                chunks_in_waveform = len(chunks_for_wvf)

                # If this block has a good fraction of contamination and good waveform, we can proceed
                if (rpv_ratio_acg <= rpv_threshold) & (big_peak_neg_block == 1) & (wvf_peaks_block_dummy == 1):

                    print('Checkpoint passed')
                    x += 1

                    # *************************************************************************************
                    # Graphics!!

                    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
                    fig.suptitle(f'Sample {sample}, '
                                 f'unit {unit} - Considered chunks: {l_good_chunks} \n'
                                 f'Total spikes: {spikes_unit_20} - Waveform detected peaks: {count_wvf_peaks}  \n',
                                 y=0.98, fontsize=10, color='#939799')

                    # [0, 0]
                    x1 = list(np.arange(len(wvf_block), step=10))
                    x2 = [round(i/30, 2) for i in x1]  # Samples to ms
                    plt.sca(axs[0, 0])

                    axs[0, 0].plot(wvf_block, color='gold')
                    plt.xticks(x1, x2)
                    axs[0, 0].set_ylabel("mV", size=9, color='#939799')
                    axs[0, 0].set_xlabel("ms", size=9, color='#939799')
                    axs[0, 0].scatter(xs_block, ys_block, marker='v', c='salmon')
                    axs[0, 0].set_title(f' \n \n \n Mean block waveform - {chunks_for_wvf}  \n',
                                        fontsize=9, loc='center', color='#939799')

                    # [0, 1]
                    if all(v is not None for v in [x_b, n_fit_no_cut_b, n_fit_b]):
                        axs[0, 1].hist(amplitudes_block_no_clipped,
                                       bins=amplitudes_block_bins,
                                       orientation='vertical',
                                       color='lightgray')
                        axs[0, 1].plot(x_b, n_fit_no_cut_b, color='silver')
                        axs[0, 1].plot(x_b, n_fit_b, color='gold')
                        axs[0, 1].set_ylabel("# of spikes", size=9, color='#939799')
                        leg_dtr_0_1 = [f'Spikes missing: {block_spikes_missing}%  \n'
                                       f'Mean Amplitude: {ma_block}']

                        axs[0, 1].axvline(p0_b[3], color='salmon', ls='--', lw=2, ymax=min_amp_b / 2, label='MS')
                        leg_0_1 = axs[0, 1].legend(leg_dtr_0_1, loc='best', frameon=False, fontsize=9)
                        for text in leg_0_1.get_texts():
                            text.set_color("gray")
                        for lh in leg_0_1.legendHandles:
                            lh.set_alpha(0)
                    else:
                        axs[0, 1].hist(amplitudes_block_no_clipped,
                                       bins=amplitudes_block_bins,
                                       orientation='vertical',
                                       color='lightgray')

                    axs[0, 1].set_title(
                        f' \n \n \n Amplitude (Gaussian fit)', fontsize=9, loc='center', color='#939799')

                    # [1, 0]
                    if all(v is not None for v in [rpv_ratio_acg]):
                        try:
                            axs[1, 0].hist(isi_block_no_clipped, bins=isi_log_bins, color='lightgray')
                            axs[1, 0].set_xscale('log')
                            # axs[1, 0].set_xlim([0.1, 1000])
                            axs[1, 0].set_xlim(left=0.1)
                            axs[1, 0].set_xlabel('Inter Spike Interval [ms] (log scale)')
                            leg_line_mfr = [f'RPV = {rpv_block}  \n'
                                            f'RPV/Spikes = {fp_block_2}  \n' 
                                            f'ACG ratio= {rpv_ratio_acg}  \n'
                                            f'MFR = {mfr_block} Hz \n']
                            leg_1_0 = axs[1, 0].legend(leg_line_mfr, loc='best', frameon=False, fontsize=9)

                            for text in leg_1_0.get_texts():
                                text.set_color("gray")

                            for lh in leg_1_0.legendHandles:
                                lh.set_alpha(0)

                        except Exception:
                            axs[1, 0].hist(isi_block_no_clipped, bins=isi_bins, color='lightgray')
                            axs[1, 0].set_xlabel('Inter Spike Interval [ms]')
                            leg_line_mfr = [f'RPV = {rpv_block}  \n'
                                            f'RPV/Spikes = {fp_block_2}  \n'
                                            f'ACG ratio= {rpv_ratio_acg}  \n'
                                            f'MFR = {mfr_block} Hz \n']
                            leg_1_0 = axs[1, 0].legend(leg_line_mfr, loc='best', frameon=False, fontsize=9)

                            for text in leg_1_0.get_texts():
                                text.set_color("gray")

                            for lh in leg_1_0.legendHandles:
                                lh.set_alpha(0)

                    # [1, 1]
                    if all(v is not None for v in [block_ACG, x_block, y_block, y_lim1_unit, yl_unit, y_lim2_unit]):
                        axs[1, 1].bar(x=x_block, height=y_block, width=0.2, color='salmon', bottom=y_lim1_unit)
                        axs[1, 1].set_ylim([y_lim1_unit, y_lim2_unit])
                        axs[1, 1].set_ylabel("Autocorrelation (Hz)", size=9, color='#939799')
                        axs[1, 1].set_xlabel("Time (ms)", size=9)
                        axs[1, 1].set_title(f' \n \n Autocorrelogram', fontsize=9, fontname="DejaVu Sans", loc='center',
                                            color='#939799')

                    # FORMATTING
                    format_plot(axs[0, 0])
                    format_plot(axs[0, 1])
                    format_plot(axs[1, 0])
                    format_plot(axs[1, 1])
                    fig.tight_layout()
                    plt.show()

                    fig.savefig(f"{paths[1]}/Sample-{sample}-unit-{unit}.png")

                    # *************************************************************************************
                    # Dataframe!!

                    df = pd.DataFrame(
                        {'Sample': sample,  # 19-01-16_YC011
                         'Unit': unit,  # 20
                         'PeakChUnit': peak_channel,  # 54
                         'SpikesUnit': spikes_unit_20,  # 135684
                         'GoodChunks': chunks_in_block,  # 19
                         'GoodChunksList': [l_good_chunks],  # [0,1,2,3,4,5,...19]
                         'WvfChunks': chunks_in_waveform,  # 3
                         'WvfChunksList': [chunks_for_wvf],  # [0,1,2]
                         'SpikesBlock': spikes_block,  # 3147
                         'PercMissingSpikesBlock': block_spikes_missing,  # 7%
                         'MFRBlockHz': mfr_block,  # 80.25
                         'MeanAmpBlock': ma_block,  # 12.5
                         'RPVBlock': rpv_block,  # 5
                         'FpBlockFancy': fp_block,  # 0.04
                         'FpBlockSimple': fp_block_2,  # 0.01
                         'ACGViolationRatio': rpv_ratio_acg,  # 0.02
                         'WvfPeaksBlock': count_wvf_peaks,  # 2
                         'PeaksBlock_x': [xs_block],
                         'PeaksBlock_y': [ys_block],
                         'BigPkNegBlock': big_peak_neg_block  # 1/0
                         #'SpikeSamples': [trn_samples_block.tolist()],
                         #'SpikeSeconds': [trn_seconds_block],
                         #'SpikeMilliseconds': [trn_milliseconds_block],
                         #'MeanWaveform': [wvf_block],
                         #'ISIHist': [isi_block_no_clipped],
                         #'Amplitudes': [amplitudes_block_no_clipped],
                         }, index=[0]
                    )

                    df.to_csv(f'{paths[0]}/Sample-{sample}-unit-{unit}.csv', index=False)
                    print(f'Summary-sample-{sample}-unit-{unit}.csv  ---> Successfully created!')

                else:

                    print('Bad block detected...')

                    # *************************************************************************************
                    # Graphics!!

                    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
                    fig.suptitle(f'Sample {sample}, '
                                 f'unit {unit} - Considered chunks: {l_good_chunks} \n'
                                 f'Total spikes: {spikes_unit_20} - Waveform detected peaks: {count_wvf_peaks}  \n',
                                 y=0.98, fontsize=10, color='#939799')

                    # [0, 0]
                    x1 = list(np.arange(len(wvf_block), step=10))
                    x2 = [round(i / 30, 2) for i in x1]  # Samples to ms
                    plt.sca(axs[0, 0])
                    axs[0, 0].plot(wvf_block, color='salmon')
                    plt.xticks(x1, x2)
                    axs[0, 0].set_ylabel("mV (norm)", size=9, color='#939799')
                    axs[0, 0].set_xlabel("ms", size=9, color='#939799')
                    axs[0, 0].scatter(xs_block, ys_block, marker='v', c='salmon')
                    axs[0, 0].set_title(f' \n \n \n Mean block waveform - {chunks_for_wvf}  \n',
                                        fontsize=9, loc='center', color='#939799')

                    # [0, 1]
                    if all(v is not None for v in [x_b, n_fit_no_cut_b, n_fit_b]):
                        axs[0, 1].hist(amplitudes_block_no_clipped,
                                       bins=amplitudes_block_bins,
                                       orientation='vertical',
                                       color='lightgray')
                        axs[0, 1].plot(x_b, n_fit_no_cut_b, color='silver')
                        axs[0, 1].plot(x_b, n_fit_b, color='gold')
                        axs[0, 1].set_ylabel("# of spikes", size=9, color='#939799')
                        leg_dtr_0_1 = [f'Spikes missing: {block_spikes_missing}%  \n'
                                       f'Mean Amplitude: {ma_block}']

                        axs[0, 1].axvline(p0_b[3], color='salmon', ls='--', lw=2, ymax=min_amp_b / 2, label='MS')
                        leg_0_1 = axs[0, 1].legend(leg_dtr_0_1, loc='best', frameon=False, fontsize=9)
                        for text in leg_0_1.get_texts():
                            text.set_color("gray")
                        for lh in leg_0_1.legendHandles:
                            lh.set_alpha(0)
                    else:
                        axs[0, 1].hist(amplitudes_block_no_clipped,
                                       bins=amplitudes_block_bins,
                                       orientation='vertical',
                                       color='lightgray')

                    axs[0, 1].set_title(
                        f' \n \n \n Amplitude (Gaussian fit)', fontsize=9, loc='center', color='#939799')

                    # [1, 0]
                    if all(v is not None for v in [rpv_ratio_acg]):
                        try:
                            axs[1, 0].hist(isi_block_no_clipped, bins=isi_log_bins, color='lightgray')
                            axs[1, 0].set_xscale('log')
                            # axs[1, 0].set_xlim([0.1, 1000])
                            axs[1, 0].set_xlim(left=0.1)
                            axs[1, 0].set_xlabel('Inter Spike Interval [ms] (log scale)')
                            leg_line_mfr = [f'RPV = {rpv_block}  \n'
                                            f'RPV/Spikes = {fp_block_2}  \n'
                                            f'ACG ratio= {rpv_ratio_acg}  \n'
                                            f'MFR = {mfr_block} Hz \n']
                            leg_1_0 = axs[1, 0].legend(leg_line_mfr, loc='best', frameon=False, fontsize=9)

                            for text in leg_1_0.get_texts():
                                text.set_color("gray")

                            for lh in leg_1_0.legendHandles:
                                lh.set_alpha(0)

                        except Exception:
                            axs[1, 0].hist(isi_block_no_clipped, bins=isi_bins, color='lightgray')
                            axs[1, 0].set_xlabel('Inter Spike Interval [ms]')
                            leg_line_mfr = [f'RPV = {rpv_block}  \n'
                                            f'RPV/Spikes = {fp_block_2}  \n'
                                            f'ACG ratio= {rpv_ratio_acg}  \n'
                                            f'MFR = {mfr_block} Hz \n']
                            leg_1_0 = axs[1, 0].legend(leg_line_mfr, loc='best', frameon=False, fontsize=9)

                            for text in leg_1_0.get_texts():
                                text.set_color("gray")

                            for lh in leg_1_0.legendHandles:
                                lh.set_alpha(0)

                                # [1, 1]
                    if all(v is not None for v in [block_ACG, x_block, y_block, y_lim1_unit, yl_unit, y_lim2_unit]):
                        axs[1, 1].bar(x=x_block, height=y_block, width=0.2, color='salmon', bottom=y_lim1_unit)
                        axs[1, 1].set_ylim([y_lim1_unit, y_lim2_unit])
                        axs[1, 1].set_ylabel("Autocorrelation (Hz)", size=9, color='#939799')
                        axs[1, 1].set_xlabel("Time (ms)", size=9)
                        axs[1, 1].set_title(f' \n \n Autocorrelogram', fontsize=9, fontname="DejaVu Sans", loc='center',
                                            color='#939799')

                    # FORMATTING
                    format_plot(axs[0, 0])
                    format_plot(axs[0, 1])
                    format_plot(axs[1, 0])
                    format_plot(axs[1, 1])
                    fig.tight_layout()
                    plt.show()

                    fig.savefig(f"{paths[3]}/Sample-{sample}-unit-{unit}.png")

                    # Dataframe!!

                    df2 = pd.DataFrame(
                        {'Sample': sample,  # 19-01-16_YC011
                         'Unit': unit,  # 20
                         'PeakChUnit': peak_channel,  # 54
                         'SpikesUnit': spikes_unit_20,  # 135684
                         'GoodChunks': chunks_in_block,  # 19
                         'GoodChunksList': [l_good_chunks],  # [0,1,2,3,4,5,...19]
                         'WvfChunks': chunks_in_waveform,  # 3
                         'WvfChunksList': [chunks_for_wvf],  # [0,1,2]
                         'SpikesBlock': spikes_block,  # 3147
                         'PercMissingSpikesBlock': block_spikes_missing,  # 7%
                         'MFRBlockHz': mfr_block,  # 80.25
                         'MeanAmpBlock': ma_block,  # 12.5
                         'RPVBlock': rpv_block,  # 5
                         'FpBlockFancy': fp_block,  # 0.04
                         'FpBlockSimple': fp_block_2,  # 0.01
                         'ACGViolationRatio': rpv_ratio_acg,  # 0.02
                         'WvfPeaksBlock': count_wvf_peaks,  # 2
                         'PeaksBlock_x': [xs_block],
                         'PeaksBlock_y': [ys_block],
                         'BigPkNegBlock': big_peak_neg_block  # 1/0
                         # 'SpikeSamples': [trn_samples_block.tolist()],
                         # 'SpikeSeconds': [trn_seconds_block],
                         # 'SpikeMilliseconds': [trn_milliseconds_block],
                         # 'MeanWaveform': [wvf_block],
                         # 'ISIHist': [isi_block_no_clipped],
                         # 'Amplitudes': [amplitudes_block_no_clipped],
                         }, index=[0]
                    )

                    df2.to_csv(f'{paths[2]}/Sample-{sample}-unit-{unit}.csv', index=False)
                    print(f'Summary-sample-{sample}-unit-{unit}.csv  ---> Successfully created!')

            except ValueError:
                print(f'All the chunks in {unit} missing lots of spikes!')

        n += 1
        print(f'{n} processed of {len(good_units)}')
        print(f'{x} good units out of {len(good_units)}')

    # delete_routines(deleteBool, routines_path)

