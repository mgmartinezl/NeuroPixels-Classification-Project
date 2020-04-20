"""
Project: Neuropixels classification pipeline
Author: Gabriela Martinez
Script: Quality.py
Description: implementation of quality metrics on units and chunks to choose the right ones for the next stage in the
classification pipeline.

"""

from AuxFunctions import *
import pandas as pd
from rtn.npix.gl import get_units
from rtn.npix.spk_t import trn
from rtn.npix.corr import acg
from rtn.npix.spk_wvf import wvf, templates, get_peak_chan
from seaborn import distplot
import matplotlib.pyplot as plt
import os

np.random.seed(42)
np.seterr(all='ignore')

# Sample for testing
dp = 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1'
sample = dp[12:26]
cell_type = dp[8:11]

# Load kilosort aux files
amplitudes_sample = np.load(f'{dp}//amplitudes.npy')  # shape N_tot_spikes x 1
spike_times = np.load(f'{dp}//spike_times.npy')  # in samples
spike_clusters = np.load(f'{dp}//spike_clusters.npy')

# Parameters
fs = 30000
exclusion_quantile = 0.01
unit_size_s = 20 * 60
unit_size_ms = unit_size_s * 1000
chunk_size_s = 60
chunk_size_ms = 60 * 1000
samples_fr = unit_size_s * fs
n_chunks = int(unit_size_s / chunk_size_s)
c_bin = 0.2
c_win = 80
fp_threshold = 5  # It is already a %
spikes_threshold = 300
peaks_threshold = 3
missing_threshold = 30

# Extract good units of current sample
good_units = get_units(dp, quality='good')
all_units = get_units(dp)
print("All units in sample:", len(all_units))
print(f"Good units found in current sample: {len(good_units)} --> {good_units}")
# units with problems... [95]

good_units = [18, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38,
              39, 40, 41, 42, 43, 44, 47, 49, 52, 53, 54, 55, 64, 67, 68, 69, 73, 76,
              78, 81, 83, 85, 87, 88, 93, 98, 99, 101, 102, 107, 108, 110, 112, 113,
              120, 121, 122, 154, 160, 192, 204, 234]

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

    print(f'Unit >>>> {unit}')

    # >> UNIT SPIKES DURING THE FIRST 20 MINUTES <<
    trn_samples_unit_20 = trn(dp, unit=unit, subset_selection=[(0, unit_size_s)], enforced_rp=0.5)
    trn_ms_unit_20 = trn_samples_unit_20 * 1. / (fs * 1. / 1000)
    spikes_unit_20 = len(trn_ms_unit_20)

    # Extract peak channel of current unit (where the deflection is maximum)
    peak_channel = get_peak_chan(dp, unit)

    if spikes_unit_20 != 0 and spikes_unit_20 > spikes_threshold:

        # Create a local path to store this unit info
        unit_path = f"Images/Pipeline/Sample_{sample}/Unit_{unit}/"
        sample_path_good = f"Images/Pipeline/Sample_{sample}/Pipeline2/GoodUnits/"
        sample_path_bad = f"Images/Pipeline/Sample_{sample}/Pipeline2/BadUnits/"

        if not os.path.exists(unit_path):
            os.makedirs(unit_path)

        # Unit amplitudes
        amplitudes_unit = amplitudes_sample[spike_clusters == unit]
        spike_times_unit = spike_times[spike_clusters == unit]
        unit_mask_20 = (spike_times_unit <= samples_fr)
        spike_times_unit_20 = spike_times_unit[unit_mask_20]
        amplitudes_unit_20 = amplitudes_unit[unit_mask_20]

        # >> CHUNKS PROCESSING <<
        num_block_spikes = 0
        block_time = 0

        for i in range(n_chunks):

            chunk_mask = (i * chunk_size_s * fs <= spike_times_unit_20) & \
                         (spike_times_unit_20 < (i + 1) * chunk_size_s * fs)
            chunk_mask = chunk_mask.reshape(len(spike_times_unit_20), )

            # Chunk amplitudes
            amplitudes_chunk = amplitudes_unit_20[chunk_mask]

            # Chunk amplitudes [normalized]
            norm_amplitudes_chunk = range_normalization(amplitudes_chunk)
            amplitudes_chunk = np.asarray(amplitudes_chunk, dtype='float64')

            # Estimate optimal number of bins for Gaussian fit to amplitudes
            chunk_bins = estimate_bins(amplitudes_chunk, rule='Fd')

            # % of missing spikes per chunk
            x_c, p0_c, min_amp_c, n_fit_c, n_fit_no_cut_c, chunk_spikes_missing = gaussian_amp_est(amplitudes_chunk,
                                                                                                   chunk_bins)

            # If chunk % of missing spikes is less than 30%, we can proceed
            if (~np.isnan(chunk_spikes_missing)) & (chunk_spikes_missing <= 30):
            # if (np.isnan(chunk_spikes_missing)) | (chunk_spikes_missing > 30):

                print(f'Chunk {i} -- Filter 1 passed!')

                l_chunk_spikes_missing.append(chunk_spikes_missing)
                l_chunk_amplitudes.append(amplitudes_chunk)

                # Chunk length and times in seconds
                chunk_start_time = i * chunk_size_s
                chunk_end_time = (i + 1) * chunk_size_s
                chunk_len = (chunk_start_time, chunk_end_time)
                l_chunk_len.append(chunk_len)

                # Chunk mean waveform around peak channel
                chunk_waveform_peak_channel = np.mean(
                    wvf(dp, unit, t_waveforms=82, subset_selection=[(chunk_start_time, chunk_end_time)])
                    [:, :, peak_channel], axis=0)
                l_chunk_waveform_peak_channel.append(chunk_waveform_peak_channel)

                # Chunk mean waveform around peak channel [normalized]
                norm_chunk_waveform_peak_channel = range_normalization(chunk_waveform_peak_channel)
                l_norm_chunk_waveform_peak_channel.append(norm_chunk_waveform_peak_channel)

                # Chunk mean waveform around peak channel [mean subtraction]
                ms_norm_chunk_waveform_peak_channel = waveform_mean_subtraction(norm_chunk_waveform_peak_channel)
                l_ms_norm_chunk_waveform_peak_channel.append(ms_norm_chunk_waveform_peak_channel)

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
                l_chunk_list.append(i)

            else:
                print(f'Chunk {i} -- Filter 1 NOT passed!')

        try:
            # Put together!

            # Join trn samples
            trn_samples_block = np.concatenate(l_trn_samples_chunk).ravel()
            trn_seconds_block = trn_samples_block * 1. / fs
            trn_milliseconds_block = trn_samples_block * 1. / (fs * 1. / 1000)
            spikes_block = len(trn_milliseconds_block)

            # Join amplitudes
            amplitudes_block_no_clipped = np.concatenate(l_chunk_amplitudes).ravel()
            amplitudes_block_clipped = remove_outliers(amplitudes_block_no_clipped, exclusion_quantile)

            # Compute Inter-Spike Interval for the block of good chunks
            # We need to exclude the times between chunks with the exclusion_quantile
            isi_block_clipped = compute_isi(trn_seconds_block, quantile=exclusion_quantile)
            isi_block_no_clipped = compute_isi(trn_seconds_block)

            # Estimate number of optimal bins for ISI
            isi_bins = estimate_bins(isi_block_no_clipped, rule='Sqrt')
            isi_log_bins = np.geomspace(isi_block_no_clipped.min(), isi_block_no_clipped.max(), isi_bins)

            # >> Compute fraction of contamination for the block << Everything in seconds!!
            rpv_block, fp_block = rvp_and_fp(isi_block_no_clipped, N=num_block_spikes, T=block_time)

            # >> Waveform analysis <<
            # Join waveforms and compute the mean
            mean_ms_norm_wvf_block = np.mean(l_ms_norm_chunk_waveform_peak_channel, axis=0)

            # >> MEAN BLOCK WAVEFORM BIGGEST PEAK DETECTION <<
            big_peak_neg_block = detect_biggest_peak(mean_ms_norm_wvf_block)

            # >> MEAN BLOCK WAVEFORM PEAK DETECTION <<
            xs_block, ys_block, wvf_peaks_block = detect_peaks(mean_ms_norm_wvf_block)
            wvf_peaks_block_dummy = 1 if wvf_peaks_block <= peaks_threshold else 0

            # If this block has a good fraction of contamination and good waveform, we can proceed
            if (~np.isnan(fp_block)) & (fp_block < 0.1) & (big_peak_neg_block == 1) & (wvf_peaks_block_dummy == 1):
            # if (np.isnan(fp_block)) | (fp_block > 0.1) | (big_peak_neg_block == 0) | (wvf_peaks_block_dummy == 0):

                # Block amplitudes
                amplitudes_block_bins = estimate_bins(amplitudes_block_no_clipped, rule='Fd')

                # Block Gaussian fit for plotting purposes
                x_b, p0_b, min_amp_b, n_fit_b, n_fit_no_cut_b, block_percent_missing = \
                    gaussian_amp_est(amplitudes_block_no_clipped, amplitudes_block_bins)

                # The real % of missing spikes of the block will be the average of the chunks
                block_spikes_missing = int(np.mean(l_chunk_spikes_missing))

                # Mean Firing Rate >> With clipped ISI
                mfr_block = mean_firing_rate(isi_block_clipped)

                # Mean Amplitude
                ma_block = mean_amplitude(amplitudes_block_clipped)

                # Chunks
                strings = [str(integer) for integer in l_chunk_list]
                strings = ', '.join(strings)
                chunks_in_block = len(l_chunk_list)

                # *************************************************************************************
                # Graphics!!

                fig, axs = plt.subplots(2, 2, figsize=(10, 7))
                fig.suptitle(f'Sample {sample}, '
                             f'unit {unit} - Good chunks: {l_chunk_list} \n'
                             # f'unit {unit} - Bad chunks: {l_chunk_list} \n'
                             f'Total spikes: {spikes_unit_20} - Waveform detected peaks: {wvf_peaks_block}  \n',
                             y=0.98, fontsize=10, color='#939799')

                # [0, 0]
                axs[0, 0].plot(mean_ms_norm_wvf_block, color='lightgray')
                axs[0, 0].scatter(xs_block, ys_block, marker='v', c='salmon')
                axs[0, 0].set_title(f' \n \n \n Mean block waveform ',
                                    fontsize=9, loc='center', color='#939799')

                # [0, 1]
                axs[0, 1].hist(amplitudes_block_no_clipped,
                               bins=amplitudes_block_bins,
                               orientation='vertical',
                               color='lightgray')
                axs[0, 1].plot(x_b, n_fit_no_cut_b, color='silver')
                axs[0, 1].plot(x_b, n_fit_b, color='gold')
                axs[0, 1].set_ylabel("# of spikes", size=9, color='#939799')
                leg_dtr_0_1 = [f'Spikes missing: {block_spikes_missing}%']
                axs[0, 1].axvline(p0_b[3], color='salmon', ls='--', lw=2, ymax=min_amp_b / 2, label='MS')
                leg_0_1 = axs[0, 1].legend(leg_dtr_0_1, loc='best', frameon=False, fontsize=9)
                for text in leg_0_1.get_texts():
                    text.set_color("gray")
                for lh in leg_0_1.legendHandles:
                    lh.set_alpha(0)

                axs[0, 1].set_title(
                    f' \n \n \n Amplitude (Gaussian fit)', fontsize=9, loc='center', color='#939799')

                # [1, 0]
                # axs[1, 0].hist(isi_block_clipped, bins=isi_bins, color='lightgray', histtype='barstacked', label='ISI')
                axs[1, 0].hist(isi_block_no_clipped, bins=isi_log_bins, color='lightgray')
                axs[1, 0].set_xscale('log')
                axs[1, 0].set_xlim([0.01, 100])
                axs[1, 0].set_xlabel('Inter Spike Interval (log scale)')
                leg_line_mfr = [f'Refractory Period Violations = {rpv_block}  \n'
                                f'Fraction of contamination = {fp_block}  \n'
                                f'Mean Firing Rate = {mfr_block} Hz \n']
                leg_1_0 = axs[1, 0].legend(leg_line_mfr, loc='best', frameon=False, fontsize=9)

                for text in leg_1_0.get_texts():
                    text.set_color("gray")

                for lh in leg_1_0.legendHandles:
                    lh.set_alpha(0)

                # [1, 1]
                # Amplitudes histogram and Mean Amplitude
                axs[1, 1].hist(amplitudes_block_no_clipped, bins=amplitudes_block_bins, color='lightgray', histtype='barstacked',
                               label='Amp')
                axs[1, 1].set_xlabel('Block amplitudes')
                labels_1_1 = [f'Block Mean Amplitude = {str(ma_block)} ']
                axs[1, 1].axvline(x=ma_block, ymin=0, ymax=0.95, linestyle='--', color='salmon')
                leg_1_1 = axs[1, 1].legend(labels_1_1, loc='best', frameon=False, fontsize=9)

                for text in leg_1_1.get_texts():
                    text.set_color("gray")

                for lh in leg_1_1.legendHandles:
                    lh.set_alpha(0)

                # FORMATTING
                format_plot(axs[0, 0])
                format_plot(axs[0, 1])
                format_plot(axs[1, 0])
                format_plot(axs[1, 1])
                fig.tight_layout()
                plt.show()

                fig.savefig(f"{sample_path_good}/Sample-{sample}-unit-{unit}.png")
                # fig.savefig(f"{sample_path_bad}/Sample-{sample}-unit-{unit}.png")

                print(f'block from unit {unit} has a block of {len(l_chunk_list)} chunks')
                print(f'Mean amplitude: {ma_block}')
                print(f'Mean Firing Rate: {mfr_block}')
                print(f'Chunks: {strings}')

                # *************************************************************************************
                # Dataframe!!

                df = pd.DataFrame(
                    {'Sample': sample,  # 19-01-16_YC011
                     'Unit': unit,  # 20
                     'PeakChUnit': peak_channel,  # 54
                     'SpikesUnit': spikes_unit_20,  # 135684
                     'NumChunks': chunks_in_block,  # 19
                     'Chunks': strings,  # [0,1,2,3,4,5,...19]
                     'SpikesBlock': spikes_block,  # 3147
                     'MissingSpikesBlock': block_spikes_missing,  # 7%
                     'MFRBlock': mfr_block,  # 80.25
                     'MeanAmpBlock': ma_block,  # 12.5
                     'RPVBlock': rpv_block,  # 5
                     'FpBlock': fp_block,  # 0.04
                     'WvfPeaksBlock': wvf_peaks_block,  # 2
                     'BigPkNegBlock': big_peak_neg_block  # 1/0
                     }, index=[0]
                )

                df.to_csv(f'{sample_path_good}Sample-{sample}-unit-{unit}.csv', index=False)
                # df.to_csv(f'{sample_path_bad}Sample-{sample}-unit-{unit}.csv', index=False)
                print(f'Summary-sample-{sample}-unit-{unit}.csv  ---> Successfully created!')

            else:

                print(f"Unit {unit} has either a big fp or a weird waveform")
                print(f'Fp: {fp_block}')

        except ValueError:
            print(f'Unit {unit} has no good chunks')