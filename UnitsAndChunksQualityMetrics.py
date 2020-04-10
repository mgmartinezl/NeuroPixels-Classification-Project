"""
Project: Neuropixels classification pipeline
Author: Gabriela Martinez
Script: UnitsAndChunksQualityMetrics.py
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
exclusion_quantile = 0.05
unit_size_s = 20 * 60
unit_size_ms = unit_size_s * 1000
chunk_size_s = 60
chunk_size_ms = 60 * 1000
samples_fr = unit_size_s * fs
N_chunks = int(unit_size_s / chunk_size_s)
c_bin = 0.2
c_win = 80

# Extract good units of current sample
good_units = get_units(dp, quality='good')
all_units = get_units(dp)
print("All units in sample:", len(all_units))
print(f"Good units found in current sample: {len(good_units)} --> {good_units}")
# units with problems... [93, 160, 192, 204, 234]
# good_units = [18, 19, 20, 23, 24, 25, 26, 27, 28]
# good_units = [29, 30, 31, 33, 34, 35, 36, 37, 38]
# good_units = [39, 40, 41, 42, 43, 44, 47, 49, 52]
# good_units = [53, 54, 55, 64, 67, 68, 69, 73, 76]
# good_units = [78, 81, 83, 85, 87, 88, 95, 98, 99]
# good_units = [101, 102, 107, 108, 110, 112, 113, 120]
# good_units = [121, 122, 154, 160, 192, 204, 234]

# good_units = [122, 154, 160, 192, 204, 234]
good_units = [93]

total = 0
for unit in good_units:

    l_samples = []
    l_units = []
    l_unit_spikes = []
    l_unit_peak_ch = []
    l_unit_wvf_peaks = []
    l_unit_temp_cs = []
    l_unit_temp_cs_dummy = []
    l_unit_rpv = []
    l_unit_fp = []
    l_unit_mfr = []
    l_unit_mean_amp = []
    l_unit_min_amp = []
    l_unit_missing_spikes = []
    l_unit_missing_spikes_dummy = []
    l_unit_big_peak_neg_dummy = []
    l_chunks = []
    l_chunk_len = []
    l_chunk_spikes = []
    l_chunk_wvf_peaks = []
    l_chunk_temp_cs = []
    l_chunk_temp_cs_dummy = []
    l_chunk_unit_cs = []
    l_chunk_unit_cs_dummy = []
    l_chunk_rpv = []
    l_chunk_fp = []
    l_chunk_mfr = []
    l_chunk_mean_amp = []
    l_chunk_missing_spikes = []
    l_chunk_missing_spikes_dummy = []
    l_chunk_big_peak_neg_dummy = []

    print(f'Unit >>>> {unit}')

    # >> UNIT SPIKES DURING THE FIRST 20 MINUTES <<
    # trn_samples_unit = trn(dp, unit=unit, sav=True, prnt=False, again=False)
    # trn_samples_unit_20 = trn(dp, unit=unit, subset_selection=[(0, unit_size_s)], enforced_rp=0.5)
    trn_samples_unit = trn(dp, unit=unit, enforced_rp=0.5)
    unit_mask = (trn_samples_unit < 20 * 60 * fs)
    trn_samples_unit_20 = trn_samples_unit[unit_mask]

    # samples to seconds
    trn_seconds_unit_20 = trn_samples_unit_20 * 1. / fs

    # samples to milliseconds
    trn_ms_unit_20 = trn_samples_unit_20 * 1. / (fs * 1. / 1000)

    # Extract spikes happening in the first 20 minutes
    spikes_unit_20 = len(trn_ms_unit_20)

    # *****************************************************************************************************************
    # >> UNIT PEAK CHANNEL <<

    # Extract peak channel of current unit (where the deflection is maximum)
    peak_channel = get_peak_chan(dp, unit)

    if spikes_unit_20 != 0:

        # >> UNIT INTER SPIKE INTERVAL <<
        isi_unit = compute_isi(trn_ms_unit_20, exclusion_quantile)

        # >> UNIT WAVEFORM AROUND PEAK CHANNEL <<
        waveform_peak_channel = np.mean(wvf(dp, unit, t_waveforms=82, subset_selection=[(0, unit_size_s)])
                                        [:, :, peak_channel], axis=0)

        # >> UNIT MEAN WAVEFORM NORMALIZATION <<
        norm_waveform_peak_channel = range_normalization(waveform_peak_channel)

        # >> UNIT MEAN WAVEFORM MEAN SUBTRACTION <<
        ms_norm_waveform_peak_channel = waveform_mean_subtraction(norm_waveform_peak_channel)

        # >> UNIT MEAN WAVEFORM PEAK DETECTION <<
        xs, ys, count_unit_wvf_peaks = detect_peaks(ms_norm_waveform_peak_channel)

        # >> UNIT MEAN WAVEFORM BIGGEST PEAK DETECTION <<
        unit_biggest_peak_negative = detect_biggest_peak(ms_norm_waveform_peak_channel)

        # >> UNIT TEMPLATE AROUND PEAK CHANNEL <<
        unit_template_peak_channel = np.mean(templates(dp, unit)[:, :, peak_channel], axis=0)

        # >> UNIT MEAN TEMPLATE NORMALIZATION <<
        norm_unit_template_peak_channel = range_normalization(unit_template_peak_channel)

        # >> UNIT MEAN WAVEFORM MEAN SUBTRACTION <<
        ms_norm_unit_template_peak_channel = waveform_mean_subtraction(norm_unit_template_peak_channel)

        # >> COSINE SIMILARITY (UNIT VS TEMPLATE) <<
        cos_similarity_template_unit, threshold_cos_similarity_template_unit = \
            cosine_similarity(ms_norm_unit_template_peak_channel, ms_norm_waveform_peak_channel)

        # >> UNIT FRACTION OF CONTAMINATION AND REFRACTORY PERIOD VIOLATIONS <<
        RVP_unit, Fp_unit = rvp_and_fp(isi_unit, N=spikes_unit_20, T=unit_size_ms, tauR=2, tauC=0.5)
        Fp_unit_threshold = 1 if Fp_unit <= 5 else 0

        # >> UNIT MEAN FIRING RATE <<
        MFR_unit = mean_firing_rate(isi_unit)

        # >> UNIT AMPLITUDES <<
        amplitudes_unit = amplitudes_sample[spike_clusters == unit]
        spike_times_unit = spike_times[spike_clusters == unit]
        unit_mask_20 = (spike_times_unit <= samples_fr)
        spike_times_unit_20 = spike_times_unit[unit_mask_20]
        amplitudes_unit_20 = amplitudes_unit[unit_mask_20]

        # >> REMOVE OUTLIERS << I can't do this NOW!
        # amplitudes_unit_20 = remove_outliers(amplitudes_unit_20, exclusion_quantile)

        # >> UNIT MEAN AMPLITUDE <<
        MA_unit = mean_amplitude(amplitudes_unit_20)

        # >> UNIT AMPLITUDES NORMALIZATION <<
        norm_amplitudes_unit_20 = range_normalization(amplitudes_unit_20)

        # >> GAUSSIAN FIT TO UNIT AMPLITUDES <<
        # >> UNIT % OF MISSING SPIKES <<
        a = np.asarray(amplitudes_unit_20, dtype='float64')

        x = None
        p0 = None
        n_fit = None
        n_fit_no_cut = None
        min_amp_unit = None
        unit_percent_missing = None

        try:
            x, p0, min_amp_unit, n_fit, n_fit_no_cut, unit_percent_missing = gaussian_amp_est(a)

        except RuntimeError:
            try:
                unit_percent_missing = not_gaussian_amp_est(a, nBins=200)

            except IndexError:
                unit_percent_missing = not_gaussian_amp_est(a, nBins=500)

        drift_free_unit = 1 if unit_percent_missing <= 30 else 0

        # >> UNIT ACG <<

        unit_ACG, x_unit, y_unit, ylim1_unit, yl_unit, ylim2_unit = None, None, None, None, None, None

        try:
            unit_ACG = acg(dp, unit, c_bin, c_win, subset_selection=[(0, unit_size_s)])
            x_unit = np.linspace(-c_win * 1. / 2, c_win * 1. / 2, unit_ACG.shape[0])
            y_unit = unit_ACG.copy()
            ylim1_unit = 0
            yl_unit = max(unit_ACG)
            ylim2_unit = int(yl_unit) + 5 - (yl_unit % 5)

            print('ylim2_unit', ylim2_unit)

        except ValueError:
            pass

        # *****************************************************************************************************
        # UNIT GRAPHICS!

        fig, axs = plt.subplots(2, 2, figsize=(10, 7))
        fig.suptitle(
            f'Sample {sample}, unit {unit} - (20min) - Total spikes: {spikes_unit_20} - '
            f'Waveform detected peaks: {count_unit_wvf_peaks}',
            y=0.98, fontsize=10, color='#939799')

        # (0, 0)
        # ******************************************************************************************************
        # MEAN WAVEFORMS: UNIT VS TEMPLATE

        labels_0_0 = ['Unit', 'Template']
        axs[0, 0].plot(ms_norm_waveform_peak_channel, color='gold')
        axs[0, 0].plot(ms_norm_unit_template_peak_channel, color='lightgray')
        axs[0, 0].scatter(xs, ys, marker='v', c='salmon')

        leg_0_0 = axs[0, 0].legend(labels_0_0, loc='best', frameon=False, fontsize=9)
        for text in leg_0_0.get_texts():
            text.set_color("gray")
        axs[0, 0].set_title(f' \n \n Mean wvf unit vs template (cos similarity: {cos_similarity_template_unit}) \n'
                            f' Biggest peak negative: {unit_biggest_peak_negative}',
                            fontsize=9, fontname="DejaVu Sans", loc='center', color='#939799')

        # (0, 1)
        # ******************************************************************************************************
        # ACG FOR UNIT

        if all(v is not None for v in [unit_ACG, x_unit, y_unit, ylim1_unit, yl_unit, ylim2_unit]):
            axs[0, 1].bar(x=x_unit, height=y_unit, width=0.2, color='salmon', bottom=ylim1_unit)
            axs[0, 1].set_ylim([ylim1_unit, ylim2_unit])
            axs[0, 1].set_ylabel("Autocorrelation (Hz)", size=9, color='#939799')
            axs[0, 1].set_xlabel("Time (ms)", size=9)
            axs[0, 1].set_title(f' \n \n Autocorrelogram', fontsize=9, fontname="DejaVu Sans", loc='center',
                                color='#939799')

        # (1, 0)
        # ******************************************************************************************************
        # UNIT ISI HISTOGRAM AND MEAN FIRING RATE GRAPH

        # Compute ideal number of bins with Freedman-Diaconis’s Rule
        len_ISI = int(len(isi_unit))
        no_RVP = [i for i in isi_unit if i >= 2.0]  # 2ms
        len_no_RVP = len(no_RVP)
        bins = int(np.floor((len_ISI ** (1 / 3)))) * 10

        axs[1, 0].hist(isi_unit, bins=bins, color='lightgray', histtype='barstacked', label='ISI')
        axs[1, 0].set_xlabel('Inter Spike Interval')

        leg_line_mfr = [f'Refractory Period Violations = {RVP_unit}  \n'
                        f'Fraction of contamination = {Fp_unit}  \n'
                        f'Mean Firing Rate = {MFR_unit}  \n']
        axs[1, 0].axvline(x=MFR_unit, ymin=0, ymax=0.95, linestyle='--', color='salmon', label='mfr', alpha=0)
        leg_1_0 = axs[1, 0].legend(leg_line_mfr, loc='best', frameon=False, fontsize=9)

        for text in leg_1_0.get_texts():
            text.set_color("gray")

        # (1, 1)
        # ******************************************************************************************************
        # % OF MISSING SPIKES

        if all(v is not None for v in [x, p0, n_fit, n_fit_no_cut, min_amp_unit, unit_percent_missing]):
            axs[1, 1].hist(a, bins=100, orientation='vertical', color='lightgray')
            axs[1, 1].plot(x, n_fit_no_cut, color='silver')
            axs[1, 1].plot(x, n_fit, color='gold')
            axs[1, 1].set_ylabel("# of spikes", size=9, color='#939799')
            axs[1, 1].set_xlabel("Amplitude (Gaussian fit)", size=9, color='#939799')
            leg_dtr_1_1 = [f'Spikes missing: {unit_percent_missing}%']
            axs[1, 1].axvline(p0[3], color='salmon', ls='--', lw=2, ymax=p0[0] / 2, label='MS')
            leg_1_1 = axs[1, 1].legend(leg_dtr_1_1, loc='best', frameon=False, fontsize=9)
            for text in leg_1_1.get_texts():
                text.set_color("gray")
            for lh in leg_1_1.legendHandles:
                lh.set_alpha(0)

        else:
            axs[1, 1].hist(a, bins=100, orientation='vertical', color='salmon')

        # ******************************************************************************************************
        # FORMATTING
        format_plot(axs[0, 0])
        format_plot(axs[0, 1])
        format_plot(axs[1, 0])
        format_plot(axs[1, 1])
        fig.tight_layout()
        plt.show()

        # SAVE UNIT FIGURES
        unit_path = f"Images/Pipeline/Sample_{sample}/Unit_{unit}"

        if not os.path.exists(unit_path):
            os.makedirs(unit_path)

        fig.savefig(f"Images/Pipeline/Sample_{sample}/Unit_{unit}/sample-{sample}-unit-{unit}.png")

        # ******************************************************************************************************
        # >> CHUNKS PROCESSING <<

        all_chunks_dict = {}

        for i in range(N_chunks):

            print(f'Chunk {i}')

            # >> SAVE UNIT INFORMATION <<
            l_samples.append(sample)
            l_units.append(unit)
            l_unit_spikes.append(spikes_unit_20)  # Criterion for units. >= 300
            l_unit_peak_ch.append(peak_channel)
            l_unit_wvf_peaks.append(count_unit_wvf_peaks)
            l_unit_big_peak_neg_dummy.append(unit_biggest_peak_negative)
            l_unit_temp_cs.append(cos_similarity_template_unit)
            l_unit_temp_cs_dummy.append(threshold_cos_similarity_template_unit)
            l_unit_rpv.append(RVP_unit)
            l_unit_fp.append(Fp_unit)  # Criterion for units. < 5%
            l_unit_mfr.append(MFR_unit)
            l_unit_mean_amp.append(MA_unit)
            l_unit_min_amp.append(min_amp_unit)
            l_unit_missing_spikes.append(unit_percent_missing)
            l_unit_missing_spikes_dummy.append(drift_free_unit)

            l_chunks.append(i)

            # >> DEFINE CHUNK LENGTH AND TIMES <<
            chunk_start_time = i * chunk_size_s
            chunk_end_time = (i + 1) * chunk_size_s
            chunk_len = (chunk_start_time, chunk_end_time)
            l_chunk_len.append(chunk_len)

            # >> CHUNK ACG <<
            chunk_ACG = acg(dp, unit, 0.2, 80, subset_selection=[(chunk_start_time, chunk_end_time)])
            x_chunk = np.linspace(-c_win * 1. / 2, c_win * 1. / 2, chunk_ACG.shape[0])
            y_chunk = chunk_ACG.copy()
            ylim1_chunk = 0
            yl_chunk = max(chunk_ACG)
            ylim2_chunk = int(yl_chunk) + 5 - (yl_chunk % 5)

            # >> CHUNK MEAN WAVEFORM AROUND PEAK CHANNEL <<
            chunk_waveform_peak_channel = np.mean(
                wvf(dp, unit, t_waveforms=82, subset_selection=[(chunk_start_time, chunk_end_time)])
                [:, :, peak_channel], axis=0)

            # >> MEAN CHUNK WAVEFORM NORMALIZATION <<
            norm_chunk_waveform_peak_channel = range_normalization(chunk_waveform_peak_channel)

            # >> MEAN CHUNK WAVEFORM MEAN SUBTRACTION <<
            ms_norm_chunk_waveform_peak_channel = waveform_mean_subtraction(norm_chunk_waveform_peak_channel)

            # >> MEAN CHUNK WAVEFORM PEAK DETECTION <<
            xs_c, ys_c, count_chunk_wvf_peaks = detect_peaks(ms_norm_chunk_waveform_peak_channel)
            l_chunk_wvf_peaks.append(count_chunk_wvf_peaks)  # Criterion for chunks. <= 3

            # >> MEAN CHUNK WAVEFORM BIGGEST PEAK DETECTION <<
            chunk_biggest_peak_negative = detect_biggest_peak(ms_norm_chunk_waveform_peak_channel)
            l_chunk_big_peak_neg_dummy.append(chunk_biggest_peak_negative)  # Criterion for chunks.

            # >> COSINE SIMILARITY (CHUNK VS UNIT) <<
            cos_similarity_unit_chunk, threshold_cos_similarity_unit_chunk = \
                cosine_similarity(ms_norm_waveform_peak_channel, ms_norm_chunk_waveform_peak_channel)
            l_chunk_unit_cs.append(cos_similarity_unit_chunk)
            l_chunk_unit_cs_dummy.append(threshold_cos_similarity_unit_chunk)

            # >> COSINE SIMILARITY (CHUNK VS TEMPLATE) <<
            cos_similarity_template_chunk, threshold_cos_similarity_template_chunk = \
                cosine_similarity(ms_norm_unit_template_peak_channel, ms_norm_chunk_waveform_peak_channel)
            l_chunk_temp_cs.append(cos_similarity_template_chunk)
            l_chunk_temp_cs_dummy.append(threshold_cos_similarity_template_chunk)

            # >> CHUNK SPIKES <<
            chunk_mask = (i * chunk_size_s * fs <= spike_times_unit_20) & (spike_times_unit_20 < (i + 1) * chunk_size_s * fs)
            chunk_mask = chunk_mask.reshape(len(spike_times_unit_20), )
            trn_samples_chunk = trn_samples_unit_20[chunk_mask]
            trn_ms_chunk = trn_samples_chunk * 1. / (fs * 1. / 1000)
            spikes_chunk = len(trn_ms_chunk)
            l_chunk_spikes.append(spikes_chunk)

            # >> CHUNK INTER SPIKE INTERVAL <<
            isi_chunk = compute_isi(trn_ms_chunk, exclusion_quantile)

            # >> CHUNK FRACTION OF CONTAMINATION AND REFRACTORY PERIOD VIOLATIONS <<
            RVP_chunk, Fp_chunk = rvp_and_fp(isi_chunk, N=spikes_chunk, T=chunk_size_ms, tauR=2, tauC=0.5)
            l_chunk_rpv.append(RVP_chunk)
            l_chunk_fp.append(Fp_chunk)

            # >> CHUNK MEAN FIRING RATE <<
            MFR_chunk = mean_firing_rate(isi_chunk)
            l_chunk_mfr.append(MFR_chunk)

            # >> CHUNK AMPLITUDES <<
            amplitudes_chunk = amplitudes_unit_20[chunk_mask]

            # >> REMOVE OUTLIERS <<
            amplitudes_chunk = remove_outliers(amplitudes_chunk, exclusion_quantile)

            # >> CHUNK MEAN AMPLITUDE <<
            MA_chunk = mean_amplitude(amplitudes_chunk)
            l_chunk_mean_amp.append(MA_chunk)

            # >> CHUNK AMPLITUDES NORMALIZATION <<
            norm_amplitudes_chunk = range_normalization(amplitudes_chunk)

            # >> GAUSSIAN FIT TO CHUNK AMPLITUDES <<
            # >> CHUNK % OF MISSING SPIKES <<
            a_c = np.asarray(amplitudes_chunk, dtype='float64')
            x_c = None
            p0_c = None
            n_fit_c = None
            n_fit_no_cut_c = None
            min_amp_c = None
            chunk_percent_missing = None

            try:
                x_c, p0_c, min_amp_c, n_fit_c, n_fit_no_cut_c, chunk_percent_missing = gaussian_amp_est(a_c)

            except RuntimeError:
                try:
                    chunk_percent_missing = not_gaussian_amp_est(a_c, nBins=150)

                except IndexError:
                    chunk_percent_missing = not_gaussian_amp_est(a_c, nBins=300)

            l_chunk_missing_spikes.append(chunk_percent_missing)  # Criterion for chunks. <= 30
            drift_free_chunk = 1 if chunk_percent_missing <= 30 else 0
            l_chunk_missing_spikes_dummy.append(drift_free_unit)

            # *********************************************************************************************************
            # GRAPHICS!

            fig, axs = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(
                f'Sample {sample}, unit {unit} - (20min), chunk {i} - {chunk_len} s - Total spikes: {spikes_chunk} - '
                f'Waveform detected peaks: {count_chunk_wvf_peaks}',
                y=0.98, fontsize=10, color='#939799')

            # (0, 0)
            # ******************************************************************************************************
            # MEAN WAVEFORMS: UNIT VS CHUNK

            labels_0_0 = ['Unit', 'Chunk']
            axs[0, 0].plot(ms_norm_waveform_peak_channel, color='gold')
            axs[0, 0].plot(ms_norm_chunk_waveform_peak_channel, color='lightgray')
            axs[0, 0].scatter(xs_c, ys_c, marker='v', c='salmon')

            leg_0_0 = axs[0, 0].legend(labels_0_0, loc='best', frameon=False, fontsize=10)
            for text in leg_0_0.get_texts():
                text.set_color("gray")
            axs[0, 0].set_title(f' \n \n Mean wvf unit vs chunk (cos similarity: {cos_similarity_unit_chunk}) \n'
                                f' Biggest peak negative: {chunk_biggest_peak_negative}',
                                fontsize=9, fontname="DejaVu Sans", loc='center', color='#939799')

            # (0, 1)
            # ******************************************************************************************************
            # MEAN WAVEFORMS: UNIT TEMPLATE VS CHUNK

            labels_0_1 = ['Unit template', 'Chunk']
            axs[0, 1].plot(ms_norm_unit_template_peak_channel, color='salmon')
            axs[0, 1].plot(ms_norm_chunk_waveform_peak_channel, color='lightgray')
            axs[0, 1].scatter(xs_c, ys_c, marker='v', c='salmon')

            leg_0_1 = axs[0, 1].legend(labels_0_1, loc='best', frameon=False, fontsize=9)
            for text in leg_0_1.get_texts():
                text.set_color("gray")

            axs[0, 1].set_title(f' \n \n Mean wvf template vs chunk (cos similarity: {cos_similarity_template_chunk})',
                                fontsize=9, fontname="DejaVu Sans", loc='center', color='#939799')

            # (0, 2)
            # ******************************************************************************************************
            # ACG FOR CHUNK

            axs[0, 2].bar(x=x_chunk, height=y_chunk, width=0.2, color='gold', bottom=ylim1_chunk)
            axs[0, 2].set_ylim([ylim1_chunk, ylim2_chunk])
            axs[0, 2].set_ylabel("Autocorrelation (Hz)", size=9, color='#939799')
            axs[0, 2].set_xlabel("Time (ms)", size=9)
            axs[0, 2].set_title(f' \n \n Autocorrelogram', fontsize=9, fontname="DejaVu Sans", loc='center',
                                color='#939799')

            # (1, 0)
            # ******************************************************************************************************
            # ISI HISTOGRAM AND MEAN FIRING RATE GRAPH

            # Compute ideal number of bins with Freedman-Diaconis’s Rule
            len_ISI = int(len(isi_chunk))
            no_RVP = [i for i in isi_chunk if i >= 2.0]  # 2ms
            len_no_RVP = len(no_RVP)
            bins = int(np.floor((len_ISI ** (1 / 3)))) * 10

            axs[1, 0].hist(isi_chunk, bins=bins, color='lightgray', histtype='barstacked', label='ISI')
            axs[1, 0].set_xlabel('Inter Spike Interval')

            leg_line_mfr = [f'Refractory Period Violations = {RVP_chunk}  \n'
                            f'Fraction of contamination = {Fp_chunk}  \n'
                            f'Mean Firing Rate = {MFR_chunk}  \n']
            axs[1, 0].axvline(x=MFR_chunk, ymin=0, ymax=0.95, linestyle='--', color='salmon', label='mfr', alpha=0)
            leg_1_0 = axs[1, 0].legend(leg_line_mfr, loc='best', frameon=False, fontsize=10)

            for text in leg_1_0.get_texts():
                text.set_color("gray")

            # (1, 1)
            # ******************************************************************************************************
            # CHUNK AMPLITUDE

            amplitudes_chunk_plot = distplot(amplitudes_chunk, ax=axs[1, 1], bins=80,
                                             kde_kws={"shade": False},
                                             kde=True, color='lightgray', hist=True)
            nice_plot(amplitudes_chunk_plot, "Amplitudes", "", "")
            # axs[1, 1].plot(norm_amplitudes_unit_20, color='lightgray')

            labels_1_1 = [f'Chunk Mean Amplitude = {str(MA_chunk)} ']
            axs[1, 1].axvline(x=MA_chunk, ymin=0, ymax=0.95, linestyle='--', color='salmon')

            leg_1_1 = axs[1, 1].legend(labels_1_1, loc='best', frameon=False, fontsize=10)

            for text in leg_1_1.get_texts():
                text.set_color("gray")

            # (1, 2)
            # ******************************************************************************************************
            # DRIFT RACKING RATIO AND MEAN AMPLITUDE

            if all(v is not None for v in [x_c, p0_c, n_fit_c, n_fit_no_cut_c, min_amp_c, chunk_percent_missing]):
                axs[1, 2].hist(amplitudes_chunk, bins=200, orientation='vertical', color='lightgray')
                axs[1, 2].plot(x_c, n_fit_no_cut_c, color='silver')
                axs[1, 2].plot(x_c, n_fit_c, color='gold')
                axs[1, 2].set_ylabel("# of spikes", size=9, color='#939799')
                axs[1, 2].set_xlabel("Amplitude (Gaussian fit)", size=9, color='#939799')
                leg_dtr_1_2 = [f'Spikes missing: {chunk_percent_missing}%']
                axs[1, 2].axvline(p0_c[3], color='salmon', ls='--', lw=2, ymax=min_amp_c/2, label='MS')
                leg_1_2 = axs[1, 2].legend(leg_dtr_1_2, loc='best', frameon=False, fontsize=10)
                for text in leg_1_2.get_texts():
                    text.set_color("gray")
                for lh in leg_1_2.legendHandles:
                    lh.set_alpha(0)

            else:
                axs[1, 2].hist(amplitudes_chunk, bins=200, orientation='vertical', color='salmon')
                leg_dtr_1_2_x = [f'Spikes missing: {chunk_percent_missing}%']
                leg_1_2 = axs[1, 2].legend(leg_dtr_1_2_x, loc='best', frameon=False, fontsize=10)
                for text in leg_1_2.get_texts():
                    text.set_color("gray")
                for lh in leg_1_2.legendHandles:
                    lh.set_alpha(0)

            # ******************************************************************************************************
            # FORMATTING
            format_plot(axs[0, 0])
            format_plot(axs[0, 1])
            format_plot(axs[0, 2])
            format_plot(axs[1, 0])
            format_plot(axs[1, 1])
            format_plot(axs[1, 2])
            fig.tight_layout()
            plt.show()

            # SAVE CHUNK FIGURES

            unit_chunks_path = f"Images/Pipeline/Sample_{sample}/Unit_{unit}/Chunks"

            if not os.path.exists(unit_chunks_path):
                os.makedirs(unit_chunks_path)

            fig.savefig(
                f"Images/Pipeline/Sample_{sample}/Unit_{unit}/Chunks/sample-{sample}-unit-{unit}-chunk-{i}-{chunk_len}.png")

        df = pd.DataFrame(
            {'Sample': l_samples,
             'Unit': l_units,
             'Unit_Peak_Ch': l_unit_peak_ch,
             'Unit_Spikes': l_unit_spikes,
             'Unit_Wvf_Peaks': l_unit_wvf_peaks,
             'Unit_vs_Temp_CS': l_unit_temp_cs,
             'Unit_vs_Temp_CS_dummy': l_unit_temp_cs_dummy,
             'Unit_RPV': l_unit_rpv,
             'Unit_Fp': l_unit_fp,
             'Unit_MFR': l_unit_mfr,
             'Unit_MeanAmp': l_unit_mean_amp,
             'Unit_MinAmp': l_unit_min_amp,
             'Unit_Missing_Spikes': l_unit_missing_spikes,
             'Unit_Missing_Spikes_dummy': l_unit_missing_spikes_dummy,
             'Chunk': l_chunks,
             'Chunk_Len': l_chunk_len,
             'Chunk_Spikes': l_chunk_spikes,
             'Chunk_Peaks_Wvf': l_chunk_wvf_peaks,
             'Chunk_vs_Temp_CS': l_chunk_temp_cs,
             'Chunk_vs_Temp_CS_dummy': l_chunk_temp_cs_dummy,
             'Chunk_vs_Unit_CS': l_chunk_unit_cs,
             'Chunk_vs_Unit_CS_dummy': l_chunk_unit_cs_dummy,
             'Chunk_RPV': l_chunk_rpv,
             'Chunk_Fp': l_chunk_fp,
             'Chunk_MFR': l_chunk_mfr,
             'Chunk_MA': l_chunk_mean_amp,
             'Chunk_Missing_Spikes': l_chunk_missing_spikes,
             'Chunk_Missing_Spikes_dummy': l_chunk_missing_spikes_dummy
            }
        )

        df.to_csv(f'Images/Pipeline/Sample_{sample}/Unit_{unit}/Summary-sample-{sample}-unit-{unit}.csv', index=False)
        print(f'Summary-sample-{sample}-unit-{unit}.csv  ---> Successfully created!')

        total += 1
        print('--- Progress: ', total, 'of', len(good_units))
