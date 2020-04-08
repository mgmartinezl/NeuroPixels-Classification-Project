# """
# Pipeline
#
# 1. Choose a curated data sample >> begin with F:\data\PkC\19-10-28_YC031
#
# 2. For each unit >> Determine good recording chunks (time)
#     2.1. Extract first 20 minutes of this recording (more than that is not useful)
#     2.2. Split recording into 60 secs (1 min) chunks
#     2.3. For each 60s chunk >> Determine if it is a **GOOD_RATE_CHUNK** BY COMPARING IT TO THE OVERALL DISTRIVUTION OF ISIs
#         2.3.1. Compute ISI dist
#         2.3.2. Extract mean firing rate
#         2.3.2. Check if mean firing rate distributes Gaussian >> falls within 5-95% of log-normal fitted to
#         inter spike interval distribution
#     2.4. For each 60s chunk >> Determine if it is a **GOOD_WAVE_CHUNK** BY COMPARING IT TO THE OVERALL DISTRIBUTION OF "AMPLITUDES"
#         2.4.1. Extract ALL the waveforms within current chunk
#         2.4.2. Check if the dot product between all the INDIVIDUAL waveforms of current unit-chunk and
#         the MEAN template of the unit falls within 5-95% of gaussian fitted to distribution
#
# Only keep spikes falling within chunks labeled **GOOD_RATE_CHUNK** AND **GOOD_WAVE_CHUNK** to compute features later.
#
# 3. For each good unit >> Check if this "good" unit is REALLY good
#     3.1. Check negative deflection. If not, we invert it and put a flag. The +1 and -1 are not going to be used for
#     classification (only for info purposes).
#
# 4. Feature selection (normalization, PCA, etc.)
# 5. Feature engineering
#
# """
#
# from seaborn import distplot, kdeplot
# from scipy import stats
# from aux_functions import *
# from rtn.npix.spk_wvf import wvf
# from itertools import compress
# import os
#
# np.random.seed(42)
# fs = 30000
# exclusion_quantile = 0.05
# np.seterr(all='ignore')
#
# # Sample for testing
# dp = 'F:/data/PkC/19-01-16_YC011/19-01-16_YC011_probe1'
# sample = dp[12:26]
# cell_type = dp[8:11]
#
# # Load kilosort aux files
# # Amplitudes = scaled version of dot product. Still, do my own dot product to compare
# # Spike times >> in KHz
# # Spike clusters >> units found by kilosort
#
# amplitudes_sample = np.load(f'{dp}//amplitudes.npy')  # shape N_tot_spikes x 1
# spike_times = np.load(f'{dp}//spike_times.npy')  # in samples
# spike_clusters = np.load(f'{dp}//spike_clusters.npy')
#
# # We only want to examine first 20 minutes of neuronal activity
# recording_end = 20 * 60  # If we want all the recording: st[-1]/fs in seconds
#
# # Chunk size to examine recordings
# chunk_size = 60  # in seconds
# chunk_size_ms = 60 * 1000
# N_chunks = int(recording_end / chunk_size)
#
# # For log-normal and gaussian fittings
# sample_size = 1000
#
# # Extract good units of current sample >>> [7 18 208 258]
# good_units = get_units(dp, quality='good')
# all_units = get_units(dp)
# print("All units in sample:", len(all_units))
# print(f"Good units found in current sample: {len(good_units)} --> {good_units}")
# # good_units = [208]  # Only for testing with 1 unit
#
# # *********************************************************************************************************************
# # >> UNITS PROCESSING <<
#
# sample_list = []
# unit_list = []
# spikes_unit_list = []  # No threshold set yet
# peak_channel_list = []
# count_unit_wvf_peaks_list = []
# cos_similarity_template_unit_list = []
# threshold_cos_similarity_template_unit_list = []
# rpv_unit_list = []  # No threshold set yet
# Fp_unit_list = []  # No threshold set yet
# MFR_unit_list = []
# mean_amplitude_unit_list = []
# peak_detection_threshold_list = []
# drift_tracking_ratio_unit_list = []
# drift_free_unit_list = []
#
# chunks_list = []
# chunk_len_list = []
# spikes_chunk_list = []
# count_wvf_peaks_list = []
# cos_similarity_template_chunk_list = []
# threshold_cos_similarity_template_chunk_list = []
# cos_similarity_unit_chunk_list = []
# threshold_cos_similarity_unit_chunk_list = []
# rpv_list = []
# Fp_list = []
# MFR_chunk_list = []
# MA_chunk_list = []
# drift_tracking_ratio_chunk_list = []
# drift_free_chunk_list = []
#
# for unit in good_units:
#
#
#
#     # sample_list.append(sample)
#     # unit_list.append(unit)
#
#     # *****************************************************************************************************************
#     # >> UNIT SPIKES IN HE WHOLE RECORDING AND FIRST 20 MINUTES <<
#
#     # Spikes happening at samples
#     trn_samples_unit = trn(dp, unit=unit, sav=True, prnt=False, subset_selection='all', again=False)  # Raw way
#     # trn_samples_unit = spike_times[spike_clusters == unit]  # from kilosort file
#
#     # Conversion from samples to seconds, may be useful
#     trn_seconds_unit = trn_samples_unit * 1. / fs
#
#     # Conversion from samples to milliseconds
#     trn_ms_unit = trn_samples_unit * 1. / (fs * 1. / 1000)
#     # t_u1 = st[sc == u1]
#
#     # Extract spikes happening in the first 20 minutes, as is during this time when "natural" activity from neurons
#     # can be recorded. After minute 20, neurons start to be stimulated for optotagging
#     # ms stands for milliseconds (not the same as microseconds)
#     trn_seconds_unit_20 = trn_seconds_unit[(trn_seconds_unit <= 20 * 60)]
#     trn_ms_unit_20 = trn_ms_unit[(trn_ms_unit <= 20 * (60 * 1000))]
#     unit_mask = (trn_samples_unit < 20 * 60 * fs)
#     trn_samples_unit_20 = trn_samples_unit[unit_mask]
#     spikes_unit = len(trn_ms_unit_20)
#     # spikes_unit_list.append(spikes_unit)
#
#     # *****************************************************************************************************************
#     # >> UNIT PEAK CHANNEL <<
#
#     # Extract peak channel of current unit (where the deflection is maximum)
#     peak_channel = get_peak_chan(dp, unit)
#     # peak_channel_list.append(peak_channel)
#     # print("---- The peak channel for unit {} is: ".format(unit), peak_channel)
#
#     if spikes_unit != 0:  # Only for units with first 20 minutes recordings
#
#         # *************************************************************************************************************
#         # >> UNIT DUPLICATE SPIKES REMOVAL <<
#
#         # Remove duplicate spikes possibly generated by Kilosort
#         diffs = [y - x for x, y in zip(trn_ms_unit_20, trn_ms_unit_20[1:])]
#         # diffs_mask = [False for i in diffs]
#         unique_mask = []
#         z = [False if i <= 0.5 else True for i in diffs]
#         unique_mask.append(z)
#         unique_spikes_mask = unique_mask[0]
#         diffs = list(compress(diffs, unique_spikes_mask))
#
#         # *************************************************************************************************************
#         # >> UNIT INTER SPIKE INTERVAL <<
#
#         # Compute Inter Spike Interval distribution for unit and first 20 mins
#         # ISI = np.diff(trn_ms_unit) # Without duplicate spikes removed
#         # ISI = np.diff(trn_ms_unit_20)
#         # ISI = np.asarray(ISI, dtype='float64')
#         ISI = np.asarray(diffs, dtype='float64')
#
#         # Remove outliers
#         ISI = ISI[(ISI >= np.quantile(ISI, exclusion_quantile)) & (ISI <= np.quantile(ISI, 1 - exclusion_quantile))]
#
#         # Check ISI histogram
#         # filter_hist = 350
#         # if filter_hist is not None:
#         #     ISI_plot = distplot(np.where(ISI < filter_hist, ISI, np.nan), bins=500, kde=True, color='red')
#         # else:
#         #     ISI_plot = distplot(ISI, bins=500, kde=True, color='red')
#         # # nice_plot(ISI_plot, "Inter Spike Interval (ISI)", "", "{} cell, unit {}, sample {} - (20 min)".format(cell_type, unit, sample))
#         # ISI_plot.figure.savefig("Images/Pipeline/ISI-histogram-unit-{}-sample-{}-20min.png".format(unit, sample))
#         # plt.show()
#
#         # Before proceeding, make some tests
#         # It seems that ISI can follow wither log-normal or power law patterns of distributions
#
#         # distributions_ISI = {'lognorm': stats.lognorm, 'powerlognorm': stats.powerlognorm}
#         # distributions_ISI = {'lognorm': stats.lognorm}
#         #
#         # for dist in distributions_ISI.keys():
#         #     KS_test = stats.kstest(ISI, dist, distributions_ISI[dist].fit(ISI))
#         #     D = KS_test[0]
#         #     p_value = KS_test[1]
#         #
#         #     if p_value < 0.005:
#         #         print(f"H0 REJECTED (95% conf) >> ISI data not coming from {dist} distribution")
#         #         if p_value < 0.001:
#         #             print(f"H0 REJECTED (99% conf) >> ISI data not coming from {dist} distribution")
#         #     else:
#         #         print(f"H0 ACCEPTED with (95% conf) >> ISI data coming from {dist} distribution >> ")
#
#         # Fit log-normal distribution to ISI to extract parameters
#         # shape, location, scale = stats.lognorm.fit(ISI)
#         # mu, sigma = np.log(scale), shape
#         #
#         # # Create log-normal sample from previous parameters
#         # np.random.seed(42)
#         # log_sample = stats.lognorm.rvs(s=sigma, loc=0, scale=np.exp(mu), size=sample_size)
#         # log_sample = log_sample[(log_sample >= np.quantile(log_sample, exclusion_quantile)) &
#         #                         (log_sample <= np.quantile(log_sample, 1 - exclusion_quantile))]
#         # shape_sample, location_sample, scale_sample = stats.lognorm.fit(log_sample, floc=0)
#         # mu_sample, sigma_sample = np.log(scale_sample), shape_sample
#
#         # Plot simulated data
#         # if filter_hist is not None:
#         #     sim_log = distplot(np.where(log_sample < filter_hist, log_sample, np.nan), bins=800)
#         # else:
#         #     sim_log = distplot(log_sample, bins=800)
#         # # nice_plot(sim_log, "log data sample", "", "Sim log-norm from {} cell, unit {}, sample {} - (20 min)".format(cell_type, unit, sample))
#         # sim_log.figure.savefig("Images/Pipeline/Sim-lognorm-sample-unit-{}-sample-{}-20min.png".format(unit, sample))
#
#         # Compute 95% confidence quantiles
#         # lower_bound_ln = round(np.quantile(log_sample, 0.05), 2)
#         # upper_bound_ln = round(np.quantile(log_sample, 0.95), 2)
#
#         # plt.axvline(x=lower_bound_ln, ymin=0, ymax=0.25, linestyle='--', color='red')
#         # plt.axvline(x=upper_bound_ln, ymin=0, ymax=0.25, linestyle='--', color='red')
#         # plt.show()
#
#         # *************************************************************************************************************
#         # >> UNIT WAVEFORM <<
#
#         # >> WAVEFORM AROUND PEAK CHANNEL <<
#         waveform_peak_channel = np.mean(
#             wvf(dp, unit, t_waveforms=82, subset_selection=[(0, 20 * 60)])[:, :, peak_channel],
#             axis=0)  # shape (82,)
#
#         # *************************************************************************************************************
#         # >> UNIT TEMPLATE <<
#
#         # >> TEMPLATE WAVEFORM AROUND PEAK CHANNEL << Replace peak_channel by : to see all the channel templates
#         unit_template_peak_channel = np.mean(templates(dp, unit)[:, :, peak_channel], axis=0)  # shape (82,)
#
#         # *************************************************************************************************************
#         # >> UNIT ALL WAVEFORMS MEAN SUBTRACTION <<
#
#         # TODO
#
#         # *************************************************************************************************************
#         # >> UNIT MEAN WAVEFORM NORMALIZATION << (AFTER MEAN SUBTRACTION)
#
#         # max_ms_waveform_peak_channel = np.max(ms_waveform_peak_channel)
#         # min_ms_waveform_peak_channel = np.min(ms_waveform_peak_channel)
#         # range_ms_waveform_peak_channel = max_ms_waveform_peak_channel - min_ms_waveform_peak_channel
#         # norm_ms_waveform_peak_channel = 2 * (
#         #         (ms_waveform_peak_channel - min_ms_waveform_peak_channel) / range_ms_waveform_peak_channel) - 1
#
#         max_waveform_peak_channel = np.max(waveform_peak_channel)
#         min_waveform_peak_channel = np.min(waveform_peak_channel)
#         range_waveform_peak_channel = max_waveform_peak_channel - min_waveform_peak_channel
#         norm_waveform_peak_channel = 2 * ((waveform_peak_channel - min_waveform_peak_channel) / range_waveform_peak_channel) - 1
#
#         # *************************************************************************************************************
#         # >> UNIT MEAN TEMPLATE NORMALIZATION <<
#
#         # max_ms_unit_template_peak_channel = np.max(ms_unit_template_peak_channel)
#         # min_ms_unit_template_peak_channel = np.min(ms_unit_template_peak_channel)
#         # range_ms_unit_template_peak_channel = max_ms_unit_template_peak_channel - min_ms_unit_template_peak_channel
#         # norm_ms_unit_template_peak_channel = 2 * ((ms_unit_template_peak_channel - min_ms_unit_template_peak_channel) / range_ms_unit_template_peak_channel) - 1
#
#         max_unit_template_peak_channel = np.max(unit_template_peak_channel)
#         min_unit_template_peak_channel = np.min(unit_template_peak_channel)
#         range_unit_template_peak_channel = max_unit_template_peak_channel - min_unit_template_peak_channel
#         norm_unit_template_peak_channel = 2 * ((unit_template_peak_channel - min_unit_template_peak_channel) / range_unit_template_peak_channel) - 1
#
#         # *************************************************************************************************************
#         # >> UNIT TEMPLATE ALL WAVEFORMS NORMALIZATION <<
#
#         # unit_template_all_channels = np.mean(templates(dp, unit)[:, :, :], axis=0)  # shape (82, 384)
#         # max_unit_template_all_channels = np.max(unit_template_all_channels)
#         # min_unit_template_all_channels = np.min(unit_template_all_channels)
#         # range_unit_template_all_channels = max_unit_template_all_channels - min_unit_template_all_channels
#         # norm_unit_template_all_channels = 2 * (
#         # (unit_template_all_channels - min_unit_template_all_channels) / range_unit_template_all_channels) - 1
#
#         # *************************************************************************************************************
#         # >> UNIT ALL WAVEFORMS NORMALIZATION << (AFTER MEAN SUBTRACTION)
#
#         # waveform_all_channels = np.mean(wvf(dp, unit, t_waveforms=82, subset_selection=[(0, 20 * 60)])[:, :, :],
#         #                                 axis=0)  # shape (82, 384)
#         # max_waveform_all_channels = np.max(waveform_all_channels)
#         # min_waveform_all_channels = np.min(waveform_all_channels)
#         # range_waveform_all_channels = max_waveform_all_channels - min_waveform_all_channels
#         # norm_waveform_all_channels = 2 * (
#         # (waveform_all_channels - min_waveform_all_channels) / range_waveform_all_channels) - 1
#
#         # *************************************************************************************************************
#         # >> UNIT MEAN WAVEFORM MEAN SUBTRACTION <<
#
#         # ms_waveform_peak_channel = (waveform_peak_channel - np.mean(waveform_peak_channel[0:10])
#         #                             ) / np.max(waveform_peak_channel)
#
#         ms_norm_waveform_peak_channel = (norm_waveform_peak_channel - np.mean(norm_waveform_peak_channel[0:10])
#                                          ) / np.max(norm_waveform_peak_channel)
#
#         # *************************************************************************************************************
#         # >> TEMPLATE WAVEFORM MEAN SUBTRACTION <<
#
#         # ms_unit_template_peak_channel = (unit_template_peak_channel - np.mean(unit_template_peak_channel[0:10])
#         #                                  ) / np.max(unit_template_peak_channel)
#
#         ms_norm_unit_template_peak_channel = (norm_unit_template_peak_channel - np.mean(norm_unit_template_peak_channel[0:10])
#                                               ) / np.max(norm_unit_template_peak_channel)
#
#         # # *************************************************************************************************************
#         # # >> UNIT MEAN WAVEFORM NORMALIZATION PASS 2 <<
#         #
#         # max_ms_norm_waveform_peak_channel = np.max(ms_norm_waveform_peak_channel)
#         # min_ms_norm_waveform_peak_channel = np.min(ms_norm_waveform_peak_channel)
#         # range_ms_norm_waveform_peak_channel = max_ms_norm_waveform_peak_channel - min_ms_norm_waveform_peak_channel
#         # norm_ms_norm_waveform_peak_channel = 2 * (
#         #         (ms_norm_waveform_peak_channel - min_ms_norm_waveform_peak_channel) / range_ms_norm_waveform_peak_channel) - 1
#         #
#         # # *************************************************************************************************************
#         # # >> UNIT MEAN TEMPLATE NORMALIZATION PASS 2 <<
#         #
#         # max_ms_norm_unit_template_peak_channel = np.max(ms_norm_unit_template_peak_channel)
#         # min_ms_norm_unit_template_peak_channel = np.min(ms_norm_unit_template_peak_channel)
#         # range_ms_norm_unit_template_peak_channel = max_ms_norm_unit_template_peak_channel - min_ms_norm_unit_template_peak_channel
#         # norm_ms_norm_unit_template_peak_channel = 2 * (
#         #         (ms_norm_unit_template_peak_channel - min_ms_norm_unit_template_peak_channel) / range_ms_norm_unit_template_peak_channel) - 1
#         #
#         # # *************************************************************************************************************
#         # # >> UNIT MEAN WAVEFORM MEAN SUBTRACTION PASS 2 <<
#         #
#         # ms_norm_ms_norm_waveform_peak_channel = (norm_ms_norm_waveform_peak_channel - np.mean(norm_ms_norm_waveform_peak_channel[0:10])
#         #                                  ) / np.max(norm_ms_norm_waveform_peak_channel)
#         #
#         # # *************************************************************************************************************
#         # # >> TEMPLATE WAVEFORM MEAN SUBTRACTION PASS 2 <<
#         #
#         # ms_norm_ms_norm_unit_template_peak_channel = (norm_ms_norm_unit_template_peak_channel - np.mean(
#         #     norm_ms_norm_unit_template_peak_channel[0:10])) / np.max(norm_ms_norm_unit_template_peak_channel)
#
#         # *************************************************************************************************************
#         # >> QUALITY CHECK OF UNIT WAVEFORM SHAPE <<
#
#         # Check if first maximum peak was negative ---> This function is mine
#         # first_peak_detector = peak_detector_norm_waveform(norm_ms_waveform_peak_channel)
#
#         # Check number of peaks in the waveform
#         look_aheads = list(range(5, 55, 5))
#         detected_peaks = []
#         xs = []
#         ys = []
#         wvf_xs = []
#         wvf_ys = []
#
#         for lk in look_aheads:
#             detected_peaks.append(peakdetect(ms_norm_waveform_peak_channel, lookahead=lk))
#
#         detected_peaks = [x for x in detected_peaks[0] if x != []]
#         detected_peaks = [item for items in detected_peaks for item in items]
#
#         for peaks in detected_peaks:
#             if (peaks[0] >= 30) and (peaks[0] < 50):
#                 wvf_xs.append(peaks[0])
#                 wvf_ys.append(peaks[1])
#             else:
#                 xs.append(peaks[0])
#                 ys.append(peaks[1])
#
#         count_unit_wvf_peaks = len(wvf_xs)
#         # count_unit_wvf_peaks_list.append(count_unit_wvf_peaks)
#
#         # *************************************************************************************************************
#         # >> COSINE SIMILARITY (UNIT VS TEMPLATE) <<
#
#         dot_product = np.dot(ms_norm_unit_template_peak_channel, ms_norm_waveform_peak_channel)
#         norm_template = np.linalg.norm(ms_norm_unit_template_peak_channel)
#         norm_unit_waveform = np.linalg.norm(ms_norm_waveform_peak_channel)
#         cos_similarity_template_unit = dot_product / (norm_template * norm_unit_waveform)
#         cos_similarity_template_unit = round(float(cos_similarity_template_unit), 2)
#         # cos_similarity_template_unit_list.append(cos_similarity_template_unit)
#         threshold_cos_similarity_template_unit = 1 if cos_similarity_template_unit >= 0.6 else 0
#         # threshold_cos_similarity_template_unit_list.append(threshold_cos_similarity_template_unit)
#
#         # *************************************************************************************************************
#         # >> UNIT FRACTION OF CONTAMINATION AND REFRACTORY PERIOD VIOLATIONS <<
#
#         # based on Hill et al., J Neuro, 2011
#         N = spikes_unit
#         T = 20 * 60 * 1000  # T: total experiment duration in milliseconds
#         tauR = 2  # tauR: refractory period >> 2 milliseconds
#         tauC = 0.5  # tauC: censored period >> 0.5 milliseconds
#
#         a = 2 * (tauR - tauC) * (N ** 2) / T  # In spikes >> r = 2*(tauR - tauC) * N^2 * (1-Fp) * Fp / T >> solve for Fp
#         rpv_unit = sum(ISI <= tauR)  # Refractory period violations: in spikes
#         # rpv_unit_list.append(rpv_unit)
#
#         if rpv_unit == 0:
#             Fp_unit = 0  # Fraction of contamination
#         else:
#             # rts = [-1, 1, -rpv/a]
#             # Fp = round(min(rts),2)
#             Fp_unit = round(rpv_unit / a, 2)  # r >> solve for Fp
#             if isinstance(Fp_unit, complex):  # function returns imaginary number if r is too high.
#                 Fp_unit = np.nan
#
#         # Fp_unit_list.append(Fp_unit)
#
#         # *************************************************************************************************************
#         # >> UNIT MEAN FIRING RATE <<
#
#         # Compute Mean Firing Rate for current chunk (output in spikes/second)
#         MFR_unit = 1000. / np.mean(ISI)
#         MFR_unit = round(MFR_unit, 2)
#         # MFR_unit_list.append(MFR_unit)
#
#         # *************************************************************************************************************
#         # >> UNIT AMPLITUDES <<
#
#         amplitudes_unit = amplitudes_sample[spike_clusters == unit]
#
#         # We need amplitudes of spikes in the first 20 mins
#         spike_times_unit = spike_times[spike_clusters == unit]
#         unit_mask_20 = (spike_times_unit <= 20 * 60 * fs)
#         amplitudes_unit_20 = amplitudes_unit[unit_mask_20]
#         mean_amplitude_unit = np.mean(amplitudes_unit_20)
#         mean_amplitude_unit = round(float(mean_amplitude_unit), 2)
#         # mean_amplitude_unit_list.append(mean_amplitude_unit)
#
#         # Normalize
#         max_amplitudes_unit_20 = np.max(amplitudes_unit_20)
#         min_amplitudes_unit_20 = np.min(amplitudes_unit_20)
#         range_amplitudes_unit_20 = max_amplitudes_unit_20 - min_amplitudes_unit_20
#         norm_amplitudes_unit_20 = 2 * ((amplitudes_unit_20 - min_amplitudes_unit_20) / range_amplitudes_unit_20) - 1
#
#         # Fit Gaussian distribution to amplitudes
#         mu, sigma = stats.norm.fit(amplitudes_unit_20)
#         gaussian_sample = stats.norm.rvs(loc=mu, scale=sigma, size=sample_size)
#
#         # Define a peak detection threshold
#         peak_detection_threshold = round(float(mu - (2 * sigma)), 2)
#         # peak_detection_threshold_list.append(peak_detection_threshold)
#
#         # Compute area under the curve for spikes above the peak detection threshold
#         drift_tracking_ratio_unit = 1 - stats.norm(mu, sigma).cdf(peak_detection_threshold)
#         drift_tracking_ratio_unit = round(drift_tracking_ratio_unit, 2)
#         # drift_tracking_ratio_unit_list.append(drift_tracking_ratio_unit)
#
#         drift_free_unit = 1 if drift_tracking_ratio_unit > 0.7 else 0
#         # drift_free_unit_list.append(drift_free_unit)
#
#         # *************************************************************************************************************
#         # >> UNIT AUTOCORRELOGRAM <<
#
#         unit_ACG = acg(dp, unit, 0.2, 80)
#         max_unit_ACG = np.max(unit_ACG)
#         min_unit_ACG = np.min(unit_ACG)
#         range_unit_ACG = max_unit_ACG - min_unit_ACG
#         norm_unit_ACG = 2 * ((unit_ACG - min_unit_ACG) / range_unit_ACG) - 1
#
#         # *************************************************************************************************************
#         # Check amplitudes histogram of the whole unit. We really dont care about this. Extract the same for the chunks
#         # filter_hist = None
#         # if filter_hist is not None:
#         #     amplitudes_unit_plot = distplot(np.where(amplitudes_unit_20 < filter_hist, amplitudes_unit_20, np.nan), bins=300, kde=True, color='orange')
#         # else:
#         #     amplitudes_unit_plot = distplot(amplitudes_unit_20, bins=300, kde=True, color='orange')
#         # nice_plot(amplitudes_unit_plot, "Waveforms amplitude", "", "{} cell, unit {}, sample {} - (20 min)".format(cell_type, unit, sample))
#         # # amplitudes_unit_plot.figure.savefig("Images/Pipeline/Amplitudes-histogram-unit-{}-sample-{}-20min.png".format(unit, sample))
#         # plt.show()
#
#         # Make some tests about amplitudes distribution
#         # It seems that amplitudes come from either Gaussian or Normal Inverse Gaussian distributions
#         # distributions_amplitudes = {'norm': stats.norm, 'norminvgauss': stats.norminvgauss}
#         # distributions_amplitudes = {'norm': stats.norm}
#         #
#         # for dist in distributions_amplitudes.keys():
#         #     KS_test = stats.kstest(amplitudes_unit, dist, distributions_amplitudes[dist].fit(amplitudes_unit))
#         #     D = KS_test[0]
#         #     p_value = KS_test[1]
#         #
#         #     if p_value < 0.005:
#         #         print(f"H0 REJECTED (95% conf) >> AMP data not coming from {dist} distribution")
#         #         if p_value < 0.001:
#         #             print(f"H0 REJECTED (99% conf) >> AMP data not coming from {dist} distribution")
#         #     else:
#         #         print(f"H0 ACCEPTED (95% conf) >> AMP data coming from {dist} distribution >> ")
#
#         # Fit Gaussian distribution to ISI to extract parameters
#         # mu, sigma = stats.norm.fit(amplitudes_unit)
#         #
#         # # Create Gaussian sample from previous parameters
#         # norm_sample = stats.norm.rvs(loc=mu, scale=sigma, size=sample_size)
#         # norm_sample = norm_sample[(norm_sample >= np.quantile(norm_sample, exclusion_quantile)) &
#         #                           (norm_sample <= np.quantile(norm_sample, 1 - exclusion_quantile))]
#         # mu_sample, sigma_sample = stats.norm.fit(norm_sample, floc=0)
#
#         # Plot simulated data
#         # if filter_hist is not None:
#         #     sim_gauss = distplot(np.where(norm_sample < filter_hist, norm_sample, np.nan), bins=800, color='green')
#         # else:
#         #     sim_gauss = distplot(norm_sample, bins=800)
#         # # nice_plot(sim_gauss, "Gaussian sample", "", "Sim gaussian from {} cell, unit {}, sample {} - (20 min)".format(cell_type, unit, sample))
#         # sim_gauss.figure.savefig("Images/Pipeline/Sim-lognorm-sample-unit-{}-sample-{}-20min.png".format(unit, sample))
#
#         # Compute 95% confidence quantiles
#         # lower_bound_norm = round(np.quantile(norm_sample, 0.05), 2)
#         # upper_bound_norm = round(np.quantile(norm_sample, 0.95), 2)
#
#         # plt.axvline(x=lower_bound_norm, ymin=0, ymax=0.25, linestyle='--', color='red')
#         # plt.axvline(x=upper_bound_norm, ymin=0, ymax=0.25, linestyle='--', color='red')
#         # plt.show()
#
#         # *************************************************************************************************************
#         # UNIT GRAPHICS!
#
#         fig, axs = plt.subplots(2, 2, figsize=(10, 7))
#         fig.suptitle(
#             f'Sample {sample}, unit {unit} - (20min) - Total spikes: {spikes_unit} - '
#             f'Waveform detected peaks: {count_unit_wvf_peaks}',
#             y=0.98, fontsize=10, color='#939799')
#
#         # (0, 0)
#         # ******************************************************************************************************
#         # MEAN WAVEFORMS: UNIT VS TEMPLATE
#
#         labels_0_0 = ['Unit', 'Template']
#         axs[0, 0].plot(ms_norm_waveform_peak_channel, color='gold')
#         axs[0, 0].plot(ms_norm_unit_template_peak_channel, color='lightgray')
#         axs[0, 0].scatter(xs, ys, marker='v', c='grey')
#         axs[0, 0].scatter(wvf_xs, wvf_ys, marker='v', c='salmon')
#
#         leg_0_0 = axs[0, 0].legend(labels_0_0, loc='best', frameon=False, fontsize=9)
#         for text in leg_0_0.get_texts():
#             text.set_color("gray")
#         axs[0, 0].set_title(f' \n \n Mean wvf unit vs template (cos similarity: {cos_similarity_template_unit})',
#                             fontsize=9, fontname="DejaVu Sans", loc='center', color='#939799')
#
#         # (0, 1)
#         # ******************************************************************************************************
#         # ACG FOR UNIT
#
#         axs[0, 1].plot(norm_unit_ACG, color='lightcoral')
#         axs[0, 1].set_title(f' \n \n Autocorrelogram', fontsize=9, fontname="DejaVu Sans", loc='center',
#                             color='#939799')
#
#         # (1, 0)
#         # ******************************************************************************************************
#         # UNIT ISI HISTOGRAM AND MEAN FIRING RATE GRAPH
#
#         # Compute ideal number of bins with Freedman-Diaconis’s Rule
#         len_ISI = int(len(ISI))
#         no_RVP = [i for i in ISI if i >= 2.0]  # 2ms
#         len_no_RVP = len(no_RVP)
#         bins = int(np.floor((len_ISI ** (1 / 3)))) * 10
#
#         axs[1, 0].hist(ISI, bins=bins, color='lightgray', histtype='barstacked', label='ISI')
#         axs[1, 0].set_xlabel('Inter Spike Interval')
#
#         leg_line_mfr = [f'Refractory Period Violations = {rpv_unit}  \n'
#                         f'Mean Firing Rate = {MFR_unit}  \n'
#                         f'Fraction of contamination = {Fp_unit}']
#         axs[1, 0].axvline(x=MFR_unit, ymin=0, ymax=0.95, linestyle='--', color='salmon', label='mfr')
#         leg_1_0 = axs[1, 0].legend(leg_line_mfr, loc='best', frameon=False, fontsize=9)
#
#         for text in leg_1_0.get_texts():
#             text.set_color("gray")
#
#         # (1, 1)
#         # ******************************************************************************************************
#         # UNIT DRIFT RACKING RATIO AND MEAN AMPLITUDE
#
#         if drift_tracking_ratio_unit > 0.7:
#             drift_free_spikes = nice_plot(distplot(gaussian_sample, ax=axs[1, 1], kde_kws={"shade": True},
#                                                    hist=False, kde=True, color='salmon'),
#                                           "Amplitudes (Gaussian fit)", "", "")
#         else:
#             drift_free_spikes = nice_plot(distplot(gaussian_sample, ax=axs[1, 1],
#                                                    kde_kws={"shade": True}, hist=False, kde=True,
#                                                    color='lightgray'),
#                                           "Amplitudes (Gaussian fit)", "", "")
#
#         leg_line_MA = [f'Drift Tracking Ratio (AUC) = {str(drift_tracking_ratio_unit)} \n'
#                        f'Unit Mean Amplitude = {str(mean_amplitude_unit)}  \n'
#                        f'Peak detection threshold = {peak_detection_threshold}']
#         axs[1, 1].axvline(x=mean_amplitude_unit, ymin=0, ymax=0.95, linestyle='--', color='dimgray', label='drift')
#         leg_1_1 = axs[1, 1].legend(leg_line_MA, loc='best', frameon=False, fontsize=9)
#
#         for text in leg_1_1.get_texts():
#             text.set_color("gray")
#
#         axs[1, 1].set_xlim(left=peak_detection_threshold)
#
#         # ******************************************************************************************************
#         # FORMATTING
#
#         axs[0, 0].spines["top"].set_visible(False)
#         axs[0, 0].spines["right"].set_visible(False)
#         axs[0, 1].spines["top"].set_visible(False)
#         axs[0, 1].spines["right"].set_visible(False)
#         axs[1, 0].spines["top"].set_visible(False)
#         axs[1, 0].spines["right"].set_visible(False)
#         axs[0, 0].tick_params(axis='x', colors='#939799')
#         axs[0, 0].tick_params(axis='y', colors='#939799')
#         axs[0, 1].tick_params(axis='x', colors='#939799')
#         axs[0, 1].tick_params(axis='y', colors='#939799')
#         axs[1, 0].tick_params(axis='x', colors='#939799')
#         axs[1, 0].tick_params(axis='y', colors='#939799')
#         axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
#         axs[0, 0].tick_params(axis='both', which='major', labelsize=10)
#         axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
#         axs[0, 1].tick_params(axis='both', which='major', labelsize=10)
#         axs[1, 0].tick_params(axis='both', which='major', labelsize=10)
#         axs[0, 0].yaxis.label.set_color('#939799')
#         axs[0, 0].xaxis.label.set_color('#939799')
#         axs[0, 1].yaxis.label.set_color('#939799')
#         axs[0, 1].xaxis.label.set_color('#939799')
#         axs[1, 0].xaxis.label.set_color('#939799')
#         axs[0, 0].spines['bottom'].set_color('#939799')
#         axs[0, 0].spines['left'].set_color('#939799')
#         axs[0, 1].spines['bottom'].set_color('#939799')
#         axs[0, 1].spines['left'].set_color('#939799')
#         axs[1, 0].spines['bottom'].set_color('#939799')
#         axs[1, 0].spines['left'].set_color('#939799')
#
#         fig.tight_layout()
#         plt.show()
#
#         # ******************************************************************************************************
#         # SAVE UNIT FIGURES
#
#         unit_path = f"Images/Pipeline/Sample_{sample}/Unit_{unit}"
#
#         if not os.path.exists(unit_path):
#             os.makedirs(unit_path)
#
#         fig.savefig(f"Images/Pipeline/Sample_{sample}/Unit_{unit}/sample-{sample}-unit-{unit}.png")
#
#         # *************************************************************************************************************
#         # >> CHUNKS PROCESSING <<
#
#         all_chunks_dict = {}
#
#         for i in range(N_chunks):
#
#             sample_list.append(sample)
#             unit_list.append(unit)
#             spikes_unit_list.append(spikes_unit)
#             peak_channel_list.append(peak_channel)
#             count_unit_wvf_peaks_list.append(count_unit_wvf_peaks)
#             cos_similarity_template_unit_list.append(cos_similarity_template_unit)
#             threshold_cos_similarity_template_unit_list.append(threshold_cos_similarity_template_unit)
#             rpv_unit_list.append(rpv_unit)
#             Fp_unit_list.append(Fp_unit)
#             MFR_unit_list.append(MFR_unit)
#             mean_amplitude_unit_list.append(mean_amplitude_unit)
#             peak_detection_threshold_list.append(peak_detection_threshold)
#             drift_tracking_ratio_unit_list.append(drift_tracking_ratio_unit)
#             drift_free_unit_list.append(drift_free_unit)
#
#             # key_dict = f'sample-{sample}-unit-{unit}-chunk-{i}'
#
#             chunks_list.append(i)
#
#             # Dict to store info about this chunk
#             # chunk_dict = {}
#
#             # *********************************************************************************************************
#             # >> DEFINE CHUNK LENGTH AND TIMES <<
#
#             chunk_start_time = i * chunk_size
#             chunk_end_time = (i + 1) * chunk_size
#             chunk_len = (chunk_start_time, chunk_end_time)
#             chunk_len_list.append(chunk_len)
#
#             # *********************************************************************************************************
#             # >> CHUNK AUTOCORRELOGRAM <<
#
#             chunk_ACG = acg(dp, unit, 0.2, 80, subset_selection=[(chunk_start_time, chunk_end_time)])
#             max_chunk_ACG = np.max(chunk_ACG)
#             min_chunk_ACG = np.min(chunk_ACG)
#             range_chunk_ACG = max_chunk_ACG - min_chunk_ACG
#             norm_chunk_ACG = 2 * ((chunk_ACG - min_chunk_ACG) / range_chunk_ACG) - 1
#
#             # *********************************************************************************************************
#             # >> EXTRACT MEAN CHUNK WAVEFORM <<
#
#             # Compute Mean Waveform for current chunk
#             chunk_waveform_peak_channel = np.mean(
#                 wvf(dp, unit, t_waveforms=82, subset_selection=[(chunk_start_time, chunk_end_time)])
#                 [:, :, peak_channel], axis=0)  # shape (82,)
#
#             # *********************************************************************************************************
#             # >> MEAN CHUNK WAVEFORM NORMALIZATION <<
#
#             # max_ms_chunk_waveform_peak_channel = np.max(ms_chunk_waveform_peak_channel)
#             # min_ms_chunk_waveform_peak_channel = np.min(ms_chunk_waveform_peak_channel)
#             # range_ms_chunk_waveform_peak_channel = max_ms_chunk_waveform_peak_channel - min_ms_chunk_waveform_peak_channel
#             # norm_ms_chunk_waveform_peak_channel = 2 * (
#             #             (ms_chunk_waveform_peak_channel - min_ms_chunk_waveform_peak_channel)
#             #             / range_ms_chunk_waveform_peak_channel) - 1
#
#             max_chunk_waveform_peak_channel = np.max(chunk_waveform_peak_channel)
#             min_chunk_waveform_peak_channel = np.min(chunk_waveform_peak_channel)
#             range_chunk_waveform_peak_channel = max_chunk_waveform_peak_channel - min_chunk_waveform_peak_channel
#             norm_chunk_waveform_peak_channel = 2 * (
#                     (chunk_waveform_peak_channel - min_chunk_waveform_peak_channel)
#                     / range_chunk_waveform_peak_channel) - 1
#
#             # *********************************************************************************************************
#             # >> MEAN CHUNK WAVEFORM MEAN SUBTRACTION <<
#
#             # ms_chunk_waveform_peak_channel = (chunk_waveform_peak_channel - np.mean(chunk_waveform_peak_channel[0:10])
#             #                                  ) / np.max(chunk_waveform_peak_channel)
#
#             ms_norm_chunk_waveform_peak_channel = (norm_chunk_waveform_peak_channel - np.mean(norm_chunk_waveform_peak_channel[0:10])
#                                                    ) / np.max(norm_chunk_waveform_peak_channel)
#
#
#             # *********************************************************************************************************
#             # >> CHECK QUALITY OF WAVEFORM SHAPE <<
#
#             # Check if first maximum peak was negative ---> This function is mine
#             # first_peak_detector = peak_detector_norm_waveform(norm_ms_chunk_waveform_peak_channel)
#
#             # Check number of peaks in the waveform
#             look_aheads = list(range(5, 55, 5))
#             detected_peaks = []
#             xs = []
#             ys = []
#             wvf_xs = []
#             wvf_ys = []
#
#             for lk in look_aheads:
#                 detected_peaks.append(peakdetect(ms_norm_chunk_waveform_peak_channel, lookahead=lk))
#
#             detected_peaks = [x for x in detected_peaks[0] if x != []]
#             detected_peaks = [item for items in detected_peaks for item in items]
#
#             for peaks in detected_peaks:
#                 if (peaks[0] >= 30) and (peaks[0] < 50):
#                     wvf_xs.append(peaks[0])
#                     wvf_ys.append(peaks[1])
#                 else:
#                     xs.append(peaks[0])
#                     ys.append(peaks[1])
#
#             count_wvf_peaks = len(wvf_xs)
#             count_wvf_peaks_list.append(count_wvf_peaks)
#
#             # *********************************************************************************************************
#             # >> COSINE SIMILARITY <<
#
#             # Compute normalized dot product between chunk waveform and unit template
#             dot_product = np.dot(ms_norm_unit_template_peak_channel, ms_norm_chunk_waveform_peak_channel)
#             norm_chunk_waveform = np.linalg.norm(ms_norm_chunk_waveform_peak_channel)
#             cos_similarity_template_chunk = dot_product / (norm_template * norm_chunk_waveform)
#             cos_similarity_template_chunk = round(float(cos_similarity_template_chunk), 2)
#             cos_similarity_template_chunk_list.append(cos_similarity_template_chunk)
#             threshold_cos_similarity_template_chunk = 1 if cos_similarity_template_chunk >= 0.6 else 0
#             threshold_cos_similarity_template_chunk_list.append(threshold_cos_similarity_template_chunk)
#
#             # Compute normalized dot product between chunk waveform and unit waveform
#             dot_product = np.dot(ms_norm_waveform_peak_channel, ms_norm_chunk_waveform_peak_channel)
#             norm_chunk_waveform = np.linalg.norm(ms_norm_chunk_waveform_peak_channel)
#             cos_similarity_unit_chunk = dot_product / (norm_unit_waveform * norm_chunk_waveform)
#             cos_similarity_unit_chunk = round(float(cos_similarity_unit_chunk), 2)
#             cos_similarity_unit_chunk_list.append(cos_similarity_unit_chunk)
#             threshold_cos_similarity_unit_chunk = 1 if cos_similarity_unit_chunk >= 0.6 else 0
#             threshold_cos_similarity_unit_chunk_list.append(threshold_cos_similarity_unit_chunk)
#
#             # Create mask to select current chunk spikes
#             chunk_mask = (i * chunk_size * fs <= spike_times_unit) & (spike_times_unit < (i + 1) * chunk_size * fs)
#             chunk_mask = chunk_mask.reshape(len(spike_times_unit), )
#             trn_samples_chunk = trn_samples_unit[chunk_mask]  # select spike times only for this chunk
#             trn_ms_chunk = trn_samples_chunk * 1. / (fs * 1. / 1000)  # in ms >> Review if this can be avoided
#             spikes_chunk = len(trn_ms_chunk)
#             spikes_chunk_list.append(spikes_chunk)
#
#             # *********************************************************************************************************
#             # >> INTER SPIKE INTERVAL <<
#
#             # Compute Inter Spike Interval for current chunk
#             ISI_chunk = np.diff(trn_ms_chunk)
#             ISI_chunk = np.asarray(ISI_chunk, dtype='float64')
#
#             # Remove outliers
#             ISI_chunk = ISI_chunk[(ISI_chunk >= np.quantile(ISI_chunk, exclusion_quantile)) &
#                                   (ISI_chunk <= np.quantile(ISI_chunk, 1 - exclusion_quantile))]
#
#             # *********************************************************************************************************
#             # >> FRACTION OF CONTAMINATION and REFRACTORY PERIOD VIOLATIONS <<
#
#             # based on Hill et al., J Neuro, 2011
#             N = spikes_chunk
#             T = chunk_size_ms  # T: total experiment duration in milliseconds
#             tauR = 2  # tauR: refractory period >> 2 milliseconds
#             tauC = 0.5  # tauC: censored period >> 0.5 milliseconds
#
#             a = 2 * (tauR - tauC) * (
#                     N ** 2) / T  # In spikes >> r = 2*(tauR - tauC) * N^2 * (1-Fp) * Fp / T >> solve for Fp
#             rpv = sum(ISI_chunk <= tauR)  # Refractory period violations: in spikes
#
#             if rpv == 0:
#                 Fp = 0  # Fraction of contamination
#             else:
#                 # rts = [-1, 1, -rpv/a]
#                 # Fp = round(min(rts),2)
#                 Fp = round(rpv / a, 2)  # r >> solve for Fp
#                 if isinstance(Fp, complex):  # function returns imaginary number if r is too high.
#                     Fp = np.nan
#
#             rpv_list.append(rpv)
#             Fp_list.append(Fp)
#
#             # *********************************************************************************************************
#             # >> MEAN FIRING RATE <<
#
#             # Compute Mean Firing Rate for current chunk (output in spikes/second)
#             MFR_chunk = 1000. / np.mean(ISI_chunk)
#             MFR_chunk = round(MFR_chunk, 2)
#             MFR_chunk_list.append(MFR_chunk)
#
#             # Check chunk ISI
#             # ISI_chunk_plot = distplot(ISI_chunk, bins=300, kde=True)
#             # nice_plot(ISI_chunk_plot, "Inter Spike Interval", "", f"unit {unit}, chunk {i}, sample {sample} ")
#             # ISI_chunk_plot.figure.savefig("Images/Pipeline/ISI-unit-{}-sample-{}-chunk-{}.png".format(unit, sample, i))
#
#             # Test log-normal goodness of fit on chunk ISI
#             KS_test_lognorm = stats.kstest(ISI_chunk, 'lognorm', stats.lognorm.fit(ISI_chunk))
#             p_value_lognorm = KS_test_lognorm[1]
#             KS_test_powerlognorm = stats.kstest(ISI_chunk, 'powerlognorm', stats.powerlognorm.fit(ISI_chunk))
#             p_value_powerlognorm = KS_test_powerlognorm[1]
#             ISI_chunk_is_lognormal = 0 if (p_value_lognorm < 0.005) & (p_value_powerlognorm < 0.005) else 1
#
#             # Fit log-normal distribution to ISI
#             shape, location, scale = stats.lognorm.fit(ISI_chunk)
#             mu, sigma = np.log(scale), shape
#             lognormal_sample = stats.lognorm.rvs(s=sigma, loc=0, scale=np.exp(mu), size=sample_size)
#
#             # Compute 5-95% quantiles
#             lower_bound_ln = round(np.quantile(lognormal_sample, 0.05), 2)
#             upper_bound_ln = round(np.quantile(lognormal_sample, 0.95), 2)
#
#             # Is MFR within the boundaries?
#             MFR_chunk_in_ci = 1 if lower_bound_ln <= MFR_chunk <= upper_bound_ln else 0
#
#             # *********************************************************************************************************
#             # >> AMPLITUDES <<
#
#             # Also, compute the mean amplitude of the chunk (looking at kilosort amplitudes file)
#             amplitudes_chunk = amplitudes_unit[chunk_mask]  # select amplitudes only for this chunk
#
#             # Remove outliers
#             amplitudes_chunk = amplitudes_chunk[
#                 (amplitudes_chunk >= np.quantile(amplitudes_chunk, exclusion_quantile)) & (
#                         amplitudes_chunk <= np.quantile(ISI, 1 - exclusion_quantile))]
#             amplitudes_chunk = amplitudes_chunk.reshape(len(amplitudes_chunk), )
#
#             # Count the amplitudes (must coincide with number of spikes)
#             amplitudes_chunk_count = len(amplitudes_chunk)
#
#             # Compute Mean Chunk Amplitude
#             MA_chunk = float(np.mean(amplitudes_chunk))
#             MA_chunk = round(MA_chunk, 2)
#             MA_chunk_list.append(MA_chunk)
#
#             # Normalize
#             max_amplitudes_chunk = np.max(amplitudes_chunk)
#             min_amplitudes_chunk = np.min(amplitudes_chunk)
#             range_amplitudes_chunk = max_amplitudes_chunk - min_amplitudes_chunk
#             norm_amplitudes_chunk = 2 * ((amplitudes_chunk - min_amplitudes_chunk) / range_amplitudes_chunk) - 1
#
#             # Check chunk amplitude
#             # amplitudes_chunk_plot = distplot(amplitudes_chunk, bins=300, kde=True, color='orange')
#             # nice_plot(amplitudes_chunk_plot, "Amplitude", "", f"unit {unit}, chunk {i}, sample {sample} ")
#             # amplitudes_unit_plot.figure.savefig("Images/Pipeline/Amplitudes-histogram-unit-{}-sample-{}-20min.png".format(unit, sample))
#
#             # Test Gaussian goodness of fit on chunk amplitudes
#             KS_test = stats.kstest(amplitudes_chunk, 'norm', stats.norm.fit(amplitudes_chunk))
#             p_value = KS_test[1]
#             MA_chunk_is_gaussian = 0 if p_value < 0.005 else 1
#
#             # Fit Gaussian distribution to amplitudes
#             mu, sigma = stats.norm.fit(amplitudes_chunk)
#             gaussian_sample = stats.norm.rvs(loc=mu, scale=sigma, size=sample_size)
#
#             # Compute area under the curve for spikes above the peak detection threshold
#             drift_tracking_ratio = 1 - stats.norm(mu, sigma).cdf(peak_detection_threshold)
#             drift_tracking_ratio = round(drift_tracking_ratio, 2)
#             drift_tracking_ratio_chunk_list.append(drift_tracking_ratio)
#             drift_free_chunk = 1 if drift_tracking_ratio > 0.7 else 0
#             drift_free_chunk_list.append(drift_free_chunk)
#
#             # *********************************************************************************************************
#             # GRAPHICS!
#
#             fig, axs = plt.subplots(2, 3, figsize=(20, 12))
#             fig.suptitle(
#                 f'Sample {sample}, unit {unit} - (20min), chunk {i} - {chunk_len} s - Total spikes: {spikes_chunk} - '
#                 f'Waveform detected peaks: {count_wvf_peaks}',
#                 y=0.98, fontsize=10, color='#939799')
#
#             # (0, 0)
#             # ******************************************************************************************************
#             # MEAN WAVEFORMS: UNIT VS CHUNK
#
#             labels_0_0 = ['Unit', 'Chunk']
#             axs[0, 0].plot(ms_norm_waveform_peak_channel, color='gold')
#             axs[0, 0].plot(ms_norm_chunk_waveform_peak_channel, color='lightgray')
#             axs[0, 0].scatter(xs, ys, marker='v', c='grey')
#             axs[0, 0].scatter(wvf_xs, wvf_ys, marker='v', c='salmon')
#
#             leg_0_0 = axs[0, 0].legend(labels_0_0, loc='best', frameon=False, fontsize=9)
#             for text in leg_0_0.get_texts():
#                 text.set_color("gray")
#             axs[0, 0].set_title(f' \n \n Mean wvf unit vs chunk (cos similarity: {cos_similarity_unit_chunk})',
#                                 fontsize=9, fontname="DejaVu Sans", loc='center', color='#939799')
#
#             # (0, 1)
#             # ******************************************************************************************************
#             # MEAN WAVEFORMS: UNIT TEMPLATE VS CHUNK
#
#             labels_0_1 = ['Unit template', 'Chunk']
#             axs[0, 1].plot(ms_norm_unit_template_peak_channel, color='salmon')
#             axs[0, 1].plot(ms_norm_chunk_waveform_peak_channel, color='lightgray')
#             axs[0, 1].scatter(xs, ys, marker='v', c='grey')
#             axs[0, 1].scatter(wvf_xs, wvf_ys, marker='v', c='salmon')
#
#             leg_0_1 = axs[0, 1].legend(labels_0_1, loc='best', frameon=False, fontsize=9)
#             for text in leg_0_1.get_texts():
#                 text.set_color("gray")
#
#             axs[0, 1].set_title(f' \n \n Mean wvf template vs chunk (cos similarity: {cos_similarity_template_chunk})',
#                                 fontsize=9, fontname="DejaVu Sans", loc='center', color='#939799')
#
#             # (0, 2)
#             # ******************************************************************************************************
#             # ACG FOR CHUNK
#
#             labels_0_2 = ['Unit', 'Chunk']
#             axs[0, 2].plot(norm_unit_ACG, color='gold')
#             axs[0, 2].plot(norm_chunk_ACG, color='lightgray')
#             axs[0, 2].set_title(f' \n \n Autocorrelogram', fontsize=9, fontname="DejaVu Sans", loc='center',
#                                 color='#939799')
#
#             leg_0_2 = axs[0, 2].legend(labels_0_2, loc='best', frameon=False, fontsize=9)
#             for text in leg_0_2.get_texts():
#                 text.set_color("gray")
#
#             # (1, 0)
#             # ******************************************************************************************************
#             # ISI HISTOGRAM AND MEAN FIRING RATE GRAPH
#
#             # Compute ideal number of bins with Freedman-Diaconis’s Rule
#             len_ISI = int(len(ISI_chunk))
#             no_RVP = [i for i in ISI_chunk if i >= 2.0]  # 2ms
#             len_no_RVP = len(no_RVP)
#             bins = int(np.floor((len_ISI ** (1 / 3)))) * 10
#
#             # nice_plot(kdeplot(lognormal_sample, ax=axs[1, 0], kernel='cos', shade=True, color='lightgrey'), "Inter Spike Interval","", "")
#             axs[1, 0].hist(ISI_chunk, bins=bins, color='lightgray', histtype='barstacked', label='ISI')
#             axs[1, 0].set_xlabel('Inter Spike Interval')
#
#             leg_line_mfr = [f'Refractory Period Violations = {rpv}  \n'
#                             f'Mean Firing Rate = {MFR_chunk}  \n'
#                             f'Fraction of contamination = {Fp}']
#             axs[1, 0].axvline(x=MFR_chunk, ymin=0, ymax=0.95, linestyle='--', color='salmon', label='mfr')
#             leg_1_0 = axs[1, 0].legend(leg_line_mfr, loc='best', frameon=False, fontsize=9)
#
#             for text in leg_1_0.get_texts():
#                 text.set_color("gray")
#
#             # (1, 1)
#             # ******************************************************************************************************
#             # CHUNK AMPLITUDE
#
#             amplitudes_chunk_plot = distplot(amplitudes_chunk, ax=axs[1, 1], bins=80,
#                                              kde_kws={"shade": False},
#                                              kde=True, color='lightgray', hist=True)
#             nice_plot(amplitudes_chunk_plot, "Amplitudes", "", "")
#             # axs[1, 1].plot(norm_amplitudes_unit_20, color='lightgray')
#
#             labels_1_1 = [f'Chunk Mean Amplitude = {str(MA_chunk)} ']
#             axs[1, 1].axvline(x=MA_chunk, ymin=0, ymax=0.95, linestyle='--', color='salmon')
#
#             leg_1_1 = axs[1, 1].legend(labels_1_1, loc='best', frameon=False, fontsize=9)
#
#             for text in leg_1_1.get_texts():
#                 text.set_color("gray")
#
#             # (1, 2)
#             # ******************************************************************************************************
#             # DRIFT RACKING RATIO AND MEAN AMPLITUDE
#
#             if drift_tracking_ratio > 0.7:
#                 drift_free_spikes = nice_plot(distplot(gaussian_sample, ax=axs[1, 2], kde_kws={"shade": True},
#                                                        hist=False, kde=True, color='salmon'),
#                                               "Amplitudes (Gaussian fit)", "", "")
#             else:
#                 drift_free_spikes = nice_plot(distplot(gaussian_sample, ax=axs[1, 2],
#                                                        kde_kws={"shade": True}, hist=False, kde=True,
#                                                        color='lightgray'),
#                                               "Amplitudes (Gaussian fit)", "", "")
#
#             labels_1_2 = [f'Drift Tracking Ratio (AUC) = {str(drift_tracking_ratio)} \n'
#                           f'Chunk Mean Amplitude = {str(MA_chunk)}  \n'
#                           f'Peak detection threshold = {peak_detection_threshold}']
#             axs[1, 2].axvline(x=MA_chunk, ymin=0, ymax=0.95, linestyle='--', color='salmon', label='drift')
#             leg_1_2 = axs[1, 2].legend(labels_1_2, loc='best', frameon=False, fontsize=9)
#
#             for text in leg_1_2.get_texts():
#                 text.set_color("gray")
#
#             axs[1, 2].set_xlim(left=peak_detection_threshold)
#
#             # ******************************************************************************************************
#             # FORMATTING
#
#             axs[0, 0].spines["top"].set_visible(False)
#             axs[0, 0].spines["right"].set_visible(False)
#             axs[0, 1].spines["top"].set_visible(False)
#             axs[0, 1].spines["right"].set_visible(False)
#             axs[1, 0].spines["top"].set_visible(False)
#             axs[1, 0].spines["right"].set_visible(False)
#             axs[0, 2].spines["top"].set_visible(False)
#             axs[0, 2].spines["right"].set_visible(False)
#             axs[1, 2].spines["top"].set_visible(False)
#             axs[1, 2].spines["right"].set_visible(False)
#             axs[0, 0].tick_params(axis='x', colors='#939799')
#             axs[0, 0].tick_params(axis='y', colors='#939799')
#             axs[0, 1].tick_params(axis='x', colors='#939799')
#             axs[0, 1].tick_params(axis='y', colors='#939799')
#             axs[1, 0].tick_params(axis='x', colors='#939799')
#             axs[1, 0].tick_params(axis='y', colors='#939799')
#             axs[0, 2].tick_params(axis='x', colors='#939799')
#             axs[0, 2].tick_params(axis='y', colors='#939799')
#             axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
#             axs[0, 0].tick_params(axis='both', which='major', labelsize=10)
#             axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
#             axs[0, 1].tick_params(axis='both', which='major', labelsize=10)
#             axs[1, 0].tick_params(axis='both', which='major', labelsize=10)
#             axs[0, 2].tick_params(axis='both', which='major', labelsize=10)
#             # axs[1, 2].tick_params(labelleft=False)
#             # axs[1, 2].tick_params(labelbottom=False)
#             axs[0, 0].yaxis.label.set_color('#939799')
#             axs[0, 0].xaxis.label.set_color('#939799')
#             axs[0, 1].yaxis.label.set_color('#939799')
#             axs[0, 1].xaxis.label.set_color('#939799')
#             axs[1, 0].xaxis.label.set_color('#939799')
#             axs[0, 2].xaxis.label.set_color('#939799')
#             axs[0, 0].spines['bottom'].set_color('#939799')
#             axs[0, 0].spines['left'].set_color('#939799')
#             axs[0, 1].spines['bottom'].set_color('#939799')
#             axs[0, 1].spines['left'].set_color('#939799')
#             axs[1, 0].spines['bottom'].set_color('#939799')
#             axs[1, 0].spines['left'].set_color('#939799')
#             axs[0, 2].spines['bottom'].set_color('#939799')
#             axs[0, 2].spines['left'].set_color('#939799')
#
#             fig.tight_layout()
#             plt.show()
#
#             # ******************************************************************************************************
#             # SAVE CHUNK FIGURES
#
#             unit_chunks_path = f"Images/Pipeline/Sample_{sample}/Unit_{unit}/Chunks"
#
#             if not os.path.exists(unit_chunks_path):
#                 os.makedirs(unit_chunks_path)
#
#             fig.savefig(
#                 f"Images/Pipeline/Sample_{sample}/Unit_{unit}/Chunks/sample-{sample}-unit-{unit}-chunk-{i}-{chunk_len}.png")
#
#             # ******************************************************************************************************
#             # SAVE CHUNK INFORMATION
#
#             # chunk_dict.update([(key_dict, [chunk_len, spikes_chunk, MFR_chunk, MFR_chunk_in_ci,
#             #                                MA_chunk, MA_chunk_is_gaussian, ISI_chunk_is_lognormal,
#             #                                trn_ms_chunk, amplitudes_chunk, cos_similarity_unit_chunk,
#             #                                threshold_cos_similarity_unit_chunk, cos_similarity_template_chunk,
#             #                                threshold_cos_similarity_template_chunk, unit_template_peak_channel,
#             #                                chunk_waveform_peak_channel])])
#
#             # Store all chunks into global dictionary for current unit
#             # all_chunks_dict.update(chunk_dict)
#
# df = pd.DataFrame(
#         {'Sample': sample_list,
#          'Unit': unit_list,
#          'Unit_Peak_Ch': peak_channel_list,
#          'Unit_Spikes': spikes_unit_list,
#          'Unit_Peaks_Wvf': count_unit_wvf_peaks_list,
#          'Unit_vs_Temp_CS': cos_similarity_template_unit_list,
#          'Unit_vs_Temp_CS_t': threshold_cos_similarity_template_unit_list,
#          'Unit_RPV': rpv_unit_list,
#          'Unit_Fp': Fp_unit_list,
#          'Unit_MFR': MFR_unit_list,
#          'Unit_MA': mean_amplitude_unit_list,
#          'Unit_Peak_Detection_t': peak_detection_threshold_list,
#          'Unit_Drift_TR': drift_tracking_ratio_unit_list,
#          'Unit_Drif_TR_t': drift_free_unit_list,
#          'Chunk': chunks_list,
#          'Chunk_Len': chunk_len_list,
#          'Chunk_Spikes': spikes_chunk_list,
#          'Chunk_Peaks_Wvf': count_wvf_peaks_list,
#          'Chunk_vs_Temp_CS': cos_similarity_template_chunk_list,
#          'Chunk_vs_Temp_CS_t': threshold_cos_similarity_template_chunk_list,
#          'Chunk_vs_Unit_CS': cos_similarity_unit_chunk_list,
#          'Chunk_vs_Unit_CS_t': threshold_cos_similarity_unit_chunk_list,
#          'Chunk_RPV': rpv_list,
#          'Chunk_Fp': Fp_list,
#          'Chunk_MFR': MFR_chunk_list,
#          'Chunk_MA': MA_chunk_list,
#          'Chunk_Drift_TR': drift_tracking_ratio_chunk_list,
#          'Chunk_Drift_TR_t': drift_free_chunk_list})
#
# df.to_csv(f'Images/Pipeline/Sample_{sample}/Summary-sample-{sample}.csv', index=False)