"""
Pipeline

1. Choose a curated data sample >> begin with F:\data\PkC\19-10-28_YC031

2. For each unit >> Determine good recording chunks (time)
    2.1. Extract first 20 minutes of this recording (more than that is not useful)
    2.2. Split recording into 60 secs (1 min) chunks
    2.3. For each 60s chunk >> Determine if it is a **GOOD_RATE_CHUNK** BY COMPARING IT TO THE OVERLAL DISTRIVUTION OF ISIs
        2.3.1. Compute ISI dist
        2.3.2. Extract mean firing rate
        2.3.2. Check if mean firing rate distributes Gaussian >> falls within 5-95% of lognormal fitted to
        inter spike interval distribution
    2.4. For each 60s chunk >> Determine if it is a **GOOD_WAVE_CHUNK** BY COMPARING IT TO THE OVERALL DISTRIBUTION OF "AMPLITUDES"
        2.4.1. Extract ALL the waveforms within current chunk
        2.4.2. Check if the dot product between all the INDIVIDUAL waveforms of current unit-chunk and
        the MEAN template of the unit falls within 5-95% of gaussian fitted to distribution

Only keep spikes falling within chunks labeled **GOOD_RATE_CHUNK** AND **GOOD_WAVE_CHUNK** to compute features later.

3. For each good unit >> Check if this "good" unit is REALLY good
    3.1. Check negative deflection. If not, we invert it and put a flag. The +1 and -1 are not going to be used for
    classification (only for info purposes).

4. Feature selection (normalization, PCA, etc.)
5. Feature engineering

"""

from seaborn import distplot, kdeplot
from scipy import stats
from aux_functions import *
from rtn.npix.spk_wvf import wvf
from sklearn.metrics.pairwise import cosine_similarity

fs = 30000
exclusion_quantile = 0.01

# Sample for testing
dp_test = 'F:/data/PkC/19-10-28_YC031/19-10-28_YC031_probe1'
sample = dp_test[12:26]
cell_type = dp_test[8:11]

# Load kilosort aux files
# Amplitudes = scaled version of dot product. Still, do my own dot product to compare
# Spike times >> in KHz
# Spike clusters >> units found by kilosort

amplitudes_sample = np.load('F:/data/PkC/19-10-28_YC031/19-10-28_YC031_probe1/amplitudes.npy')  # shape N_tot_spikes x 1
spike_times = np.load('F:/data/PkC/19-10-28_YC031/19-10-28_YC031_probe1//spike_times.npy')  # in samples
spike_clusters = np.load('F:/data/PkC/19-10-28_YC031/19-10-28_YC031_probe1/spike_clusters.npy')

# We only want to examine first 20 minutes of neuronal activity
recording_end = 20 * 60  # If we want all the recording: st[-1]/fs in seconds

# Chunk size to examine recordings
chunk_size = 60  # in seconds
N_chunks = int(recording_end / chunk_size)

# For log-normal and gaussian fittings
sample_size = 1000

# Extract good units of current sample >>> [7 18 208 258]
good_units = get_units(dp_test, quality='good')
all_units = get_units(dp_test)
print("All units in sample:", len(all_units))
print(f"Good units found in current sample: {len(good_units)}")
good_units = [7]  # Only for testing with 1 unit

for unit in good_units:

    # Spikes happening at samples
    trn_samples_unit = trn(dp_test, unit=unit, sav=True, prnt=False, subset_selection='all', again=False)  # Raw way
    # trn_samples_unit = spike_times[spike_clusters == unit]  # from kilosort file

    # Conversion from samples to seconds, may be useful
    trn_seconds_unit = trn_samples_unit * 1. / fs

    # Conversion from samples to milliseconds
    trn_ms_unit = trn_samples_unit * 1. / (fs * 1. / 1000)
    # t_u1 = st[sc == u1]

    # Extract spikes happening in the first 20 minutes, as is during this time when "natural" activity from neurons
    # can be recorded. After minute 20, neurons start to be stimulated for optotagging
    # ms stands for milliseconds (not the same as microseconds)
    trn_seconds_unit_20 = trn_seconds_unit[(trn_seconds_unit <= 20 * 60)]
    trn_ms_unit_20 = trn_ms_unit[(trn_ms_unit <= 20 * (60 * 1000))]
    unit_mask = (trn_samples_unit < 20 * 60 * fs)
    trn_samples_unit_20 = trn_samples_unit[unit_mask]

    # PEAK CHANNEL EXTRACTION
    # Extract peak channel of current unit (where the deflection is maximum)
    peak_channel = get_peak_chan(dp_test, unit)

    # print("---- The peak channel for unit {} is: ".format(unit), peak_channel)

    if len(trn_ms_unit_20) != 0:

        print('******')

        # >>> PART 1 <<< ----------------------------------------------------------------------------------------------
        # ISI analysis

        # Compute Inter Spike Interval distribution for unit and first 20 mins
        # ISI = np.diff(trn_ms_unit)
        ISI = np.diff(trn_ms_unit_20)
        ISI = np.asarray(ISI, dtype='float64')

        # Remove outliers
        ISI = ISI[(ISI >= np.quantile(ISI, exclusion_quantile)) & (ISI <= np.quantile(ISI, 1 - exclusion_quantile))]

        # Check ISI histogram
        filter_hist = 350
        # if filter_hist is not None:
        #     ISI_plot = distplot(np.where(ISI < filter_hist, ISI, np.nan), bins=500, kde=True, color='red')
        # else:
        #     ISI_plot = distplot(ISI, bins=500, kde=True, color='red')
        # # nice_plot(ISI_plot, "Inter Spike Interval (ISI)", "", "{} cell, unit {}, sample {} - (20 min)".format(cell_type, unit, sample))
        # ISI_plot.figure.savefig("Images/Pipeline/ISI-histogram-unit-{}-sample-{}-20min.png".format(unit, sample))
        # plt.show()

        # Before proceeding, make some tests
        # It seems that ISI can follow wither log-normal or power law patterns of distributions

        # distributions_ISI = {'lognorm': stats.lognorm, 'powerlognorm': stats.powerlognorm}
        # distributions_ISI = {'lognorm': stats.lognorm}
        #
        # for dist in distributions_ISI.keys():
        #     KS_test = stats.kstest(ISI, dist, distributions_ISI[dist].fit(ISI))
        #     D = KS_test[0]
        #     p_value = KS_test[1]
        #
        #     if p_value < 0.005:
        #         print(f"H0 REJECTED (95% conf) >> ISI data not coming from {dist} distribution")
        #         if p_value < 0.001:
        #             print(f"H0 REJECTED (99% conf) >> ISI data not coming from {dist} distribution")
        #     else:
        #         print(f"H0 ACCEPTED with (95% conf) >> ISI data coming from {dist} distribution >> ")

        # Fit log-normal distribution to ISI to extract parameters
        shape, location, scale = stats.lognorm.fit(ISI)
        mu, sigma = np.log(scale), shape

        # Create log-normal sample from previous parameters
        np.random.seed(42)
        log_sample = stats.lognorm.rvs(s=sigma, loc=0, scale=np.exp(mu), size=sample_size)
        log_sample = log_sample[(log_sample >= np.quantile(log_sample, exclusion_quantile)) &
                                (log_sample <= np.quantile(log_sample, 1 - exclusion_quantile))]
        shape_sample, location_sample, scale_sample = stats.lognorm.fit(log_sample, floc=0)
        mu_sample, sigma_sample = np.log(scale_sample), shape_sample

        # Plot simulated data
        # if filter_hist is not None:
        #     sim_log = distplot(np.where(log_sample < filter_hist, log_sample, np.nan), bins=800)
        # else:
        #     sim_log = distplot(log_sample, bins=800)
        # # nice_plot(sim_log, "log data sample", "", "Sim log-norm from {} cell, unit {}, sample {} - (20 min)".format(cell_type, unit, sample))
        # sim_log.figure.savefig("Images/Pipeline/Sim-lognorm-sample-unit-{}-sample-{}-20min.png".format(unit, sample))

        # Compute 95% confidence quantiles
        lower_bound_ln = round(np.quantile(log_sample, 0.05), 2)
        upper_bound_ln = round(np.quantile(log_sample, 0.95), 2)

        # plt.axvline(x=lower_bound_ln, ymin=0, ymax=0.25, linestyle='--', color='red')
        # plt.axvline(x=upper_bound_ln, ymin=0, ymax=0.25, linestyle='--', color='red')
        # plt.show()

        # >>> PART 2 <<< ----------------------------------------------------------------------------------------------
        # Waveform analysis
        waveform_peak_channel = np.mean(wvf(dp_test, unit, t_waveforms=82, subset_selection=[(0, 20*60)])[:, :, peak_channel], axis=0)  # shape (82,)
        max_waveform_peak_channel = np.max(waveform_peak_channel)
        min_waveform_peak_channel = np.min(waveform_peak_channel)
        range_waveform_peak_channel = max_waveform_peak_channel - min_waveform_peak_channel
        norm_waveform_peak_channel = 2 * ((waveform_peak_channel - min_waveform_peak_channel) / range_waveform_peak_channel) - 1

        waveform_all_channels = np.mean(wvf(dp_test, unit, t_waveforms=82, subset_selection=[(0, 20*60)])[:, :, :], axis=0)  # shape (82, 384)
        max_waveform_all_channels = np.max(waveform_all_channels)
        min_waveform_all_channels = np.min(waveform_all_channels)
        range_waveform_all_channels = max_waveform_all_channels - min_waveform_all_channels
        norm_waveform_all_channels = 2 * ((waveform_all_channels - min_waveform_all_channels) / range_waveform_all_channels) - 1

        # EXTRACT TEMPLATE MATCHED AGAINST CURRENT UNIT
        # Extraction only for the peak channel. Replace peak_channel by : to see all the channel templates
        unit_template_peak_channel = np.mean(templates(dp_test, unit)[:, :, peak_channel], axis=0)  # shape (82,) >> 1 channel, because it's a mean
        max_unit_template_peak_channel = np.max(unit_template_peak_channel)
        min_unit_template_peak_channel = np.min(unit_template_peak_channel)
        range_unit_template_peak_channel = max_unit_template_peak_channel - min_unit_template_peak_channel
        norm_unit_template_peak_channel = 2 * ((unit_template_peak_channel - min_unit_template_peak_channel) / range_unit_template_peak_channel) - 1

        unit_template_all_channels = np.mean(templates(dp_test, unit)[:, :, :], axis=0)  # shape (82, 384)
        max_unit_template_all_channels = np.max(unit_template_all_channels)
        min_unit_template_all_channels = np.min(unit_template_all_channels)
        range_unit_template_all_channels = max_unit_template_all_channels - min_unit_template_all_channels
        norm_unit_template_all_channels = 2 * ((unit_template_all_channels - min_unit_template_all_channels) / range_unit_template_all_channels) - 1

        # Visualize current unit waveform and template
        fig, axs = plt.subplots(2, 2, figsize=(10, 7))
        # fig.subplots_adjust(top=0.8)
        # fig.suptitle(f'Unit {unit}', fontsize=12, fontname="DejaVu Sans", color='#939799')
        axs[0, 0].plot(norm_waveform_peak_channel)
        axs[0, 0].set_title(f'Mean waveform unit {unit} (20min)', fontsize=12, fontname="DejaVu Sans", loc='center', color='#939799')
        axs[0, 1].plot(norm_unit_template_peak_channel, color='red')
        axs[0, 1].set_title(f'Mean template unit {unit}', fontsize=12, fontname="DejaVu Sans", loc='center', color='#939799')
        axs[1, 0].plot(norm_waveform_all_channels)
        axs[1, 0].set_title(f'Individual waveforms unit {unit} (20min)', fontsize=12, fontname="DejaVu Sans", loc='center', color='#939799')
        axs[1, 1].plot(norm_unit_template_all_channels)
        axs[1, 1].set_title(f'Individual templates unit {unit}', fontsize=12, fontname="DejaVu Sans", loc='center', color='#939799')  # Many will exist in case the current unit is a merging
        axs[0, 0].spines["top"].set_visible(False)
        axs[0, 0].spines["right"].set_visible(False)
        axs[0, 1].spines["top"].set_visible(False)
        axs[0, 1].spines["right"].set_visible(False)
        axs[1, 0].spines["top"].set_visible(False)
        axs[1, 0].spines["right"].set_visible(False)
        axs[1, 1].spines["top"].set_visible(False)
        axs[1, 1].spines["right"].set_visible(False)
        axs[0, 0].tick_params(axis='x', colors='#939799')
        axs[0, 0].tick_params(axis='y', colors='#939799')
        axs[0, 1].tick_params(axis='x', colors='#939799')
        axs[0, 1].tick_params(axis='y', colors='#939799')
        axs[1, 0].tick_params(axis='x', colors='#939799')
        axs[1, 0].tick_params(axis='y', colors='#939799')
        axs[1, 1].tick_params(axis='x', colors='#939799')
        axs[1, 1].tick_params(axis='y', colors='#939799')
        axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=10)
        axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=10)
        axs[1, 0].tick_params(axis='both', which='minor', labelsize=8)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=10)
        axs[1, 1].tick_params(axis='both', which='minor', labelsize=8)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=10)
        axs[0, 0].yaxis.label.set_color('#939799')
        axs[0, 0].xaxis.label.set_color('#939799')
        axs[0, 1].yaxis.label.set_color('#939799')
        axs[0, 1].xaxis.label.set_color('#939799')
        axs[1, 0].yaxis.label.set_color('#939799')
        axs[1, 0].xaxis.label.set_color('#939799')
        axs[1, 1].yaxis.label.set_color('#939799')
        axs[1, 1].xaxis.label.set_color('#939799')
        axs[0, 0].spines['bottom'].set_color('#939799')
        axs[0, 0].spines['left'].set_color('#939799')
        axs[0, 1].spines['bottom'].set_color('#939799')
        axs[0, 1].spines['left'].set_color('#939799')
        axs[1, 0].spines['bottom'].set_color('#939799')
        axs[1, 0].spines['left'].set_color('#939799')
        axs[1, 1].spines['bottom'].set_color('#939799')
        axs[1, 1].spines['left'].set_color('#939799')

        fig.savefig("Images/Pipeline/wvf-tmp-unit-{}-sample-{}.png".format(unit, sample))
        plt.show()

        # Compute amplitudes for current unit
        amplitudes_unit = amplitudes_sample[spike_clusters == unit]

        # We need amplitudes of spikes in the first 20 mins
        spike_times_unit = spike_times[spike_clusters == unit]
        unit_mask_20 = (spike_times_unit <= 20*60*fs)
        amplitudes_unit_20 = amplitudes_unit[unit_mask_20]

        # Check amplitudes histogram
        # filter_hist = None
        # if filter_hist is not None:
        #     amplitudes_unit_plot = distplot(np.where(amplitudes_unit < filter_hist, amplitudes_unit, np.nan), bins=300, kde=True, color='orange')
        # else:
        #     amplitudes_unit_plot = distplot(amplitudes_unit, bins=300, kde=True, color='orange')
        # nice_plot(amplitudes_unit_plot, "Waveforms amplitude", "", "{} cell, unit {}, sample {} - (20 min)".format(cell_type, unit, sample))
        # amplitudes_unit_plot.figure.savefig("Images/Pipeline/Amplitudes-histogram-unit-{}-sample-{}-20min.png".format(unit, sample))
        # plt.show()

        # Make some tests about amplitudes distribution
        # It seems that amplitudes come from either Gaussian or Normal Inverse Gaussian distributions
        # distributions_amplitudes = {'norm': stats.norm, 'norminvgauss': stats.norminvgauss}
        # distributions_amplitudes = {'norm': stats.norm}
        #
        # for dist in distributions_amplitudes.keys():
        #     KS_test = stats.kstest(amplitudes_unit, dist, distributions_amplitudes[dist].fit(amplitudes_unit))
        #     D = KS_test[0]
        #     p_value = KS_test[1]
        #
        #     if p_value < 0.005:
        #         print(f"H0 REJECTED (95% conf) >> AMP data not coming from {dist} distribution")
        #         if p_value < 0.001:
        #             print(f"H0 REJECTED (99% conf) >> AMP data not coming from {dist} distribution")
        #     else:
        #         print(f"H0 ACCEPTED (95% conf) >> AMP data coming from {dist} distribution >> ")

        # Fit Gaussian distribution to amplitudes to extract parameters
        mu, sigma = stats.norm.fit(amplitudes_unit)

        # Create Gaussian sample from previous parameters
        norm_sample = stats.norm.rvs(loc=mu, scale=sigma, size=sample_size)
        norm_sample = norm_sample[(norm_sample >= np.quantile(norm_sample, exclusion_quantile)) &
                                  (norm_sample <= np.quantile(norm_sample, 1 - exclusion_quantile))]
        mu_sample, sigma_sample = stats.norm.fit(norm_sample, floc=0)

        # Plot simulated data
        # if filter_hist is not None:
        #     sim_gauss = distplot(np.where(norm_sample < filter_hist, norm_sample, np.nan), bins=800, color='green')
        # else:
        #     sim_gauss = distplot(norm_sample, bins=800)
        # # nice_plot(sim_gauss, "Gaussian sample", "", "Sim gaussian from {} cell, unit {}, sample {} - (20 min)".format(cell_type, unit, sample))
        # sim_gauss.figure.savefig("Images/Pipeline/Sim-lognorm-sample-unit-{}-sample-{}-20min.png".format(unit, sample))

        # Compute 95% confidence quantiles
        lower_bound_norm = round(np.quantile(norm_sample, 0.05), 2)
        upper_bound_norm = round(np.quantile(norm_sample, 0.95), 2)

        # plt.axvline(x=lower_bound_norm, ymin=0, ymax=0.25, linestyle='--', color='red')
        # plt.axvline(x=upper_bound_norm, ymin=0, ymax=0.25, linestyle='--', color='red')
        # plt.show()

        all_chunks_dict = {}
        cosine_similarities_units_chunks = []
        cosine_similarities_templates_chunks = []

        for i in range(N_chunks):

            key_dict = f'sample-{sample}-unit-{unit}-chunk-{i}'

            # Dict to store info about this chunk
            chunk_dict = {}

            chunk_start_time = i * chunk_size
            chunk_end_time = (i + 1) * chunk_size
            chunk_len = (chunk_start_time, chunk_end_time)

            # Compute Mean Waveform for current chunk
            chunk_waveform_peak_channel = np.mean(
                wvf(dp_test, unit, t_waveforms=82, subset_selection=[(chunk_start_time, chunk_end_time)])
                [:, :, peak_channel], axis=0)  # shape (82,)

            # Normalize waveforms
            max_chunk_waveform_peak_channel = np.max(chunk_waveform_peak_channel)
            min_chunk_waveform_peak_channel = np.min(chunk_waveform_peak_channel)
            range_chunk_waveform_peak_channel = max_chunk_waveform_peak_channel - min_chunk_waveform_peak_channel
            norm_chunk_waveform_peak_channel = 2 * ((chunk_waveform_peak_channel - min_chunk_waveform_peak_channel) / range_chunk_waveform_peak_channel) - 1

            # Compute normalized dot product between chunk waveform and unit template
            dot_product = np.dot(norm_unit_template_peak_channel, norm_chunk_waveform_peak_channel)
            norm_template = np.linalg.norm(norm_unit_template_peak_channel)
            norm_chunk_waveform = np.linalg.norm(norm_chunk_waveform_peak_channel)
            cos_similarity_template_chunk = dot_product / (norm_template * norm_chunk_waveform)
            cosine_similarities_templates_chunks.append(cos_similarity_template_chunk)
            threshold_cos_similarity_template_chunk = 1 if cos_similarity_template_chunk >= 0.6 else 0

            # Compute normalized dot product between chunk waveform and unit waveform
            dot_product = np.dot(norm_waveform_peak_channel, norm_chunk_waveform_peak_channel)
            norm_unit = np.linalg.norm(norm_waveform_peak_channel)
            norm_chunk_waveform = np.linalg.norm(norm_chunk_waveform_peak_channel)
            cos_similarity_unit_chunk = dot_product / (norm_unit * norm_chunk_waveform)
            cosine_similarities_units_chunks.append(cos_similarity_unit_chunk)
            threshold_cos_similarity_unit_chunk = 1 if cos_similarity_unit_chunk >= 0.6 else 0

            # Create mask to select current chunk spikes
            chunk_mask = (i * chunk_size * fs <= spike_times_unit) & (spike_times_unit < (i + 1) * chunk_size * fs)
            chunk_mask = chunk_mask.reshape(len(spike_times_unit),)
            trn_samples_chunk = trn_samples_unit[chunk_mask]  # select spike times only for this chunk
            trn_ms_chunk = trn_samples_chunk * 1. / (fs * 1. / 1000)  # in ms >> Review if this can be avoided
            spikes_chunk = len(trn_ms_chunk)

            # Compute Mean Firing Rate per chunk (output in spikes/second)
            ISI_chunk = np.diff(trn_ms_chunk)
            MFR_chunk = 1000. / np.mean(ISI_chunk)
            MFR_chunk = round(MFR_chunk, 2)
            MFR_chunk_in_ci = 1 if lower_bound_ln <= MFR_chunk <= upper_bound_ln else 0

            # Also, compute the mean amplitude of the chunk (looking at kilosort amplitudes file)
            amplitudes_chunk = amplitudes_unit[chunk_mask]  # select amplitudes only for this chunk
            amplitudes_chunk = amplitudes_chunk.reshape(len(amplitudes_chunk),)
            amplitudes_chunk_count = len(amplitudes_chunk)
            MA_chunk = float(np.mean(amplitudes_chunk))
            MA_chunk = round(MA_chunk, 2)
            MA_chunk_in_ci = 1 if lower_bound_norm <= MA_chunk <= upper_bound_norm else 0

            chunk_dict.update([(key_dict, [chunk_len, spikes_chunk, amplitudes_chunk_count,
                                MFR_chunk, MFR_chunk_in_ci, MA_chunk, MA_chunk_in_ci,
                                trn_ms_chunk, amplitudes_chunk, cos_similarity_unit_chunk,
                                threshold_cos_similarity_unit_chunk, cos_similarity_template_chunk,
                                threshold_cos_similarity_template_chunk, unit_template_peak_channel,
                                chunk_waveform_peak_channel])])

            # Store all chunks into global dictionary for current unit
            all_chunks_dict.update(chunk_dict)

        for i in range(N_chunks):

            chunk_start_time = i * chunk_size
            chunk_end_time = (i + 1) * chunk_size
            chunk_len = (chunk_start_time, chunk_end_time)

            # Compute Mean Waveform for current chunk
            chunk_waveform_peak_channel = np.mean(
                wvf(dp_test, unit, t_waveforms=82, subset_selection=[(chunk_start_time, chunk_end_time)])
                [:, :, peak_channel], axis=0)  # shape (82,)

            # Normalize waveforms
            max_chunk_waveform_peak_channel = np.max(chunk_waveform_peak_channel)
            min_chunk_waveform_peak_channel = np.min(chunk_waveform_peak_channel)
            range_chunk_waveform_peak_channel = max_chunk_waveform_peak_channel - min_chunk_waveform_peak_channel
            norm_chunk_waveform_peak_channel = 2 * ((chunk_waveform_peak_channel - min_chunk_waveform_peak_channel) / range_chunk_waveform_peak_channel) - 1

            # Compute normalized dot product between chunk waveform and unit template
            dot_product = np.dot(norm_unit_template_peak_channel, norm_chunk_waveform_peak_channel)
            norm_template = np.linalg.norm(norm_unit_template_peak_channel)
            norm_chunk_waveform = np.linalg.norm(norm_chunk_waveform_peak_channel)
            cos_similarity_template_chunk = dot_product / (norm_template * norm_chunk_waveform)
            cosine_similarities_templates_chunks.append(cos_similarity_template_chunk)
            threshold_cos_similarity_template_chunk = 1 if cos_similarity_template_chunk >= 0.6 else 0

            # Compute normalized dot product between chunk waveform and unit waveform
            dot_product = np.dot(norm_waveform_peak_channel, norm_chunk_waveform_peak_channel)
            norm_unit = np.linalg.norm(norm_waveform_peak_channel)
            norm_chunk_waveform = np.linalg.norm(norm_chunk_waveform_peak_channel)
            cos_similarity_unit_chunk = dot_product / (norm_unit * norm_chunk_waveform)
            cosine_similarities_units_chunks.append(cos_similarity_unit_chunk)
            threshold_cos_similarity_unit_chunk = 1 if cos_similarity_unit_chunk >= 0.6 else 0

            # Create mask to select current chunk spikes
            chunk_mask = (i * chunk_size * fs <= spike_times_unit) & (spike_times_unit < (i + 1) * chunk_size * fs)
            chunk_mask = chunk_mask.reshape(len(spike_times_unit), )
            trn_samples_chunk = trn_samples_unit[chunk_mask]  # select spike times only for this chunk
            trn_ms_chunk = trn_samples_chunk * 1. / (fs * 1. / 1000)  # in ms >> Review if this can be avoided
            spikes_chunk = len(trn_ms_chunk)

            # Compute Mean Firing Rate per chunk (output in spikes/second)
            ISI_chunk = np.diff(trn_ms_chunk)
            MFR_chunk = 1000. / np.mean(ISI_chunk)
            MFR_chunk = round(MFR_chunk, 2)
            MFR_chunk_in_ci = 1 if lower_bound_ln <= MFR_chunk <= upper_bound_ln else 0

            # Also, compute the mean amplitude of the chunk (looking at kilosort amplitudes file)
            amplitudes_chunk = amplitudes_unit[chunk_mask]  # select amplitudes only for this chunk
            amplitudes_chunk = amplitudes_chunk.reshape(len(amplitudes_chunk), )
            amplitudes_chunk_count = len(amplitudes_chunk)
            MA_chunk = float(np.mean(amplitudes_chunk))
            MA_chunk = round(MA_chunk, 2)
            MA_chunk_in_ci = 1 if lower_bound_norm <= MA_chunk <= upper_bound_norm else 0

            fig, axs = plt.subplots(3, 2, figsize=(14, 17))
            labels_0_0 = ['Unit', 'Chunk']
            axs[0, 0].plot(norm_waveform_peak_channel, color='gold')
            axs[0, 0].plot(norm_chunk_waveform_peak_channel, color='lightgray')
            leg_0_0 = axs[0, 0].legend(labels_0_0, loc='best', frameon=False)
            for text in leg_0_0.get_texts():
                text.set_color("gray")

            axs[0, 0].set_title(f'Mean waveforms: unit {unit} vs chunk {i} - {chunk_len} s', fontsize=11, fontname="DejaVu Sans", loc='center', pad=10, color='#939799')
            labels_0_1 = ['Unit template', 'Chunk']
            axs[0, 1].plot(norm_unit_template_peak_channel, color='salmon')
            axs[0, 1].plot(norm_chunk_waveform_peak_channel, color='lightgray')
            leg_0_1 = axs[0, 1].legend(labels_0_1, loc='best', frameon=False)
            for text in leg_0_1.get_texts():
                text.set_color("gray")

            axs[0, 1].set_title(f'Mean waveforms: unit {unit} template vs chunk {i} - {chunk_len} s', fontsize=11, fontname="DejaVu Sans",loc='center', color='#939799')
            axs[0, 0].spines["top"].set_visible(False)
            axs[0, 0].spines["right"].set_visible(False)
            axs[0, 1].spines["top"].set_visible(False)
            axs[0, 1].spines["right"].set_visible(False)
            axs[0, 0].tick_params(axis='x', colors='#939799')
            axs[0, 0].tick_params(axis='y', colors='#939799')
            axs[0, 1].tick_params(axis='x', colors='#939799')
            axs[0, 1].tick_params(axis='y', colors='#939799')
            axs[0, 0].tick_params(axis='both', which='minor', labelsize=8)
            axs[0, 0].tick_params(axis='both', which='major', labelsize=10)
            axs[0, 1].tick_params(axis='both', which='minor', labelsize=8)
            axs[0, 1].tick_params(axis='both', which='major', labelsize=10)
            axs[0, 0].yaxis.label.set_color('#939799')
            axs[0, 0].xaxis.label.set_color('#939799')
            axs[0, 1].yaxis.label.set_color('#939799')
            axs[0, 1].xaxis.label.set_color('#939799')
            axs[0, 0].spines['bottom'].set_color('#939799')
            axs[0, 0].spines['left'].set_color('#939799')
            axs[0, 1].spines['bottom'].set_color('#939799')
            axs[0, 1].spines['left'].set_color('#939799')

            nice_plot(distplot(log_sample, hist=False, bins=800,
                      kde_kws={"shade": True}, color='lightgrey',
                      ax=axs[1, 0]), "Mean Firing Rate", "",
                      "Unit {}, sample {} - (20min)".format(unit, sample))

            nice_plot(distplot(norm_sample, hist=False, bins=300,
                      kde_kws={"shade": True}, color='lightgrey',
                      ax=axs[1, 1]), "Mean Amplitude", "",
                     "Unit {}, sample {} - (20 min)".format(unit, sample))

            axs[1, 1].axvline(x=MA_chunk, ymin=0, ymax=0.95, linestyle='--', color='salmon')
            axs[1, 1].axvline(x=lower_bound_norm, ymin=0, ymax=0.95, linestyle='--', color='dimgrey')
            axs[1, 1].axvline(x=upper_bound_norm, ymin=0, ymax=0.95, linestyle='--', color='dimgrey')
            axs[1, 0].axvline(x=MFR_chunk, ymin=0, ymax=0.95, linestyle='--', color='salmon')
            axs[1, 0].axvline(x=lower_bound_ln, ymin=0, ymax=0.95, linestyle='--', color='dimgrey')
            axs[1, 0].axvline(x=upper_bound_ln, ymin=0, ymax=0.95, linestyle='--', color='dimgrey')
            axs[1, 0].set_xlim(left=0)

            # Cosine similarities between unit and all the chunks!
            nice_plot(kdeplot(cosine_similarities_units_chunks, kernel='cos',
                              clip=(0, 1), shade=True, color='lightgrey',
                              ax=axs[2, 0]), "Cosine similarity", "",
                      f"Cosine similarity: unit {unit} vs all chunks")

            # Compute 95% confidence quantiles
            lower_bound_cos_units_chunks = round(np.quantile(cosine_similarities_units_chunks, 0.05), 2)
            upper_bound_cos_units_chunks = round(np.quantile(cosine_similarities_units_chunks, 0.95), 2)

            axs[2, 0].axvline(x=cos_similarity_unit_chunk, ymin=0, ymax=0.95, linestyle='--', color='salmon')
            axs[2, 0].axvline(x=0.5, ymin=0, ymax=0.95, linestyle='--', color='dimgrey')
            # axs[2, 0].axvline(x=lower_bound_cos_units_chunks, ymin=0, ymax=0.95, linestyle='--', color='dimgrey')
            # axs[2, 0].axvline(x=upper_bound_cos_units_chunks, ymin=0, ymax=0.95, linestyle='--', color='dimgrey')
            axs[2, 0].set_xlim([0, 1])

            # Cosine similarities between unit template and all the chunks!
            nice_plot(kdeplot(cosine_similarities_templates_chunks, kernel='cos',
                               clip=(0,1), shade=True, color='lightgrey',
                               ax=axs[2, 1]), "Cosine similarity", "",
                      f"Cosine similarity: unit {unit} template vs all chunks")

            # Compute 95% confidence quantiles
            lower_bound_cos_temp_chunks = round(np.quantile(cosine_similarities_templates_chunks, 0.05), 2)
            upper_bound_cos_temp_chunks = round(np.quantile(cosine_similarities_templates_chunks, 0.95), 2)

            axs[2, 1].axvline(x=cos_similarity_template_chunk, ymin=0, ymax=0.95, linestyle='--', color='salmon')
            axs[2, 1].axvline(x=0.5, ymin=0, ymax=0.95, linestyle='--', color='dimgrey')
            # axs[2, 1].axvline(x=lower_bound_cos_temp_chunks, ymin=0, ymax=0.95, linestyle='--', color='dimgrey')
            # axs[2, 1].axvline(x=upper_bound_cos_temp_chunks, ymin=0, ymax=0.95, linestyle='--', color='dimgrey')
            axs[2, 1].set_xlim([0, 1])

            fig.savefig(f"Images/Pipeline/Unit_7/sample-{sample}-unit-{unit}-chunk{i}")
            plt.show()

        # UPDATE THIS WITH NEW COSINE MEASURE
        for key, values in all_chunks_dict.items():
            print(key, '-->',
                  f'Len(s):{values[0]},',
                  f'Spikes:{values[1]},',
                  f'[LB:{lower_bound_norm} - MA:{values[5]} - UP:{upper_bound_norm}]',
                  f'--> MA into interval? {values[6]},',
                  f'[LB:{lower_bound_ln} - MFR:{values[3]} - UP:{upper_bound_ln}]',
                  f'--> MA into interval? {values[4]},',
                  f'Cosine_Sim: {values[9]}',
                  f'--> Similar wvf/temp? {values[10]}')
































#     # Extract amplitude of current unit
#     amplitude_unit = amplitudes_sample[spike_clusters == unit]
#     spike_times_unit = spike_times[spike_clusters == unit]  # in samples
#
#     # Fit Gaussian
#     ci_amps = [np.mean(amplitudes_sample) - 2 * np.std(amplitudes_sample), np.mean(amplitudes_sample) + 2 * np.std(amplitudes_sample)]
#
#     chunks = []
#     amps_chunks = []  # will have N_chunks np arrays containing the amplitudes of this unit in these chunks
#     fall_in_ci = []
#
#     for i in range(N_chunks):
#         chunks.append((i * chunk_size, (i + 1) * chunk_size))
#         chunk_mask = (i * chunk_size * fs <= spike_times_unit) & (spike_times_unit < (i + 1) * chunk_size * fs)  # select amplitudes between the right times
#         amp_chunk = amplitude_unit[chunk_mask]  # spike_times_unit and amplitude_unit have the same shape so chunk_mask will have the same shape as amplitude_unit
#         amps_chunks.append(amp_chunk)
#         fall_in_ci.append(1) if ci_amps <= np.mean(amplitudes_sample) <= ci_amps else fall_in_ci.append(0)
