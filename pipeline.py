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

from aux_functions import *

fs = 30000
exclusion_quantile = 0.005

dp_test = 'F:/data/PkC/19-10-28_YC031/19-10-28_YC031_probe1'
sample = dp_test[12:26]

# Extract good units >>> [7 18 208 258]
good_units = get_units(dp_test, quality='good')
print(good_units)

# ---------------------------------------------------------------------------------------------------------------------

# The times returned by trn() are given in samples
# The sampling frequency is 30000Hz. This means, every second, 30000 samples are recorded.
# For instance, a spike happening at time '5400' according to 'spike_times.npy' (in samples) actually happens
# at time 5400/30000 = 0.168966 seconds
# In this single data path, we have approximately 3600 seconds of recordings (1 hour)

# For each unit:

good_units = [7]  # Only for testing with 1 unit

for unit in good_units:

    print(">> unit {} <<".format(unit))

    # Spikes happening at samples
    trn_samples = trn(dp_test, unit=unit, sav=True, prnt=False, rec_section='all', again=False)

    # Extract spikes times (in seconds)
    # Conversion from samples to seconds
    trn_seconds = trn_samples * 1. / fs
    print("--- Number of spikes happening in this recording for unit {}: ".format(unit), len(trn_seconds))

    # PEAK CHANNEL EXTRACTION
    # Extract peak channel of current unit where the deflection is maximum)
    peak_channel = get_peak_chan(dp_test, unit)

    print("--- The peak channel for unit {} is: ".format(unit), peak_channel)

    # Conversion from samples to miliseconds
    trn_ms = trn_samples * 1. / (fs * 1. / 1000)

    # Extract spikes happening in the first 20 minutes, as is duing this time when "natural" activity from neurons
    # can be recorded. After minute 20, neurons start to be stimulated for optotagging
    # ms stands for milliseconds (not the same as microseconds)
    trn_20_min_s = trn_seconds[(trn_seconds <= 20*(60))]
    trn_20_min_ms = trn_ms[(trn_ms <= 20*(60*1000))]

    # for i in trn_ms:
    #    print(f"{i:,}")

    print("--- Number of spikes of unit {} happening in the first 20 minutes: ".format(unit), len(trn_20_min_ms))

    # Split into 60sec chunks
    # start = 0
    # chunks = []
    # for i in np.arange(60, 1260, 60):
    #     chunk = trn_20_min_s[(trn_20_min_s > start) & (trn_20_min_s <= i)]
    #     chunks.append(chunk)
    #     start += 60

    # Do the following processing if and only if spikes were found in the first 20 minutes:
    if len(trn_20_min_ms) != 0:

        # EXTRACT MEAN WAVEFORM
        # Waveforms are usually of dims (Spikes(waveforms) x Samples x Channels), usually (100, 82, 16)
        # Extract mean spike waveform of current unit around peak channel

        # ALL the waveforms (DO NOT CONSIDER ONLY 100!!!!)
        N_waveforms =len(st[unit==sc]) # len(trn_20_min_ms)

        waveform_peak_channel = np.mean(wvf(dp_test, unit, n_waveforms=100, t_waveforms=82)[:, :, peak_channel], axis=0) # shape (82,)
        waveform_all_channels = np.mean(wvf(dp_test, unit, n_waveforms=100, t_waveforms=82)[:, :, :], axis=0) # shape (82, 384)

        # EXTRACT TEMPLATE MATCHED AGAINST CURRENT UNIT
        # Extraction only for the peak channel. Replace peak_channel by : to see all the channel templates
        unit_template_peak_channel = np.mean(templates(dp_test, unit)[:, :, peak_channel], axis=0)  # shape (82,) >> 1 channel, because it's a mean
        unit_template_all_channels = np.mean(templates(dp_test, unit)[:, :, :], axis=0)  # shape (82, 384)

        # Visualize current unit waveform and template
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].set_title('Mean waveform of unit')
        axs[0, 0].plot(waveform_peak_channel)
        axs[0, 1].set_title('Mean template of unit')
        axs[0, 1].plot(unit_template_peak_channel, color='red')
        axs[1, 0].set_title('Individual waveforms')
        axs[1, 0].plot(waveform_all_channels)
        axs[1, 1].set_title('Individual templates')  # Many will exist in case the current unit is a merging
        axs[1, 1].plot(unit_template_all_channels)

        #fig.savefig("Images/Pipeline/wvf-tmp-unit-{}-sample-{}.png".format(unit, sample))
        plt.show()

        # DOT PRODUCT (SIMILARITY) BETWEEN ALL WAVEFORMS AND THEIR MATCHING TEMPLATES
        # Mean matching template >> unit_template_peak_channel >> numpy.ndarray
        # Individual waveforms >> waveform_all_channels >> list of lists numpy.ndarray

        # print("WAVEFORMS")
        # for wvf in waveform_all_channels:
        #
        #     # Measure similarity with template
        #     print(wvf.reshape(384, 1).shape)  # 82
        #     print(unit_template_peak_channel.reshape(82, 1).shape)
        #     #print(unit_template_peak_channel.shape) #384
        #     #similarity = np.dot(wvf, unit_template_peak_channel) >> waveform_peak_channel!!

        # ## WRITTEN BY MAX
        #
        # # ALSO SEE IF THESE 'AMPLITUDE' MATCH WHAT YOU FIND WITH THE DOT PRODUCT BETWEEN WAVEFORMS AND THE TEMPLATE!!!
        # # SHOULD BE THE SAME ROUGHLY, BUT SCALED!!!!
        # # Load amplitudes = scaled version of dot product
        # fs=30000 # sampling frequency: 30kHz
        # u1=# PICK A UNIT
        # amps=np.load('amplitudes.npy') # shape will be N_tot_spikes x 1
        # st=np.load('spike_times.npy') # in samples
        # recording_end=20*60#st[-1]/fs in seconds
        # sc=np.load('spike_clusters.npy')
        #
        # # ACTUALLY COMPUTE REAL CI BY FITTING A GAUSSIAN TO DISTRIBUTION NOT JUST 2 SD...
        # ci_amps=[np.mean(amps)-2*np.std(amps), np.mean(amps)+2*np.std(amps)]
        #
        # amp_u1=amps[sc==u1]
        # t_u1=st[sc==u1] # in samples
        #
        # chunk_size=10 #s
        # N_chunks=recording_end//chunk_size
        #
        # chunks=[]
        # amp_chunks=[] # will have N_chunks np arrays containing the amplitudes of this unit in this chunks
        # fall_in_ci=[]
        # for i in range(N_chunks):
        #     chunks.append((i*chunk_size,(i+1)*chunk_size))
        #     chunk_mask=(i*chunk_size*fs<=t_u1)&(t_u1<(i+1)*chunk_size*fs) # select amplitudes between the right times
        #     amps=amp_u1[chunk_mask] # t_u1 and amps_u1 have the same shape so chunk_mask will have the same shape as amps_u1
        #     amp_chunks.append(amps)
        #     fall_in_ci.append(1) if ci_amps<=np.mean(amps)<=ci_amps else fall_in_ci.append(0)
        #
        # amps_df=pd.DataFrame(index=chunks, data={'amplitudes':amp_chunks, 'GOOD_WAVE_CHUNK':fall_in_ci})
        #
        # across_units_df #index is: units, columns are: list of GOOD_RATE_CHUNKS (index of chunks with flag 1, from 0 to N_chunks), list of GOOD_WAVE_CHUNKS, feature1, feature2, feature3....



        # Split into 60000ms chunks
        start = 0
        chunks = []
        for i in np.arange(60*1000, 1260*1000, 60*1000):
            chunk = trn_20_min_ms[(trn_20_min_ms > start) & (trn_20_min_ms <= i)]
            chunks.append(chunk)
            start += 60*1000

        # For every unit, chunks is a list of arrays. Every array contains the spikes times in every chunk of 60secs
        unit_all_chunks = {}
        n = 1
        for chunk in chunks:

            # Dict to store info about this chunk
            unit_chunk_info = {}

            # Compute inter spike interval (ISI) (1 order difference)
            ISI = np.diff(chunk)
            ISI = np.asarray(ISI, dtype='float64')

            # Remove outliers
            # We most commonly look for the 99.5th percentile, i.e. the point at which the probability that a random
            # event exceeds this value is 0.5%
            ISI = ISI[(ISI >= np.quantile(ISI, exclusion_quantile)) & (ISI <= np.quantile(ISI, 1 - exclusion_quantile))]
            # print(ISI)
            # plt.plot(ISI)
            # plt.show()

            # Fit log-normal distribution to ISI. This is known from previous work.
            from scipy.stats import lognorm
            from scipy import stats as scistats

            obs = len(ISI)
            ln_ISI = np.log(ISI)
            ln_ISI_sq = ln_ISI ** 2
            ln_ISI_sum = np.sum(ln_ISI)
            ln_ISI_sq_sum = np.sum(ln_ISI_sq)

            # Parameter estimation
            mu = np.mean(ln_ISI)
            sigma = np.sqrt(((obs * ln_ISI_sq_sum)-ln_ISI_sum**2)/(obs * (obs-1)))

            param = scistats.lognorm.fit(ISI)
            print("Log-normal distribution parameters : ", param)
            pdf_fitted = scistats.lognorm.pdf(ISI, *param)
            plt.plot(ISI, pdf_fitted, lw=5, label="Fitted Lognormal distribution")
            plt.legend()
            plt.show()

            # Compute Mean Firing Rate for this chunk (MFR) >> Spikes per second? (Hz)
            MFR = 1000./np.mean(ISI)
            MFR = round(MFR, 2)

            # Store current chunk MFR local dictionary for current unit
            unit_chunk_info.update([('chunk_{}'.format(n), MFR)])

            # Store all chunks MFR into global dictionary for current unit
            unit_all_chunks.update(unit_chunk_info)

            n += 1

        # print(unit_all_chunks)

        # import operator
        # sorted_d = sorted(unit_all_chunks.items(), key=operator.itemgetter(1))
        # pd.DataFrame(sorted_d, columns=['chunk', 'MFR']).set_index('chunk').plot(kind='bar')
        # plt.show()

        print(" ")
        print("-----------")

    else:
        # Why some units do not have info in the first 20 minutes, but until the end?
        # This case may be possible due to electrode drift
        print('Unit {} is discarded, as it has no spikes during first 20 minutes of recordings'.format(unit))



